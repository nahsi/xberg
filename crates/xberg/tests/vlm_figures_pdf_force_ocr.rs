#![cfg(all(feature = "pdf", feature = "liter-llm", not(target_arch = "wasm32")))]

use base64::Engine;
use lopdf::{Document, Object, Stream, dictionary};
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};
use xberg::core::config::{ExtractionConfig, ImageExtractionConfig, LlmConfig, OcrConfig, OutputFormat};
use xberg::types::NodeContent;

fn pdf(rotation: i64) -> Vec<u8> {
    let mut pdf = Document::with_version("1.5");
    let pages = pdf.new_object_id();
    let mut kids = Vec::new();
    for (width, height, rotate) in [(100, 200, 0), (200, 300, rotation)] {
        let raster = image::RgbImage::from_fn(16, 16, |x, y| image::Rgb([(x * 15) as u8, (y * 15) as u8, 80]));
        let image = pdf.add_object(Stream::new(
            dictionary! {
                "Type" => "XObject", "Subtype" => "Image", "Width" => 16, "Height" => 16,
                "ColorSpace" => "DeviceRGB", "BitsPerComponent" => 8,
            },
            raster.into_raw(),
        ));
        let contents = pdf.add_object(Stream::new(
            dictionary! {},
            format!("q {width} 0 0 {height} 0 0 cm /Photo Do Q").into_bytes(),
        ));
        let page = pdf.add_object(dictionary! {
            "Type" => "Page", "Parent" => pages,
            "MediaBox" => vec![0.into(), 0.into(), width.into(), height.into()], "Rotate" => rotate,
            "Resources" => dictionary! { "XObject" => dictionary! { "Photo" => image } }, "Contents" => contents,
        });
        kids.push(Object::Reference(page));
    }
    pdf.objects.insert(
        pages,
        Object::Dictionary(dictionary! { "Type" => "Pages", "Kids" => kids, "Count" => 2 }),
    );
    let catalog = pdf.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages });
    pdf.trailer.set("Root", catalog);
    let mut bytes = Vec::new();
    pdf.save_to(&mut bytes).unwrap();
    bytes
}

fn request_image(request: &Request) -> image::RgbImage {
    let body: Value = request.body_json().unwrap();
    let url = body["messages"][0]["content"][1]["image_url"]["url"].as_str().unwrap();
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(url.split_once(',').unwrap().1)
        .unwrap();
    image::load_from_memory(&bytes).unwrap().to_rgb8()
}

#[derive(Clone, Copy)]
enum PageResponse {
    Figures,
    Text(&'static str),
    NativeRecovery { fail_page: bool },
}

async fn extract_case(
    rotation: i64,
    figure_only: bool,
    figures: bool,
) -> (xberg::types::ExtractedDocument, image::RgbImage) {
    extract_transcript_case(rotation, figure_only, figures, PageResponse::Figures).await
}

async fn extract_transcript_case(
    rotation: i64,
    figure_only: bool,
    figures: bool,
    response: PageResponse,
) -> (xberg::types::ExtractedDocument, image::RgbImage) {
    let server = MockServer::start().await;
    let expected_calls = if matches!(response, PageResponse::NativeRecovery { .. }) {
        6
    } else {
        2
    };
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(move |request: &Request| {
            let body: Value = request.body_json().unwrap();
            let page_read = body["messages"][0]["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("<figure bbox=");
            if page_read && matches!(response, PageResponse::NativeRecovery { fail_page: true }) {
                return ResponseTemplate::new(500);
            }
            let image = request_image(request);
            let first = (f64::from(image.width()) / f64::from(image.height()) - 0.5).abs() < 0.01;
            let transcript = match response {
                PageResponse::Text(text) => text,
                PageResponse::NativeRecovery { .. } if page_read => "",
                PageResponse::NativeRecovery { .. } => "Recovered text from the native scan.",
                PageResponse::Figures if first => "The first page contains ordinary transcribed text.",
                PageResponse::Figures if !figures => "The second page contains ordinary transcribed text.",
                PageResponse::Figures if figure_only => "<figure bbox=\"100,200,600,800\" description=\"Photo\"/>",
                PageResponse::Figures => concat!(
                    "Before the photograph.\n",
                    "<figure bbox=\"100,200,600,800\" description=\"Photo\"/>\n",
                    "Caption after the photograph.",
                ),
            };
            ResponseTemplate::new(200).set_body_json(json!({
                "id": "pdf-figures",
                "object": "chat.completion",
                "created": 0,
                "model": "figure-test",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": transcript},
                    "finish_reason": "stop",
                }],
            }))
        })
        .expect(expected_calls as u64)
        .mount(&server)
        .await;
    let config = ExtractionConfig {
        force_ocr: true,
        use_cache: false,
        include_document_structure: true,
        output_format: OutputFormat::Markdown,
        images: Some(ImageExtractionConfig {
            extract_images: true,
            include_page_rasters: true,
            auto_adjust_dpi: false,
            ..Default::default()
        }),
        ocr: Some(OcrConfig {
            backend: "vlm".into(),
            backend_options: Some(json!({"vlm_figures": figures})),
            vlm_config: Some(LlmConfig {
                model: "openai/figure-test".into(),
                api_key: Some("test-key".into()),
                base_url: Some(server.uri()),
                load_env: Some(false),
                timeout_secs: Some(10),
                max_retries: Some(0),
                ..Default::default()
            }),
            ..Default::default()
        }),
        ..Default::default()
    };
    let output = xberg::extract(
        xberg::ExtractInput::from_bytes(pdf(rotation), "application/pdf", None),
        &config,
    )
    .await
    .unwrap();
    assert!(output.errors.is_empty(), "{:?}", output.errors);
    let result = output.results.into_iter().next().unwrap();
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), expected_calls);
    let sent = requests
        .iter()
        .map(request_image)
        .find(|image| (f64::from(image.width()) / f64::from(image.height()) - 0.5).abs() >= 0.01)
        .unwrap();
    (result, sent)
}

fn assert_figure(result: &xberg::types::ExtractedDocument, sent: &image::RgbImage, rotation: i64) {
    let images = result.images.as_ref().unwrap();
    let crop = images
        .iter()
        .find(|image| image.description.as_deref() == Some("Photo"))
        .expect("crop survives native inventory attachment");
    assert_eq!(crop.page_number, Some(2));
    assert_eq!(images.len(), 1, "VLM-read pages return only their cropped figure");
    assert_eq!(images[crop.image_index as usize].data, crop.data);
    let left = sent.width() / 10;
    let top = sent.height() / 5;
    let right = (u64::from(sent.width()) * 600).div_ceil(1000) as u32;
    let bottom = (u64::from(sent.height()) * 800).div_ceil(1000) as u32;
    assert_eq!(
        image::load_from_memory(&crop.data).unwrap().to_rgb8(),
        image::imageops::crop_imm(sent, left, top, right - left, bottom - top).to_image()
    );
    let bbox = crop.bounding_box.expect("PDF-point crop placement");
    let expected = if rotation == 90 {
        [40.0, 30.0, 160.0, 180.0]
    } else {
        [20.0, 60.0, 120.0, 240.0]
    };
    for (actual, expected) in [bbox.x0, bbox.y0, bbox.x1, bbox.y1].into_iter().zip(expected) {
        assert!((actual - expected).abs() < 1.0, "{bbox:?} differs from {expected}");
    }
    let structure = result.document.as_ref().unwrap();
    let refs: Vec<_> = structure
        .nodes
        .iter()
        .filter_map(|node| match node.content {
            NodeContent::Image {
                image_index: Some(index),
                ..
            } => Some((node.page, index)),
            _ => None,
        })
        .collect();
    assert_eq!(
        refs.iter().filter(|(page, _)| *page == Some(2)).count(),
        1,
        "no duplicate native link on crop page"
    );
    assert!(refs.contains(&(Some(2), crop.image_index)));
    assert!(
        refs.iter().all(|(page, _)| *page != Some(1)),
        "a VLM-read page without figures has no native or page-raster link"
    );
    assert!(
        images
            .iter()
            .all(|image| image.image_kind != Some(xberg::types::ImageKind::PageRaster))
    );
    assert!(
        result
            .content
            .contains(&format!("![Photo](image_{}.png)", crop.image_index)),
        "{}",
        result.content
    );
}

#[tokio::test]
async fn forced_pdf_keeps_order_assets_and_per_page_placeholders() {
    let (result, sent) = extract_case(0, false, true).await;
    assert_figure(&result, &sent, 0);
    assert!(result.content.find("Before the photograph").unwrap() < result.content.find("![Photo]").unwrap());
    assert!(result.content.find("![Photo]").unwrap() < result.content.find("Caption after").unwrap());
}

#[tokio::test]
async fn forced_pdf_retains_a_figure_only_page() {
    let (result, sent) = extract_case(0, true, true).await;
    assert_figure(&result, &sent, 0);
    assert!(
        result
            .images
            .as_ref()
            .unwrap()
            .iter()
            .all(|image| image.description.as_deref() == Some("Photo"))
    );
}

#[tokio::test]
async fn rotated_forced_pdf_keeps_upright_pixels_and_pdf_points() {
    let (result, sent) = extract_case(90, false, true).await;
    assert_figure(&result, &sent, 90);
}

#[tokio::test]
async fn forced_pdf_without_figures_keeps_native_links() {
    let (result, _) = extract_case(0, false, false).await;
    let images = result.images.as_ref().unwrap();
    let refs: Vec<_> = result
        .document
        .as_ref()
        .unwrap()
        .nodes
        .iter()
        .filter_map(|node| {
            if let NodeContent::Image {
                image_index: Some(index),
                ..
            } = node.content
            {
                Some(index)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(images.len(), 4, "two native scans and two page rasters");
    for image in images {
        assert!(
            refs.contains(&image.image_index),
            "figure mode off keeps image {} linked",
            image.image_index
        );
    }
}

#[tokio::test]
async fn marker_free_figure_mode_keeps_crlf_list_markdown() {
    const TRANSCRIPT: &str = "A list of supplies:\r\n\r\n- First item\r\n- Second item\r\n\r\nA final paragraph.";
    let (plain, _) = extract_transcript_case(0, false, false, PageResponse::Text(TRANSCRIPT)).await;
    let (figures, _) = extract_transcript_case(0, false, true, PageResponse::Text(TRANSCRIPT)).await;
    let plain_text = plain.content.split("![").next().unwrap().trim();
    assert_eq!(figures.content.trim(), plain_text);
    assert!(
        figures.images.as_ref().is_none_or(Vec::is_empty),
        "marker-free VLM pages retain no scans"
    );
}

#[tokio::test]
async fn fallback_and_failed_pages_keep_native_scans() {
    for fail_page in [false, true] {
        let (result, _) = extract_transcript_case(0, false, true, PageResponse::NativeRecovery { fail_page }).await;
        let images = result.images.as_ref().unwrap();
        let native: Vec<_> = images
            .iter()
            .filter(|image| image.image_kind != Some(xberg::types::ImageKind::PageRaster))
            .collect();
        assert_eq!(native.len(), 4, "two recovered source images and two native scans");
        assert!(native.iter().all(|image| !image.data.is_empty()));
        assert!(native.iter().any(|image| image.page_number == Some(1)));
        assert!(native.iter().any(|image| image.page_number == Some(2)));
        assert!(result.content.contains("Recovered text from the native scan."));
    }
}
