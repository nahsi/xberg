#![cfg(all(feature = "liter-llm", not(target_arch = "wasm32")))]

use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};
use xberg::core::config::{ExtractionConfig, ImageExtractionConfig, LlmConfig, OcrConfig, OutputFormat};
use xberg::types::NodeContent;

fn png() -> Vec<u8> {
    let image = image::RgbImage::from_fn(31, 47, |x, y| image::Rgb([(x * 7) as u8, (y * 5) as u8, 100]));
    let mut output = std::io::Cursor::new(Vec::new());
    image.write_to(&mut output, image::ImageFormat::Png).unwrap();
    output.into_inner()
}

#[tokio::test]
async fn standalone_crops_keep_order_and_do_not_alias_the_source_image() {
    for (extract_images, inject_placeholders) in [(true, true), (true, false), (false, true)] {
        let server = MockServer::start().await;
        let transcript = "Before\n<figure bbox=\"0,0,500,500\" description=\"Still\"/>\nCaption\n<figure bbox=\"500,500,1000,1000\"/>\nAfter";
        Mock::given(method("POST")).and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "image-figures", "object": "chat.completion", "created": 0, "model": "figure-test",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": transcript}, "finish_reason": "stop"}]
            }))).expect(1).mount(&server).await;
        let config = ExtractionConfig {
            use_cache: false,
            include_document_structure: true,
            output_format: OutputFormat::Markdown,
            images: Some(ImageExtractionConfig {
                extract_images,
                inject_placeholders,
                target_dpi: 72,
                min_dpi: 72,
                max_dpi: 72,
                auto_adjust_dpi: false,
                ..Default::default()
            }),
            ocr: Some(OcrConfig {
                backend: "vlm".into(),
                backend_options: Some(json!({"vlm_figures": true})),
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
        assert_eq!(config.wants_own_bytes_in_result(), extract_images);
        let source = png();
        let output = xberg::extract(
            xberg::ExtractInput::from_bytes(source.clone(), "image/png", None),
            &config,
        )
        .await
        .unwrap();
        assert!(output.errors.is_empty(), "{:?}", output.errors);
        let result = &output.results[0];
        if !extract_images {
            assert!(result.images.as_ref().is_none_or(Vec::is_empty));
            assert!(
                result.content.contains("figure bbox"),
                "the extraction veto leaves the response literal"
            );
            continue;
        }
        let images = result.images.as_ref().unwrap();
        assert_eq!(images.len(), 3, "two crops and the source inventory");
        assert_eq!(images[0].description.as_deref(), Some("Still"));
        assert_eq!(images[1].description.as_deref(), Some("Figure"));
        assert_eq!(images[2].image_index, 2, "source is the actual tail index");
        assert_eq!(images[2].data.as_ref(), source);
        assert!(images.iter().all(|image| image.bounding_box.is_none()));
        assert!(
            !result.content.contains("image_2."),
            "source bytes do not add a duplicate page-flow link"
        );
        let nodes = &result.document.as_ref().unwrap().nodes;
        let image_indices: Vec<_> = nodes
            .iter()
            .filter_map(|node| match node.content {
                NodeContent::Image { image_index, .. } => image_index,
                _ => None,
            })
            .collect();
        if inject_placeholders {
            assert_eq!(image_indices, [0, 1]);
            let expected = [
                "Before",
                "![Still](image_0.png)",
                "Caption",
                "![Figure](image_1.png)",
                "After",
            ];
            let positions: Vec<_> = expected
                .iter()
                .map(|text| result.content.find(text).expect(text))
                .collect();
            assert!(positions.windows(2).all(|pair| pair[0] < pair[1]), "{}", result.content);
        } else {
            assert!(image_indices.is_empty());
            assert!(!result.content.contains("!["));
        }
    }
}
