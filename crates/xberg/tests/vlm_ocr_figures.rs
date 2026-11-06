#![cfg(all(feature = "liter-llm", feature = "tokio-runtime", not(target_arch = "wasm32")))]

use base64::Engine;
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};
use xberg::core::config::{LlmConfig, OcrConfig};
use xberg::llm::vlm_ocr::VlmOcrBackend;
use xberg::plugins::OcrBackend;

fn page_image() -> Vec<u8> {
    encoded_image(31, 47)
}

fn encoded_image(width: u32, height: u32) -> Vec<u8> {
    let image = image::RgbImage::from_fn(width, height, |x, y| {
        image::Rgb([(x * 7) as u8, (y * 5) as u8, ((x + y) * 3) as u8])
    });
    let mut encoded = std::io::Cursor::new(Vec::new());
    image.write_to(&mut encoded, image::ImageFormat::Png).unwrap();
    encoded.into_inner()
}

fn ocr_config(server: &MockServer) -> OcrConfig {
    OcrConfig {
        backend: "vlm".into(),
        vlm_config: Some(LlmConfig {
            model: "openai/figure-test".into(),
            api_key: Some("test-key".into()),
            base_url: Some(server.uri()),
            load_env: Some(false),
            timeout_secs: Some(10),
            max_retries: Some(0),
            ..Default::default()
        }),
        backend_options: Some(json!({"vlm_figure_policy": {"max_per_page": null, "max_dimension": 4096}})),
        ..Default::default()
    }
}

async fn serve(server: &MockServer, transcript: &str) {
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "figures", "object": "chat.completion", "created": 0, "model": "figure-test",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": transcript}, "finish_reason": "stop"}]
        })))
        .expect(1)
        .mount(server)
        .await;
}

#[tokio::test]
async fn stamped_policy_materializes_exact_crops_and_appends_to_custom_prompt() {
    let server = MockServer::start().await;
    serve(&server, "Πριν\n<figure bbox=\"0,0,500,500\" description=\"Still\"/>\nCaption\n<figure bbox=\"500,500,1000,1000\"/>\nAfter").await;
    let mut config = ocr_config(&server);
    config.vlm_prompt = Some("Transcribe {{ language }} exactly.".into());
    let bytes = page_image();
    let result = VlmOcrBackend.process_image(&bytes, &config).await.unwrap();
    assert_eq!(result.content, "Πριν\n\nCaption\n\nAfter");
    let crops = result.images.unwrap();
    assert_eq!(crops.len(), 2);
    let source = image::load_from_memory(&bytes).unwrap().to_rgb8();
    for (crop, (x, y, w, h), alt) in crops
        .iter()
        .zip([(0, 0, 16, 24), (15, 23, 16, 24)])
        .zip(["Still", "Figure"])
        .map(|((c, r), a)| (c, r, a))
    {
        assert_eq!(crop.description.as_deref(), Some(alt));
        assert_eq!((crop.width, crop.height), (Some(w), Some(h)));
        assert_eq!(
            image::load_from_memory(&crop.data).unwrap().to_rgb8(),
            image::imageops::crop_imm(&source, x, y, w, h).to_image()
        );
        assert!(crop.bounding_box.is_none());
    }
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let body: Value = requests[0].body_json().unwrap();
    let parts = &body["messages"][0]["content"];
    let prompt = parts[0]["text"].as_str().unwrap();
    assert!(prompt.starts_with("Transcribe eng exactly."));
    assert!(prompt.contains("<figure bbox="));
    assert!(prompt.contains("reverse-side show-through"));
    let payload = parts[1]["image_url"]["url"]
        .as_str()
        .unwrap()
        .split_once(',')
        .unwrap()
        .1;
    assert_eq!(
        base64::engine::general_purpose::STANDARD.decode(payload).unwrap(),
        bytes
    );
}

#[tokio::test]
async fn user_intent_without_policy_preserves_prompt_and_marker_bytes() {
    let server = MockServer::start().await;
    let transcript = "Before\n<figure bbox=\"0,0,1000,1000\"/>\nAfter";
    serve(&server, transcript).await;
    let mut config = ocr_config(&server);
    config.backend_options = Some(json!({"vlm_figures": true}));
    config.vlm_prompt = Some("Literal transcription.".into());
    let result = VlmOcrBackend.process_image(&page_image(), &config).await.unwrap();
    assert_eq!(result.content, transcript);
    assert!(result.images.is_none());
    let requests = server.received_requests().await.unwrap();
    let body: Value = requests[0].body_json().unwrap();
    assert_eq!(body["messages"][0]["content"][0]["text"], "Literal transcription.");
}

#[tokio::test]
async fn count_limit_drops_only_the_excess_marker() {
    let server = MockServer::start().await;
    serve(
        &server,
        "Before<figure bbox=\"0,0,100,100\"/>Caption<figure bbox=\"0,0,1000,1000\"/>After",
    )
    .await;
    let mut config = ocr_config(&server);
    config.backend_options = Some(json!({"vlm_figure_policy": {"max_per_page": 1, "max_dimension": 4096}}));
    let result = VlmOcrBackend.process_image(&page_image(), &config).await.unwrap();
    assert_eq!(result.content, "BeforeCaptionAfter");
    assert_eq!(result.images.unwrap().len(), 1);
    assert_eq!(result.processing_warnings.len(), 1);
}

#[tokio::test]
async fn content_limit_keeps_a_small_sibling_and_warns_once() {
    let server = MockServer::start().await;
    serve(&server, "Before<figure bbox=\"0,0,100,100\"/>Caption<figure bbox=\"0,0,1000,1000\"/><figure bbox=\"0,0,1000,1000\"/>After").await;
    let mut config = ocr_config(&server);
    config.security_limits = Some(xberg::extractors::security::SecurityLimits {
        max_content_size: 10_000_000,
        ..Default::default()
    });
    let result = VlmOcrBackend
        .process_image(&encoded_image(1000, 1000), &config)
        .await
        .unwrap();
    assert_eq!(result.content, "BeforeCaptionAfter");
    assert_eq!(result.images.unwrap().len(), 1);
    assert_eq!(result.processing_warnings.len(), 1);
}
