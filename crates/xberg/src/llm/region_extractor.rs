//! Per-region VLM extraction for diagrams, dense tables, and complex layouts.
//!
//! When layout detection identifies a region as a figure, dense table, or
//! complex layout, this module crops the region's bounding box from the page
//! image and sends it to a VLM for precise extraction. The result is spliced
//! back into the markdown at the region's anchor position.
//!
//! This module is only compiled when `liter-llm` is available (non-Windows).

use super::vlm_ocr::vlm_ocr;
use crate::core::config::LlmConfig;
use crate::types::{LlmUsage, RegionKind};

/// Extract content from a pre-cropped image region using a VLM.
///
/// The caller is responsible for cropping the page image to the region's bounding
/// box before calling this function. The `image_bytes` parameter must contain the
/// raw bytes of the **cropped** region image (JPEG, PNG, WebP, etc.).
///
/// # Arguments
///
/// * `image_bytes` — Raw bytes of the **pre-cropped** region image.
/// * `image_mime` — MIME type of the image (`"image/png"`, `"image/jpeg"`, etc.).
/// * `region_kind` — Content type of the region, used to select the default prompt.
/// * `llm_config` — LLM provider and model configuration.
/// * `custom_prompt` — Optional override for the default per-region prompt template.
///
/// # Returns
///
/// Extracted Markdown text from the VLM, or an error if the VLM call fails.
///
/// # Errors
///
/// - [`crate::XbergError::Ocr`] if the VLM call fails or returns no content.
/// - [`crate::XbergError::MissingDependency`] if the liter-llm client cannot
///   be initialised.
///
/// # Example
///
/// ```rust,no_run
/// use xberg::llm::region_extractor::extract_region_with_vlm;
/// use xberg::{LlmConfig, RegionKind};
///
/// # async fn example() -> xberg::Result<()> {
/// let image_bytes: Vec<u8> = std::fs::read("cropped_figure.png")?;
/// let config = LlmConfig {
///     model: "openai/gpt-4o-mini".to_string(),
///     base_url: Some("http://localhost:9999".to_string()),
///     ..Default::default()
/// };
/// let markdown = extract_region_with_vlm(
///     &image_bytes,
///     "image/png",
///     RegionKind::Figure,
///     &config,
///     None,
/// )
/// .await?;
/// println!("Extracted: {markdown}");
/// # Ok(())
/// # }
/// ```
pub async fn extract_region_with_vlm(
    image_bytes: &[u8],
    image_mime: &str,
    region_kind: RegionKind,
    llm_config: &LlmConfig,
    custom_prompt: Option<&str>,
) -> crate::Result<String> {
    let (text, _usage) =
        extract_region_with_vlm_usage(image_bytes, image_mime, region_kind, llm_config, custom_prompt).await?;
    Ok(text)
}

/// Same as [`extract_region_with_vlm`], but also returns the [`LlmUsage`] data captured
/// from the underlying VLM call.
///
/// Callers that need to track token / cost data per call (for example the captioning
/// post-processor, which appends every call's usage to
/// [`ExtractedDocument::llm_usage`](crate::types::ExtractedDocument::llm_usage)) should
/// prefer this variant. The plain [`extract_region_with_vlm`] is kept for callers that
/// only care about the markdown output (PDF region splicing).
///
/// # Errors
///
/// Same as [`extract_region_with_vlm`].
#[cfg_attr(alef, alef(skip))]
pub async fn extract_region_with_vlm_usage(
    image_bytes: &[u8],
    image_mime: &str,
    region_kind: RegionKind,
    llm_config: &LlmConfig,
    custom_prompt: Option<&str>,
) -> crate::Result<(String, Option<LlmUsage>)> {
    let prompt = custom_prompt.unwrap_or_else(|| region_kind.default_prompt());

    vlm_ocr(image_bytes, image_mime, "eng", llm_config, Some(prompt), false).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_kind_default_prompt_figure() {
        let prompt = RegionKind::Figure.default_prompt();
        assert!(
            prompt.contains("diagram") || prompt.contains("figure"),
            "figure prompt must mention figures/diagrams; got: {prompt}"
        );
    }

    #[test]
    fn test_region_kind_default_prompt_dense_table() {
        let prompt = RegionKind::DenseTable.default_prompt();
        assert!(
            prompt.contains("Markdown") || prompt.contains("table"),
            "dense table prompt must mention Markdown/table; got: {prompt}"
        );
    }

    #[test]
    fn test_region_kind_default_prompt_complex_layout() {
        let prompt = RegionKind::ComplexLayout.default_prompt();
        assert!(
            prompt.contains("Markdown") || prompt.contains("reading order"),
            "complex layout prompt must mention Markdown; got: {prompt}"
        );
    }

    #[test]
    fn test_region_kind_prompts_are_non_empty() {
        for kind in [
            RegionKind::Figure,
            RegionKind::DenseTable,
            RegionKind::ComplexLayout,
            RegionKind::Caption,
        ] {
            assert!(
                !kind.default_prompt().is_empty(),
                "{kind:?} default prompt must not be empty"
            );
        }
    }

    #[test]
    fn test_region_kind_default_prompt_caption() {
        let prompt = RegionKind::Caption.default_prompt();
        assert!(
            prompt.contains("caption") || prompt.contains("alt text"),
            "caption prompt must mention captions/alt text; got: {prompt}"
        );
    }

    #[test]
    fn test_region_kind_equality() {
        assert_eq!(RegionKind::Figure, RegionKind::Figure);
        assert_ne!(RegionKind::Figure, RegionKind::DenseTable);
        assert_ne!(RegionKind::DenseTable, RegionKind::ComplexLayout);
    }
}
