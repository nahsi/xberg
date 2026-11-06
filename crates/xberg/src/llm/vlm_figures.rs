use std::borrow::Cow;
use std::io::Write;
use std::ops::Range;
use std::sync::LazyLock;

use image::ImageEncoder;
use regex::Regex;

use crate::core::config::ocr::VlmFigurePolicy;
use crate::extraction::image_decode::decode_standard_image_with_security_limits;
use crate::extractors::security::SecurityLimits;
use crate::types::internal::{ElementKind, InternalDocument, InternalElement};
use crate::types::{BoundingBox, ExtractedDocument, ExtractedImage, ProcessingWarning};

const MAX_DESCRIPTION_BYTES: usize = 256;

pub(super) const PROMPT_CLAUSE: &str = "\n\nMark photographs, illustrations, charts, maps and diagrams only, \
    at their position in reading order, using <figure bbox=\"x1,y1,x2,y2\" description=\"short description\"/>. \
    Give the top-left and bottom-right corners in integer coordinates from 0 to 1000 of the provided image. \
    Mark the picture only, not its caption. Transcribe captions as ordinary text. \
    Ignore reverse-side show-through and decorative icons. Give a short description of what is \
    visible, in the language of the page text; do not name a person unless the page names them. \
    Do not use a code fence.";

#[derive(Debug)]
struct FigureMarker {
    span: Range<usize>,
    bbox: Option<[u16; 4]>,
    description: Option<String>,
}

fn scan_markers(transcript: &str, limits: &SecurityLimits) -> Vec<FigureMarker> {
    static MARKER: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r#"<figure bbox="([0-9]+,[0-9]+,[0-9]+,[0-9]+)"(?: description="([^"<\r\n]*)")?[ \t]*/>"#).unwrap()
    });
    let mut markers = Vec::new();
    let marker_budget = limits.max_iterations.min(
        limits.max_content_size
            / (std::mem::size_of::<FigureMarker>()
                + 2 * std::mem::size_of::<InternalElement>()
                + std::mem::size_of::<ExtractedImage>()
                + MAX_DESCRIPTION_BYTES),
    );
    for capture in MARKER.captures_iter(transcript).take(marker_budget) {
        let token = capture.get(0).unwrap();
        let description = capture.get(2).map(|value| value.as_str());
        if token.len() > limits.max_entity_length
            || description.is_some_and(|value| value.len() > MAX_DESCRIPTION_BYTES.min(limits.max_entity_length))
        {
            continue;
        }
        if markers.try_reserve(1).is_err() {
            break;
        }
        let mut bbox = [0; 4];
        let valid = capture[1].split(',').zip(&mut bbox).all(|(value, slot)| {
            value
                .parse::<u16>()
                .ok()
                .filter(|value| *value <= 1000)
                .map(|value| *slot = value)
                .is_some()
        }) && bbox[0] < bbox[2]
            && bbox[1] < bbox[3];
        markers.push(FigureMarker {
            span: token.range(),
            bbox: valid.then_some(bbox),
            description: description.map(str::to_owned),
        });
    }
    markers
}

fn pixel_rect(bbox: [u16; 4], width: u32, height: u32) -> Option<[u32; 4]> {
    let [left, top, right, bottom] = bbox;
    if width == 0 || height == 0 || left >= right || top >= bottom {
        return None;
    }
    let floor = |coordinate, extent| u32::try_from(u64::from(coordinate) * u64::from(extent) / 1000).ok();
    let ceil = |coordinate, extent| u32::try_from((u64::from(coordinate) * u64::from(extent)).div_ceil(1000)).ok();
    Some([
        floor(left, width)?,
        floor(top, height)?,
        ceil(right, width)?,
        ceil(bottom, height)?,
    ])
}

fn warn(warnings: &mut Vec<ProcessingWarning>) {
    crate::core::diagnostics::push_warning_deduped(
        warnings,
        ProcessingWarning {
            source: Cow::Borrowed("vlm_figures"),
            message: Cow::Borrowed(
                "One or more VLM figure markers were dropped because their boxes, image limits or crop encoding were invalid.",
            ),
        },
    );
}

struct LimitedOutput {
    data: Vec<u8>,
    remaining: usize,
}

impl Write for LimitedOutput {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.remaining = self
            .remaining
            .checked_sub(bytes.len())
            .ok_or_else(|| std::io::Error::other("figure output limit"))?;
        self.data
            .try_reserve_exact(bytes.len())
            .map_err(std::io::Error::other)?;
        self.data.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn encode_crop(
    image: &image::DynamicImage,
    rect: [u32; 4],
    policy: &VlmFigurePolicy,
    available: usize,
) -> Option<(Vec<u8>, u32, u32)> {
    let [left, top, right, bottom] = rect;
    let (width, height) = (right - left, bottom - top);
    let pixels = u64::from(width).checked_mul(u64::from(height))?;
    let bytes_per_pixel = u64::from(image.color().bytes_per_pixel());
    let crop_bytes = pixels.checked_mul(bytes_per_pixel)?;
    let encoder_bytes = pixels
        .checked_mul(crate::image::dpi::OCR_PNG_ENCODE_BYTES_PER_PIXEL.max(bytes_per_pixel))?
        .checked_add(crate::image::dpi::OCR_PNG_ENCODE_FIXED_BYTES)?;
    let resizing = policy.max_dimension > 0 && width.max(height) > policy.max_dimension;
    // Resize retains the source crop beside a floating-point RGBA intermediate and its output. ~keep
    let resize_bytes = if resizing {
        pixels
            .checked_mul(std::mem::size_of::<[f32; 4]>() as u64)?
            .checked_add(crop_bytes)?
    } else {
        0
    };
    let working = usize::try_from(crop_bytes.checked_add(encoder_bytes)?.checked_add(resize_bytes)?).ok()?;
    let output_limit = available.checked_sub(working)?;
    let mut crop = image.crop_imm(left, top, width, height);
    if resizing {
        crop = crop.resize(
            policy.max_dimension,
            policy.max_dimension,
            image::imageops::FilterType::Triangle,
        );
    }
    let mut output = LimitedOutput {
        data: Vec::new(),
        remaining: output_limit,
    };
    image::codecs::png::PngEncoder::new(&mut output)
        .write_image(crop.as_bytes(), crop.width(), crop.height(), crop.color().into())
        .ok()?;
    Some((output.data, crop.width(), crop.height()))
}

fn materialization_bytes(
    transcript_capacity: usize,
    sent_bytes: usize,
    markers: &Vec<FigureMarker>,
    policy: &VlmFigurePolicy,
) -> Option<usize> {
    // The response, cleaned transcript and ordered text coexist; the borrowed raster and
    // blocking worker's copy coexist too. Charge both before copying or decoding. ~keep
    transcript_capacity
        .checked_mul(3)
        .and_then(|bytes| bytes.checked_add(sent_bytes.checked_mul(2)?))
        .and_then(|bytes| bytes.checked_add(policy.retained_bytes))
        .and_then(|bytes| bytes.checked_add(markers.capacity().checked_mul(std::mem::size_of::<FigureMarker>())?))
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<InternalElement>()))
        .and_then(|bytes| {
            bytes.checked_add(markers.len().checked_mul(
                std::mem::size_of::<InternalElement>() * 2
                    + std::mem::size_of::<ExtractedImage>()
                    + MAX_DESCRIPTION_BYTES,
            )?)
        })
}

pub(super) async fn materialize(
    transcript: String,
    sent: &[u8],
    policy: VlmFigurePolicy,
    limits: SecurityLimits,
) -> crate::Result<ExtractedDocument> {
    let markers = scan_markers(&transcript, &limits);
    if markers.is_empty() {
        return Ok(ExtractedDocument {
            content: transcript,
            ..Default::default()
        });
    }
    let mut raster = Vec::new();
    let copy_fits = materialization_bytes(transcript.capacity(), sent.len(), &markers, &policy)
        .is_some_and(|bytes| bytes <= limits.max_content_size);
    if copy_fits
        && policy.max_per_page != Some(0)
        && markers.iter().any(|marker| marker.bbox.is_some())
        && raster.try_reserve_exact(sent.len()).is_ok()
    {
        raster.extend_from_slice(sent);
    }
    tokio::task::spawn_blocking(move || materialize_crops(transcript, &raster, &policy, &limits, markers))
        .await
        .map_err(|error| crate::XbergError::ocr(format!("VLM figure processing failed: {error}")))
}

fn materialize_crops(
    transcript: String,
    sent: &[u8],
    policy: &VlmFigurePolicy,
    limits: &SecurityLimits,
    markers: Vec<FigureMarker>,
) -> ExtractedDocument {
    let mut result = ExtractedDocument::default();
    let mut doc = InternalDocument::new("ocr");
    let storage = materialization_bytes(transcript.capacity(), sent.len(), &markers, policy);
    let available = storage.and_then(|storage| limits.max_content_size.checked_sub(storage));
    let decoded = if markers.iter().any(|marker| marker.bbox.is_some()) && policy.max_per_page != Some(0) {
        available.and_then(|available| {
            let decode_limits = SecurityLimits {
                max_content_size: available,
                ..limits.clone()
            };
            decode_standard_image_with_security_limits(sent, &decode_limits).ok()
        })
    } else {
        None
    };
    let mut remaining = available
        .and_then(|available| {
            decoded
                .as_ref()
                .map(|image| available.saturating_sub(image.as_bytes().len()))
        })
        .unwrap_or(0);
    let mut cleaned = String::new();
    if cleaned.try_reserve_exact(transcript.len()).is_err()
        || doc
            .elements
            .try_reserve_exact(markers.len().saturating_mul(2).saturating_add(1))
            .is_err()
        || doc.images.try_reserve_exact(markers.len()).is_err()
        || doc.image_ocr_skip_indices.try_reserve(markers.len()).is_err()
    {
        warn(&mut result.processing_warnings);
        result.content = transcript;
        return result;
    }
    let mut cursor = 0;
    let mut text_start = 0;
    for marker in markers {
        let text = &transcript[cursor..marker.span.start];
        cleaned.push_str(text);
        cursor = marker.span.end;
        let crop = (|| {
            if policy.max_per_page.is_some_and(|max| doc.images.len() >= max as usize) {
                return None;
            }
            let image = decoded.as_ref()?;
            let rect = pixel_rect(marker.bbox?, image.width(), image.height())?;
            let image_index = u32::try_from(doc.images.len()).ok()?;
            let (data, width, height) = encode_crop(image, rect, policy, remaining)?;
            remaining = remaining.checked_sub(data.capacity())?;
            let text = &cleaned[text_start..];
            if !text.trim().is_empty() {
                doc.push_element(InternalElement::text(ElementKind::Paragraph, text, 0));
            }
            text_start = cleaned.len();
            let mut element = InternalElement::text(ElementKind::Image { image_index }, "", 0);
            element.bbox = Some(BoundingBox {
                x0: f64::from(rect[0]),
                y0: f64::from(rect[1]),
                x1: f64::from(rect[2]),
                y1: f64::from(rect[3]),
            });
            doc.push_element(element);
            doc.images.push(ExtractedImage {
                data: data.into(),
                format: Cow::Borrowed("png"),
                image_index,
                width: Some(width),
                height: Some(height),
                description: Some(
                    marker
                        .description
                        .filter(|description| !description.is_empty())
                        .unwrap_or_else(|| "Figure".into()),
                ),
                ..Default::default()
            });
            doc.image_ocr_skip_indices.insert(image_index);
            Some(())
        })();
        if crop.is_none() {
            warn(&mut result.processing_warnings);
        }
    }
    let tail = &transcript[cursor..];
    cleaned.push_str(tail);
    let text = &cleaned[text_start..];
    if !text.trim().is_empty() {
        doc.push_element(InternalElement::text(ElementKind::Paragraph, text, 0));
    }
    result.content = cleaned;
    if !doc.images.is_empty() {
        result.images = Some(std::mem::take(&mut doc.images));
        result.ocr_internal_document = Some(doc);
        result.ocr_ordered_contribution = true;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_marker_grammar_preserves_all_other_bytes() {
        let limits = SecurityLimits {
            max_entity_length: 64,
            ..Default::default()
        };
        for (marker, recognized, valid) in [
            ("<figure bbox=\"10,20,30,40\" description=\"Still\"/>", true, true),
            ("<figure bbox=\"0,0,1000,1000\"/>", true, true),
            ("<figure bbox=\"0,0,1001,1000\"/>", true, false),
            ("<figure bbox=\"30,20,10,40\"/>", true, false),
            ("<figure bbox=\"10,20,10,40\"/>", true, false),
            ("<figure bbox=\"NaN,20,30,40\"/>", false, false),
            ("<figure bbox=\"10,20,30\"/>", false, false),
            (
                "<figure bbox=\"10,20,30,40\" description=\"This description deliberately exceeds the entity limit of sixty-four bytes.\"/>",
                false,
                false,
            ),
            ("<figure bbox=\"10,20,30,40\"", false, false),
        ] {
            let input = format!("Πριν\n{marker}\nAfter");
            let markers = scan_markers(&input, &limits);
            assert_eq!(markers.len(), usize::from(recognized), "{marker}");
            let cleaned = if let Some(found) = markers.first() {
                assert_eq!(found.bbox.is_some(), valid, "{marker}");
                assert_eq!(&input[found.span.clone()], marker);
                format!("{}{}", &input[..found.span.start], &input[found.span.end..])
            } else {
                input.clone()
            };
            assert_eq!(
                cleaned,
                if recognized {
                    "Πριν\n\nAfter".to_owned()
                } else {
                    input
                }
            );
            if marker.contains("Still") {
                assert_eq!(markers[0].description.as_deref(), Some("Still"));
            }
        }
    }

    #[test]
    fn pixel_rect_rounds_outwards_without_overflow() {
        for (bbox, width, height, expected) in [
            ([1, 1, 999, 999], 31, 47, Some([0, 0, 31, 47])),
            ([500, 500, 1000, 1000], 31, 47, Some([15, 23, 31, 47])),
            ([0, 0, 1000, 1000], u32::MAX, u32::MAX, Some([0, 0, u32::MAX, u32::MAX])),
            ([500, 0, 500, 1000], 31, 47, None),
            ([0, 0, 1000, 1000], 0, 47, None),
        ] {
            assert_eq!(pixel_rect(bbox, width, height), expected);
        }
    }
}
