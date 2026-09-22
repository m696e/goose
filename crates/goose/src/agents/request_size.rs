//! Responding to a provider that refuses a request for its size.
//!
//! Gateways that count encoded attachments instead of tokens reject a request
//! whose conversation is nowhere near the model's token window. Nothing needs
//! summarizing in that case — something in the payload has to get smaller — and
//! only the model knows what the payload is for, so goose describes the largest
//! contributors and lets the model decide what to drop or shrink.

use crate::conversation::message::{Message, MessageContent};
use crate::conversation::Conversation;

/// How many times goose hands the model a description of what made the request
/// too large before it stops asking and removes the offending content itself.
pub(super) const MAX_REQUEST_SIZE_ADVISORIES: usize = 2;

/// Content smaller than this is not worth naming as a culprit: a request is only
/// too large because of attachments, not because of a few lines of text.
const REPORTED_OFFENDER_BYTES: usize = 64 * 1024;

fn content_bytes(content: &[MessageContent]) -> usize {
    serde_json::to_vec(content)
        .map(|encoded| encoded.len())
        .unwrap_or(0)
}

fn agent_visible_by_size(conversation: &Conversation) -> Vec<(usize, &Message)> {
    let mut sized: Vec<(usize, &Message)> = conversation
        .messages()
        .iter()
        .filter(|message| message.is_agent_visible())
        .map(|message| (content_bytes(&message.content), message))
        .collect();
    sized.sort_by_key(|(bytes, _)| std::cmp::Reverse(*bytes));
    sized
}

/// Describes what to shrink, for the model's eyes only.
pub(super) fn oversized_request_message(
    conversation: &Conversation,
    provider_message: &str,
    limit_bytes: Option<usize>,
) -> String {
    let sized = agent_visible_by_size(conversation);
    let total: usize = sized.iter().map(|(bytes, _)| bytes).sum();
    let limit = match limit_bytes {
        Some(limit) => format!(
            "The provider accepts about {} per request",
            human_bytes(limit)
        ),
        None => "The provider rejects the request on its size".to_string(),
    };

    let mut offenders = String::new();
    for (bytes, message) in sized
        .iter()
        .take(3)
        .filter(|(bytes, _)| *bytes >= REPORTED_OFFENDER_BYTES)
    {
        offenders.push_str(&format!(
            "\n- {} ({}) starting: \"{}\"",
            human_bytes(*bytes),
            content_kinds(message),
            first_text(message).chars().take(160).collect::<String>()
        ));
    }

    format!(
        "Your last request was rejected as too large by the provider: \"{provider_message}\". \
         {limit}; that request carried about {}. This is a byte limit, not the model's token \
         window, so nothing needs summarizing — the content itself has to get smaller.{offenders} \
         \n\nFix it yourself and continue: re-read an image with a smaller `crop`, or downscale \
         it first (for example `magick input.jpg -resize 1568x1568 out.jpg`) and read the smaller \
         file; summarize or truncate oversized command output; or proceed without the content if \
         it is not needed. Do not resend the identical request.",
        human_bytes(total),
    )
}

fn content_kinds(message: &Message) -> String {
    let kinds: Vec<&str> = message
        .content
        .iter()
        .filter_map(|content| match content {
            MessageContent::Image(image) => Some(image.mime_type.as_str()),
            MessageContent::Document(document) => Some(document.mime_type.as_str()),
            _ => None,
        })
        .collect();
    if kinds.is_empty() {
        "text".to_string()
    } else {
        kinds.join(", ")
    }
}

fn first_text(message: &Message) -> String {
    message
        .content
        .iter()
        .filter_map(|content| content.as_text())
        .collect::<Vec<_>>()
        .join(" ")
}

pub(super) fn human_bytes(bytes: usize) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    const KIB: f64 = 1024.0;
    let bytes = bytes as f64;
    if bytes >= MIB {
        format!("{:.1} MiB", bytes / MIB)
    } else if bytes >= KIB {
        format!("{:.0} KiB", bytes / KIB)
    } else {
        format!("{bytes:.0} bytes")
    }
}
