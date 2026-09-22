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

/// How many times goose removes the largest messages when the model has not
/// managed to shrink the request itself. Bounded so a request that cannot fit
/// however much is removed still reaches the user instead of looping.
pub(super) const MAX_REQUEST_SIZE_EVICTIONS: usize = 2;

/// Marks an eviction notice, so later decisions can tell how many times the
/// conversation has already been trimmed and whether this was goose's own doing.
pub(super) const EVICTION_NOTICE_PREFIX: &str = "goose removed";

/// Openings of the messages goose itself writes to steer the model, which are
/// never what makes a request too large.
const GOOSE_INSTRUCTION_PREFIXES: [&str; 2] = [
    "Your last request was rejected as too large",
    EVICTION_NOTICE_PREFIX,
];

/// Content smaller than this is not worth naming as a culprit: a request is only
/// too large because of attachments, not because of a few lines of text.
const REPORTED_OFFENDER_BYTES: usize = 64 * 1024;

/// How much of the request to clear out once the model has not shrunk it, as a
/// multiple of the largest offender's size, so that a second refusal does not
/// follow immediately.
const EVICTION_HEADROOM: f64 = 1.5;

fn content_bytes(content: &[MessageContent]) -> usize {
    serde_json::to_vec(content)
        .map(|encoded| encoded.len())
        .unwrap_or(0)
}

/// Agent-visible messages by serialized size, largest first, each with its
/// position in the conversation so callers can reason about recency.
fn agent_visible_by_size(conversation: &Conversation) -> Vec<(usize, usize, &Message)> {
    let mut sized: Vec<(usize, usize, &Message)> = conversation
        .messages()
        .iter()
        .enumerate()
        .filter(|(_, message)| message.is_agent_visible())
        .map(|(index, message)| (content_bytes(&message.content), index, message))
        .collect();
    sized.sort_by_key(|(bytes, _, _)| std::cmp::Reverse(*bytes));
    sized
}

/// Describes what to shrink, for the model's eyes only.
pub(super) fn oversized_request_message(
    conversation: &Conversation,
    provider_message: &str,
    limit_bytes: Option<usize>,
) -> String {
    let sized = agent_visible_by_size(conversation);
    let total: usize = sized.iter().map(|(bytes, _, _)| bytes).sum();
    let limit = match limit_bytes {
        Some(limit) => format!(
            "The provider accepts about {} per request",
            human_bytes(limit)
        ),
        None => "The provider rejects the request on its size".to_string(),
    };

    let mut offenders = String::new();
    for (bytes, _, message) in sized
        .iter()
        .take(3)
        .filter(|(bytes, _, _)| *bytes >= REPORTED_OFFENDER_BYTES)
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

/// A message taken out of the request to make it fit.
pub(super) struct EvictedMessage {
    pub id: String,
    pub bytes: usize,
    pub kinds: String,
    pub starts_with: String,
}

/// The agent-visible messages to take out of the request: the
/// largest first, until at least `EVICTION_HEADROOM` times the biggest offender
/// is accounted for. Messages carrying a tool request or response pull their
/// partner out with them, so a pair never leaves half behind.
pub(super) fn messages_to_evict(conversation: &Conversation) -> Vec<EvictedMessage> {
    let sized = agent_visible_by_size(conversation);
    let most_recent = sized.iter().map(|(_, index, _)| *index).max();
    let evictable: Vec<&(usize, usize, &Message)> = sized
        .iter()
        .filter(|(_, index, message)| Some(*index) != most_recent && !is_goose_instruction(message))
        .collect();
    let Some((largest, _, _)) = evictable
        .iter()
        .find(|(bytes, _, _)| *bytes >= REPORTED_OFFENDER_BYTES)
    else {
        return Vec::new();
    };
    let target = (*largest as f64 * EVICTION_HEADROOM) as usize;
    // Removing a trickle of small messages costs context without shrinking the
    // request, so only content comparable to the offender goes.
    let worth_removing = largest / 2;

    let mut removed = 0usize;
    let mut chosen: Vec<&Message> = Vec::new();
    for (bytes, _, message) in &evictable {
        if removed >= target || (*bytes < worth_removing && !chosen.is_empty()) {
            break;
        }
        removed += bytes;
        chosen.push(message);
    }

    let mut evicted: Vec<EvictedMessage> = chosen
        .iter()
        .filter_map(|message| evicted_message(message))
        .collect();
    let partners: Vec<String> = chosen
        .iter()
        .flat_map(|message| {
            message
                .get_tool_request_ids()
                .into_iter()
                .chain(message.get_tool_response_ids())
        })
        .map(str::to_string)
        .collect();
    for (_, _, message) in &evictable {
        let already = evicted
            .iter()
            .any(|entry| Some(&entry.id) == message.id.as_ref());
        if already {
            continue;
        }
        let paired = message.content.iter().any(|content| match content {
            MessageContent::ToolRequest(request) => partners.contains(&request.id),
            MessageContent::ToolResponse(response) => partners.contains(&response.id),
            _ => false,
        });
        if paired {
            evicted.extend(evicted_message(message));
        }
    }
    evicted
}

fn evicted_message(message: &Message) -> Option<EvictedMessage> {
    Some(EvictedMessage {
        id: message.id.clone()?,
        bytes: content_bytes(&message.content),
        kinds: content_kinds(message),
        starts_with: first_text(message).chars().take(160).collect(),
    })
}

/// Goose's own instructions to the model, and the turn's most recent message,
/// are not candidates: removing them would take away the reason the turn is
/// running rather than the content weighing it down.
fn is_goose_instruction(message: &Message) -> bool {
    let text = first_text(message);
    GOOSE_INSTRUCTION_PREFIXES
        .iter()
        .any(|prefix| text.starts_with(prefix))
}

/// Tells the model what goose took out of the conversation, and why.
pub(super) fn eviction_message(removed: &[EvictedMessage]) -> String {
    let listed = removed
        .iter()
        .map(|entry| {
            format!(
                "\n- {} ({}), starting: \"{}\"",
                human_bytes(entry.bytes),
                entry.kinds,
                entry.starts_with
            )
        })
        .collect::<String>();
    format!(
        "{EVICTION_NOTICE_PREFIX} the following oversized content so this request could be sent:\
         {listed}\n\nIt is no longer in the conversation. Continue the task without it, and use \
         something smaller if you still need it (crop or downscale an image, or summarize the \
         output). Do not ask for the same content again."
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

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::{CallToolRequestParams, CallToolResult, ContentBlock};

    fn sized_text(marker: &str, bytes: usize) -> Message {
        Message::user()
            .with_text(format!("{marker}{}", "x".repeat(bytes)))
            .with_generated_id_if_missing()
    }

    fn tool_call(id: &str) -> Message {
        Message::assistant()
            .with_tool_request(id, Ok(CallToolRequestParams::new("read")))
            .with_generated_id_if_missing()
    }

    fn tool_result(id: &str, bytes: usize) -> Message {
        Message::user()
            .with_tool_response(
                id,
                Ok(CallToolResult::success(vec![ContentBlock::text(
                    "x".repeat(bytes),
                )])),
            )
            .with_generated_id_if_missing()
    }

    fn conversation(messages: Vec<Message>) -> Conversation {
        Conversation::new_unvalidated(messages)
    }

    #[test]
    fn evicts_the_largest_message() {
        let small = sized_text("small", 1024);
        let large = sized_text("large", 100 * 1024);
        let recent = sized_text("recent", 8 * 1024);
        let conversation = conversation(vec![small, large.clone(), recent.clone()]);

        let evicted = messages_to_evict(&conversation);

        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].id, large.id.clone().unwrap());
        assert!(evicted[0].bytes >= 100 * 1024);
        assert_eq!(evicted[0].kinds, "text");
        assert!(evicted[0].starts_with.starts_with("large"));
    }

    #[test]
    fn keeps_the_most_recent_message() {
        // The turn is working from the newest message; taking it away would
        // remove the reason the turn is running.
        let large = sized_text("large", 100 * 1024);
        let recent = sized_text("recent", 100 * 1024);
        let conversation = conversation(vec![large, recent.clone()]);

        let evicted = messages_to_evict(&conversation);

        assert!(
            evicted
                .iter()
                .all(|entry| entry.id != recent.id.clone().unwrap()),
            "the newest message must stay: {evicted:?}",
            evicted = evicted.iter().map(|entry| &entry.id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn keeps_goose_own_instructions() {
        let instruction = Message::user()
            .with_text(format!(
                "Your last request was rejected as too large by the provider: {}",
                "x".repeat(100 * 1024)
            ))
            .with_visibility(false, true)
            .with_generated_id_if_missing();
        let recent = sized_text("recent", 1024);
        let conversation = conversation(vec![instruction.clone(), recent]);

        assert!(messages_to_evict(&conversation).is_empty());
    }

    #[test]
    fn takes_a_tool_result_out_with_its_request() {
        let call = tool_call("call-1");
        let result = tool_result("call-1", 100 * 1024);
        let recent = sized_text("recent", 1024);
        let conversation = conversation(vec![call.clone(), result.clone(), recent]);

        let evicted = messages_to_evict(&conversation);
        let ids = evicted
            .iter()
            .map(|entry| entry.id.clone())
            .collect::<Vec<_>>();

        assert!(ids.contains(&result.id.clone().unwrap()));
        assert!(
            ids.contains(&call.id.clone().unwrap()),
            "a tool pair must leave together: {ids:?}"
        );
    }

    #[test]
    fn reports_nothing_when_the_conversation_is_small() {
        let conversation = conversation(vec![sized_text("a", 1024), sized_text("b", 2048)]);

        assert!(messages_to_evict(&conversation).is_empty());
    }

    #[test]
    fn advisory_names_the_largest_contributors() {
        let large = sized_text("an-oversized-file", 100 * 1024);
        let conversation = conversation(vec![large, sized_text("recent", 16)]);

        let advisory = oversized_request_message(&conversation, "Request body is too large", None);

        assert!(advisory.contains("Request body is too large"), "{advisory}");
        assert!(advisory.contains("an-oversized-file"), "{advisory}");
        assert!(!advisory.contains("context window"), "{advisory}");
    }
}
