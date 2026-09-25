//! Requests the model makes about its own session, and the policy that decides
//! whether the runtime carries them out.
//!
//! A request is only ever a request: the extension records it in the session's
//! extension data, and the runtime applies it at a turn boundary, after the
//! tool result that asked for it is complete. Policy — the capability grant, and
//! refusing a compaction that would free nothing — lives here so both loops
//! behave identically.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::capabilities::{SessionPermissions, SESSION_MODIFICATION};
use crate::context_mgmt::count_context_tokens;
use crate::conversation::message::{Message, SystemNotificationType};
use crate::conversation::Conversation;
use crate::session::extension_data::{ExtensionData, ExtensionState};
use crate::session::Session;

/// Below this there is nothing worth compressing: the summary and its
/// continuation would replace fewer messages than they are worth.
pub const MIN_AGENT_VISIBLE_MESSAGES_TO_COMPACT: usize = 4;

/// A compaction carries the prompt it was asked for into the conversation it
/// produces, so the model sees that same request again as soon as the summary
/// lands and would ask for the compaction again. A repeat is only carried out
/// once the context has grown by what another summary costs, which is on the
/// order of the size the last compaction left behind and never less than this.
pub const MIN_CONTEXT_GROWTH_TO_COMPACT: i32 = 4_096;

/// A note the model carries across a compaction adds to the context it just
/// paid to shrink, so it is bounded rather than silently truncated.
pub const MAX_CARRY_CHARS: usize = 4_096;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionRequest {
    pub reason: String,
    /// Text to keep verbatim in the context after the compaction. Absent means
    /// nothing but the summary survives.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub carry: Option<String>,
    pub requested_at: i64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionRequestState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pending_compaction: Option<CompactionRequest>,
    /// A request that arrived while the capability was denied, waiting to be
    /// reported to the user.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub denied_compaction: Option<CompactionRequest>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub requested_compaction_count: u32,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub applied_compaction_count: u32,
    /// Size of the context the last applied compaction produced, which is on the
    /// order of what a summary of it costs. A request that arrives before the
    /// context has grown by at least that much would summarize the conversation
    /// that compaction just produced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_applied_context_tokens: Option<i32>,
}

fn is_zero(count: &u32) -> bool {
    *count == 0
}

impl ExtensionState for SessionRequestState {
    const EXTENSION_NAME: &'static str =
        crate::agents::platform_extensions::session_manager_ext::EXTENSION_NAME;
    const VERSION: &'static str = "v0";
}

impl SessionRequestState {
    pub fn read(session: &Session) -> Self {
        Self::from_extension_data(&session.extension_data).unwrap_or_default()
    }

    /// Write this state back into a session's extension data, leaving other
    /// extensions' records alone.
    pub fn write_into(&self, extension_data: &mut ExtensionData) -> Result<()> {
        self.to_extension_data(extension_data)
    }

    /// Write this state back to the session, re-reading it first so another
    /// writer's keys are not clobbered.
    pub async fn persist(
        &self,
        manager: &crate::session::SessionManager,
        session_id: &str,
    ) -> Result<()> {
        let mut session = manager.get_session(session_id, false).await?;
        self.write_into(&mut session.extension_data)?;
        manager
            .update(session_id)
            .extension_data(session.extension_data)
            .apply()
            .await
    }

    pub fn take_pending(&mut self) -> Option<CompactionRequest> {
        self.pending_compaction.take()
    }

    /// Move a pending request to the denied slot so the user gets told about it.
    pub fn defer_denied(&mut self, request: CompactionRequest) {
        self.pending_compaction = None;
        self.denied_compaction = Some(request);
    }

    pub fn mark_applied(&mut self, request: &CompactionRequest, retained_context_tokens: i32) {
        self.pending_compaction = None;
        self.denied_compaction = None;
        self.applied_compaction_count = self.applied_compaction_count.saturating_add(1);
        self.last_applied_context_tokens = Some(retained_context_tokens);
        let _ = request;
    }
}

pub fn session_modification_permitted(session: &Session) -> bool {
    SessionPermissions::read(&session.extension_data).is_granted(SESSION_MODIFICATION)
}

pub enum CompactionDecision {
    /// The capability is not granted; the user has to be told.
    Denied,
    /// Nothing to gain from compacting right now.
    Refused(String),
    Apply,
}

pub async fn decide(session: &Session, conversation: &Conversation) -> CompactionDecision {
    if !session_modification_permitted(session) {
        return CompactionDecision::Denied;
    }

    let agent_visible = conversation.agent_visible_messages().len();
    if agent_visible < MIN_AGENT_VISIBLE_MESSAGES_TO_COMPACT {
        return CompactionDecision::Refused(format!(
            "only {agent_visible} message(s) are in your context"
        ));
    }

    // A compaction replaces the history but carries the prompt it was asked for
    // into the result, where the model reads it as the outstanding request and
    // asks for the same compaction again. Each pass then pays for a summary that
    // the next pass immediately re-summarizes. Refuse that repeat until the
    // context has grown enough for another summary to be worth its call, or the
    // user has sent something new.
    if let Some(last) = SessionRequestState::read(session).last_applied_context_tokens {
        if outstanding_prompt_is_carried(conversation) {
            // A summary of this conversation is on the order of the size the
            // last compaction left behind, so another one cannot free much until
            // the context has grown by at least that — and by at least
            // MIN_CONTEXT_GROWTH_TO_COMPACT, so a conversation small enough to
            // grow that far within a turn cannot loop on small additions.
            let growth_needed = last.max(MIN_CONTEXT_GROWTH_TO_COMPACT);
            match count_context_tokens(conversation.messages()).await {
                Ok(current) if current.saturating_sub(last) < growth_needed => {
                    return CompactionDecision::Refused(format!(
                        "the context has not grown since the last compaction \
                         ({current} tokens, {last} when it was compacted)"
                    ));
                }
                Ok(_) => {}
                Err(error) => warn!("Could not measure the context, allowing compaction: {error}"),
            }
        }
    }

    CompactionDecision::Apply
}

/// Whether the outstanding prompt is one a compaction carried forward — an
/// agent-only copy — rather than a message the user has just sent. A user-visible
/// prompt means the user has spoken since, and that request is theirs to make.
fn outstanding_prompt_is_carried(conversation: &Conversation) -> bool {
    conversation
        .messages()
        .iter()
        .rev()
        .find(|message| crate::context_mgmt::is_preservable_prompt(message))
        .is_some_and(|message| !message.is_user_visible())
}

/// The note the model asked to keep, as a message in the compacted
/// conversation. Assistant-role on purpose: a user-role message would qualify
/// as the next compaction's preserved user prompt and be re-preserved forever.
pub fn carry_message(carry: &str) -> Message {
    Message::assistant()
        .with_text(format!(
            "Note kept verbatim across the compaction:\n\n{}",
            carry.trim()
        ))
        .with_visibility(true, true)
        .with_generated_id_if_missing()
}

/// Append the note a request carried to the conversation that replaces its
/// history. The note is the point of the request, so it is added after the
/// summary, verbatim, rather than handed to the summarizer.
pub fn with_carry(conversation: Conversation, carry: Option<&str>) -> Conversation {
    let Some(carry) = carry.map(str::trim).filter(|carry| !carry.is_empty()) else {
        return conversation;
    };
    let mut conversation = conversation;
    conversation.push(carry_message(carry));
    conversation
}

/// A request that is not applied cannot keep its note; say so, so a model that
/// still needs the information writes it somewhere durable instead.
pub fn carry_not_kept(request: &CompactionRequest) -> &'static str {
    match request.carry {
        Some(_) => " The note you attached was not kept.",
        None => "",
    }
}

/// Trim a note and reject one too long to be worth keeping: a carry that
/// re-creates the context you just paid to compact defeats the point.
pub fn validate_carry(carry: Option<&str>) -> Result<Option<String>, String> {
    let Some(carry) = carry.map(str::trim).filter(|carry| !carry.is_empty()) else {
        return Ok(None);
    };
    let length = carry.chars().count();
    if length > MAX_CARRY_CHARS {
        return Err(format!(
            "The note is {length} characters; the limit is {MAX_CARRY_CHARS}. Keep only what you cannot re-derive, and write the rest somewhere durable."
        ));
    }
    Ok(Some(carry.to_string()))
}

/// The user-facing notice for a request that was refused because the capability
/// is denied.
pub fn denied_notice(request: &CompactionRequest) -> Message {
    Message::assistant().with_system_notification(
        SystemNotificationType::InlineMessage,
        format!(
            "goose asked to compact this conversation ({}). \
             Session modification is not permitted; run /permit session-modification to allow it.",
            request.reason
        ),
    )
}

/// The user-facing notice for a compaction the model asked for and got.
pub fn applied_notice(
    request: &CompactionRequest,
    before_tokens: Option<i32>,
    after_tokens: Option<i32>,
    archived: usize,
) -> Message {
    let tokens = match (before_tokens, after_tokens) {
        (Some(before), Some(after)) => format!(": {before} → {after} tokens"),
        _ => String::new(),
    };
    Message::assistant().with_system_notification(
        SystemNotificationType::InlineMessage,
        format!(
            "Compacted at goose's request ({}){tokens}, {archived} message(s) archived — /archive lists them.",
            request.reason
        ),
    )
}

/// Refusals go to the model, which is the one that needs to react to them.
pub fn refusal_message(detail: &str) -> Message {
    Message::assistant().with_text(format!("No compaction this time: {detail}."))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GooseMode;
    use crate::conversation::message::MessageMetadata;
    use crate::session::session_manager::SessionType;
    use rmcp::model::Role;

    fn message(id: &str) -> Message {
        Message::user().with_id(id).with_text(id)
    }

    fn conversation_of(count: usize) -> Conversation {
        Conversation::new_unvalidated((0..count).map(|i| message(&format!("m{i}"))))
    }

    async fn session(with_grant: bool) -> (Session, tempfile::TempDir) {
        let temp_dir = tempfile::tempdir().unwrap();
        let manager = crate::session::SessionManager::new(temp_dir.path().to_path_buf());
        let session = manager
            .create_session(
                temp_dir.path().to_path_buf(),
                "requests".to_string(),
                SessionType::User,
                GooseMode::default(),
            )
            .await
            .unwrap();

        let mut extension_data = session.extension_data.clone();
        let mut permissions = SessionPermissions::default();
        if with_grant {
            permissions.grant(SESSION_MODIFICATION);
        }
        permissions.write_into(&mut extension_data).unwrap();
        manager
            .update(&session.id)
            .extension_data(extension_data)
            .apply()
            .await
            .unwrap();

        (
            manager.get_session(&session.id, false).await.unwrap(),
            temp_dir,
        )
    }

    #[tokio::test]
    async fn a_request_needs_the_capability() {
        let (session, _tmp) = session(false).await;

        assert!(matches!(
            decide(&session, &conversation_of(10)).await,
            CompactionDecision::Denied
        ));
    }

    #[tokio::test]
    async fn a_granted_request_on_a_short_conversation_is_refused() {
        let (session, _tmp) = session(true).await;

        assert!(matches!(
            decide(&session, &conversation_of(2)).await,
            CompactionDecision::Refused(_)
        ));
        assert!(matches!(
            decide(&session, &conversation_of(10)).await,
            CompactionDecision::Apply
        ));
    }

    /// The conversation a compaction produces: the summary, the continuation
    /// that tells the model to carry on, the prompt the compaction carried
    /// forward, and the note it kept.
    fn conversation_after_a_compaction(prompt_user_visible: bool) -> Conversation {
        let mut prompt = Message::user()
            .with_id("prompt")
            .with_text("update the client and compact");
        if !prompt_user_visible {
            prompt = prompt.with_metadata(MessageMetadata::agent_only());
        }

        Conversation::new_unvalidated(vec![
            Message::user()
                .with_id("summary")
                .with_text("# Conversation Summary")
                .with_metadata(MessageMetadata::agent_only()),
            Message::assistant()
                .with_id("continuation")
                .with_text("Your context was compacted.")
                .with_metadata(MessageMetadata::agent_only()),
            prompt,
            Message::assistant()
                .with_id("carry")
                .with_text("Note kept verbatim across the compaction: ..."),
        ])
    }

    /// The task carrying on, which is what the context does under the prompt a
    /// compaction carried forward.
    fn with_work(mut conversation: Conversation, chars: usize) -> Conversation {
        conversation.push(
            Message::assistant()
                .with_id("work")
                .with_text("step output ".repeat(chars / 12)),
        );
        conversation
    }

    async fn session_compacted_at(tokens: i32) -> (Session, tempfile::TempDir) {
        let (mut session, temp_dir) = session(true).await;
        let mut state = SessionRequestState::read(&session);
        state.last_applied_context_tokens = Some(tokens);
        state.write_into(&mut session.extension_data).unwrap();
        (session, temp_dir)
    }

    #[tokio::test]
    async fn a_repeat_of_the_prompt_the_last_compaction_carried_is_refused() {
        let conversation = conversation_after_a_compaction(false);
        let tokens = count_context_tokens(conversation.messages()).await.unwrap();
        let (session, _tmp) = session_compacted_at(tokens).await;

        assert!(
            matches!(
                decide(&session, &conversation).await,
                CompactionDecision::Refused(_)
            ),
            "a request against the conversation the last compaction produced must be refused"
        );
    }

    #[tokio::test]
    async fn a_repeat_with_only_a_turn_of_new_context_is_refused() {
        let conversation = conversation_after_a_compaction(false);
        let tokens = count_context_tokens(conversation.messages()).await.unwrap();
        let (session, _tmp) = session_compacted_at(tokens).await;
        // The verify-and-ask turn the model adds between compactions.
        let conversation = with_work(conversation, 12_000);

        assert!(
            matches!(
                decide(&session, &conversation).await,
                CompactionDecision::Refused(_)
            ),
            "a turn's worth of context is not enough to pay for another summary"
        );
    }

    #[tokio::test]
    async fn a_repeat_is_applied_once_the_context_has_grown() {
        let base = conversation_after_a_compaction(false);
        let tokens = count_context_tokens(base.messages()).await.unwrap();
        let (session, _tmp) = session_compacted_at(tokens).await;
        let conversation = with_work(base, 48_000);

        let grown = count_context_tokens(conversation.messages()).await.unwrap();
        assert!(
            grown - tokens >= MIN_CONTEXT_GROWTH_TO_COMPACT,
            "the fixture must grow past the bar to be worth asserting on: {tokens} -> {grown}"
        );
        assert!(matches!(
            decide(&session, &conversation).await,
            CompactionDecision::Apply
        ));
    }

    #[tokio::test]
    async fn a_repeat_is_refused_until_the_context_regrows_past_what_it_left() {
        // A compaction that left a large context: a summary of it is on the
        // order of that size, so a turn of new context cannot pay for another.
        let conversation = with_work(conversation_after_a_compaction(false), 48_000);
        let tokens = count_context_tokens(conversation.messages()).await.unwrap();
        let (session, _tmp) = session_compacted_at(tokens).await;

        assert!(matches!(
            decide(&session, &conversation).await,
            CompactionDecision::Refused(_)
        ));

        let (session, _tmp) = session_compacted_at(tokens / 4).await;
        assert!(matches!(
            decide(&session, &conversation).await,
            CompactionDecision::Apply
        ));
    }

    #[tokio::test]
    async fn a_prompt_the_user_has_just_sent_is_applied_without_growth() {
        let conversation = conversation_after_a_compaction(true);
        let tokens = count_context_tokens(conversation.messages()).await.unwrap();
        let (session, _tmp) = session_compacted_at(tokens).await;

        assert!(matches!(
            decide(&session, &conversation).await,
            CompactionDecision::Apply
        ));
    }

    #[test]
    fn state_round_trips_through_extension_data() {
        let mut extension_data = ExtensionData::new();
        let state = SessionRequestState {
            pending_compaction: Some(CompactionRequest {
                reason: "the tool output is no longer needed".to_string(),
                carry: Some("step 3 of 7: inputs are in /tmp/step3.json".to_string()),
                requested_at: 42,
            }),
            denied_compaction: None,
            requested_compaction_count: 3,
            applied_compaction_count: 1,
            last_applied_context_tokens: Some(4_321),
        };

        state.write_into(&mut extension_data).unwrap();

        let reloaded =
            SessionRequestState::from_extension_data(&extension_data).unwrap_or_default();
        assert_eq!(reloaded, state);
        assert!(!SessionPermissions::read(&extension_data).is_granted(SESSION_MODIFICATION));
    }

    #[test]
    fn notices_name_the_reason_and_the_way_to_allow_it() {
        use crate::conversation::message::MessageContentBlock;

        fn notification_text(message: &Message) -> String {
            message
                .content
                .iter()
                .find_map(|content| match content {
                    MessageContentBlock::SystemNotification(notification) => {
                        Some(notification.msg.clone())
                    }
                    _ => None,
                })
                .unwrap_or_default()
        }

        let request = CompactionRequest {
            reason: "finished reading the logs".to_string(),
            carry: None,
            requested_at: 0,
        };

        let denied = notification_text(&denied_notice(&request));
        assert!(denied.contains("finished reading the logs"));
        assert!(denied.contains("/permit session-modification"));
        assert!(denied_notice(&request).is_user_visible());
        assert!(!denied_notice(&request).is_agent_visible());

        let applied = notification_text(&applied_notice(&request, Some(100), Some(20), 7));
        assert!(applied.contains("100 → 20 tokens"));
        assert!(applied.contains("7 message(s) archived"));

        assert!(refusal_message("only 2 message(s) are in your context")
            .as_concat_text()
            .contains("No compaction this time"));
    }

    #[test]
    fn a_carry_is_added_after_the_compacted_history_and_visible_to_both() {
        let conversation = with_carry(conversation_of(2), Some("step 3 of 7"));

        let carried = conversation.messages().last().unwrap();
        assert!(carried.as_concat_text().contains("step 3 of 7"));
        assert!(carried.is_agent_visible());
        assert!(carried.is_user_visible());
        assert!(carried.id.is_some());
        assert!(conversation
            .messages()
            .iter()
            .filter(|message| message.as_concat_text().contains("step 3 of 7"))
            .all(|message| message.role == Role::Assistant));
    }

    #[test]
    fn no_carry_leaves_the_conversation_alone() {
        let before = conversation_of(2);

        assert_eq!(with_carry(before.clone(), None).messages().len(), 2);
        assert_eq!(
            with_carry(before.clone(), Some("  \n ")).messages().len(),
            2
        );
    }

    #[test]
    fn a_carry_is_trimmed_and_bounded() {
        assert_eq!(
            validate_carry(Some("  keep this  ")).unwrap(),
            Some("keep this".to_string())
        );
        assert_eq!(validate_carry(Some("   ")).unwrap(), None);
        assert_eq!(validate_carry(None).unwrap(), None);

        let long = "x".repeat(MAX_CARRY_CHARS + 1);
        let error = validate_carry(Some(&long)).unwrap_err();
        assert!(error.contains(&MAX_CARRY_CHARS.to_string()));
        assert!(validate_carry(Some(&"x".repeat(MAX_CARRY_CHARS))).is_ok());
    }

    #[test]
    fn a_request_that_is_not_applied_says_its_note_was_not_kept() {
        let mut request = CompactionRequest {
            reason: "done with the logs".to_string(),
            carry: None,
            requested_at: 0,
        };
        assert!(carry_not_kept(&request).is_empty());

        request.carry = Some("step 3 of 7".to_string());
        assert!(carry_not_kept(&request).contains("was not kept"));
    }
}
