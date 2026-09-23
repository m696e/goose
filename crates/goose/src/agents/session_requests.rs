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

use crate::capabilities::{SessionPermissions, SESSION_MODIFICATION};
use crate::conversation::message::{Message, SystemNotificationType};
use crate::conversation::Conversation;
use crate::session::extension_data::{ExtensionData, ExtensionState};
use crate::session::Session;

/// Below this there is nothing worth compressing: the summary and its
/// continuation would replace fewer messages than they are worth.
pub const MIN_AGENT_VISIBLE_MESSAGES_TO_COMPACT: usize = 4;

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

    pub fn mark_applied(&mut self, request: &CompactionRequest) {
        self.pending_compaction = None;
        self.denied_compaction = None;
        self.applied_compaction_count = self.applied_compaction_count.saturating_add(1);
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

pub fn decide(session: &Session, conversation: &Conversation) -> CompactionDecision {
    if !session_modification_permitted(session) {
        return CompactionDecision::Denied;
    }

    let agent_visible = conversation.agent_visible_messages().len();
    if agent_visible < MIN_AGENT_VISIBLE_MESSAGES_TO_COMPACT {
        return CompactionDecision::Refused(format!(
            "only {agent_visible} message(s) are in your context"
        ));
    }

    CompactionDecision::Apply
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
            decide(&session, &conversation_of(10)),
            CompactionDecision::Denied
        ));
    }

    #[tokio::test]
    async fn a_granted_request_on_a_short_conversation_is_refused() {
        let (session, _tmp) = session(true).await;

        assert!(matches!(
            decide(&session, &conversation_of(2)),
            CompactionDecision::Refused(_)
        ));
        assert!(matches!(
            decide(&session, &conversation_of(10)),
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
