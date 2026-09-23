//! A record of every time a session's history was taken away from the agent.
//!
//! Compaction, clearing and eviction do not delete messages: the stored rows
//! stay, with `agent_visible` turned off. These records say when that happened,
//! why, and which messages it affected, so the archive can be told apart from
//! the messages that were never part of a conversation.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::conversation::message::Message;
use crate::conversation::Conversation;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionTrigger {
    /// The configured context threshold was crossed.
    Threshold,
    /// The provider rejected a request because the conversation was too long.
    Recovery,
    /// The user ran /compact.
    Manual,
    /// The model asked for compaction.
    Model,
    /// The user ran /clear.
    Clear,
    /// A request was refused for its size and goose removed the largest content.
    Eviction,
}

impl CompactionTrigger {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Threshold => "threshold",
            Self::Recovery => "recovery",
            Self::Manual => "manual",
            Self::Model => "model",
            Self::Clear => "clear",
            Self::Eviction => "eviction",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "threshold" => Some(Self::Threshold),
            "recovery" => Some(Self::Recovery),
            "manual" => Some(Self::Manual),
            "model" => Some(Self::Model),
            "clear" => Some(Self::Clear),
            "eviction" => Some(Self::Eviction),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactionEvent {
    pub trigger: CompactionTrigger,
    pub reason: Option<String>,
    pub before_tokens: Option<i32>,
    pub after_tokens: Option<i32>,
    pub archived_message_ids: Vec<String>,
}

impl CompactionEvent {
    pub fn new(
        trigger: CompactionTrigger,
        reason: Option<String>,
        before: &Conversation,
        after: &Conversation,
        before_tokens: Option<i32>,
        after_tokens: Option<i32>,
    ) -> Self {
        Self {
            trigger,
            reason,
            before_tokens,
            after_tokens,
            archived_message_ids: newly_archived_ids(before, after),
        }
    }

    pub fn archived_message_count(&self) -> usize {
        self.archived_message_ids.len()
    }
}

/// The messages `before` exposed to the agent and `after` no longer does.
pub fn newly_archived_ids(before: &Conversation, after: &Conversation) -> Vec<String> {
    let still_visible: HashSet<&str> = after
        .messages()
        .iter()
        .filter(|message| message.is_agent_visible())
        .filter_map(|message| message.id.as_deref())
        .collect();

    before
        .messages()
        .iter()
        .filter(|message| message.is_agent_visible())
        .filter_map(|message| message.id.as_deref())
        .filter(|id| !still_visible.contains(id))
        .map(str::to_string)
        .collect()
}

/// Give every message an id, so a rewritten conversation can be persisted by
/// id without duplicating rows that are already stored.
pub fn ensure_message_ids(conversation: Conversation) -> Conversation {
    Conversation::new_unvalidated(
        conversation
            .messages()
            .iter()
            .cloned()
            .map(Message::with_generated_id_if_missing),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::message::Message;

    fn message(id: &str, agent_visible: bool) -> Message {
        Message::user()
            .with_id(id)
            .with_text(id)
            .with_visibility(true, agent_visible)
    }

    #[test]
    fn newly_archived_ids_finds_the_messages_that_became_invisible() {
        let before = Conversation::new_unvalidated(vec![
            message("a", true),
            message("b", true),
            message("kept", true),
        ]);
        let after = Conversation::new_unvalidated(vec![
            message("a", false),
            message("b", false),
            message("kept", true),
        ]);

        assert_eq!(newly_archived_ids(&before, &after), vec!["a", "b"]);
    }

    #[test]
    fn newly_archived_ids_ignores_messages_that_were_already_hidden() {
        let before =
            Conversation::new_unvalidated(vec![message("old", false), message("now", true)]);
        let after =
            Conversation::new_unvalidated(vec![message("old", false), message("now", false)]);

        assert_eq!(newly_archived_ids(&before, &after), vec!["now"]);
    }

    #[test]
    fn clearing_archives_everything_the_agent_could_see() {
        let before =
            Conversation::new_unvalidated(vec![message("a", true), message("user-only", false)]);
        let after = Conversation::empty();

        let event =
            CompactionEvent::new(CompactionTrigger::Clear, None, &before, &after, None, None);

        assert_eq!(event.archived_message_ids, vec!["a"]);
        assert_eq!(event.archived_message_count(), 1);
    }

    #[test]
    fn ensure_message_ids_leaves_existing_ids_alone() {
        let conversation = Conversation::new_unvalidated(vec![
            message("mine", true),
            Message::user().with_text("new"),
        ]);

        let conversation = ensure_message_ids(conversation);

        assert_eq!(conversation.messages()[0].id.as_deref(), Some("mine"));
        assert!(conversation.messages()[1].id.is_some());
    }
}
