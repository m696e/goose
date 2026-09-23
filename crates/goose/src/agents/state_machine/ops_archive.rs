//! Shows the history that compaction or `/clear` took away from the agent.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_trait::async_trait;

use crate::agents::state_machine::{
    messages_since_kickoff, not_applicable, yielded_with, ConversationEffect, Emitter, GooseEffect,
    Operation, OperationResult, SlashCommand,
};
use crate::conversation::message::Message;
use crate::conversation::Conversation;
use crate::session::session_manager::archive_report;
use crate::session::{Session, SessionManager};

pub struct ArchiveOperation {
    session_manager: Arc<SessionManager>,
}

impl ArchiveOperation {
    pub fn new(session_manager: Arc<SessionManager>) -> Self {
        Self { session_manager }
    }
}

#[async_trait]
impl Operation<Session, GooseEffect> for ArchiveOperation {
    fn name(&self) -> &'static str {
        "archive"
    }

    async fn run_command(
        &self,
        command: &SlashCommand<'_>,
        session: &Session,
        conversation: &Conversation,
        emit: &Emitter,
    ) -> Result<OperationResult<GooseEffect>> {
        if command.command != "archive" {
            return not_applicable();
        }

        let report = archive_report(&self.session_manager, &session.id, command.params_str).await?;
        let response = Message::assistant()
            .with_text(report)
            .with_visibility(true, false);
        let command_message = messages_since_kickoff(conversation)?
            .first()
            .cloned()
            .ok_or_else(|| anyhow!("archive command conversation has no kickoff message"))?;
        let message_id = command_message
            .id
            .clone()
            .ok_or_else(|| anyhow!("Persisted slash command message has no id"))?;
        emit.message(command_message.with_visibility(true, false))
            .await;
        let response = emit.message(response).await;

        yielded_with([
            ConversationEffect::SetMessageVisibility {
                message_id,
                user_visible: true,
                agent_visible: false,
            }
            .into(),
            response.into(),
        ])
    }
}
