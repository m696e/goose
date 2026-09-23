use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use goose_providers::base::Provider;
use goose_providers::model::ModelConfig;

use crate::agents::state_machine::{
    messages_since_kickoff, not_applicable, yielded_with, ConversationEffect, Emitter, GooseEffect,
    Operation, OperationResult, SlashCommand,
};
use crate::conversation::message::Message;
use crate::conversation::Conversation;
use crate::session::Session;

pub struct StatusOperation {
    provider: Arc<dyn Provider>,
    model_config: ModelConfig,
    session_manager: Arc<crate::session::SessionManager>,
}

impl StatusOperation {
    pub fn new(
        provider: Arc<dyn Provider>,
        model_config: ModelConfig,
        session_manager: Arc<crate::session::SessionManager>,
    ) -> Self {
        Self {
            provider,
            model_config,
            session_manager,
        }
    }
}

/// A queued compaction request, if there is one, so a user reading /status knows
/// history is about to change.
fn pending_request(session: &Session) -> String {
    let state = crate::agents::session_requests::SessionRequestState::read(session);
    if let Some(pending) = state.pending_compaction.as_ref() {
        let carried = pending
            .carry
            .as_deref()
            .map(|carry| format!(", carrying {} character(s) of note", carry.chars().count()))
            .unwrap_or_default();
        return format!(
            "\n- Compaction: requested ({}){carried}; applied at the next turn boundary",
            pending.reason
        );
    }
    if state.denied_compaction.is_some() {
        return "\n- Compaction: requested while not permitted; nothing was applied".to_string();
    }
    String::new()
}

#[async_trait]
impl Operation<Session, GooseEffect> for StatusOperation {
    fn name(&self) -> &'static str {
        "status"
    }

    async fn run_command(
        &self,
        command: &SlashCommand<'_>,
        session: &Session,
        conversation: &Conversation,
        emit: &Emitter,
    ) -> Result<OperationResult<GooseEffect>> {
        if command.command != "status" {
            return not_applicable();
        }
        let context_limit = crate::context_limit::get_context_limit(
            self.provider.as_ref(),
            &self.model_config.model_name,
        )
        .await?;
        let context_tokens = session.usage.total_tokens.unwrap_or(0);
        let lifetime_tokens = session.accumulated_usage.total_tokens.unwrap_or(0);
        let archive = self
            .session_manager
            .list_compaction_events(&session.id)
            .await
            .map(|events| {
                crate::session::session_manager::archive_summary(&events)
                    .map(|detail| format!("\n- Archived history: {detail}"))
                    .unwrap_or_default()
            })
            .unwrap_or_default();
        let context_pct = if context_limit > 0 {
            format!(
                "{}%",
                ((context_tokens as f64 / context_limit as f64) * 100.0)
                    .round()
                    .min(100.0) as usize
            )
        } else {
            "N/A".to_string()
        };
        let response = Message::assistant().with_text(format!("**Session status**\n\n- Model: {}\n- Provider: {}\n- Mode: {}\n- Tokens (lifetime): {}\n- Context: {} / {} tokens ({}){}{}", self.model_config.model_name, self.provider.get_name(), session.goose_mode, lifetime_tokens, context_tokens, context_limit, context_pct, archive, pending_request(session))).with_visibility(true, false);
        let command_message = messages_since_kickoff(conversation)?
            .first()
            .cloned()
            .ok_or_else(|| anyhow!("status command conversation has no kickoff message"))?;
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
