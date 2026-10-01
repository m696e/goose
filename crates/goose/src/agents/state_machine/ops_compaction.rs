//! Compacts conversation history when it is too large for the configured context window.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use tracing_futures::Instrument;

use crate::agents::final_output_tool::FinalOutputTool;
use crate::agents::state_machine::ops_llm::{chat_span, record_chat_usage};
use crate::agents::state_machine::{
    applied, awaits_tool_responses, last_effective_role, messages_since_kickoff, not_applicable,
    trailing_error, yielded, yielded_with, ConversationEffect, Emitter, GooseEffect, Operation,
    OperationResult, SlashCommand,
};
use crate::context_mgmt::{compact_messages, count_context_tokens};
use crate::conversation::message::{
    Message, MessageContent, MessageErrorKind, SystemNotificationType,
};
use crate::conversation::token_usage::TokenCount;
use crate::conversation::{Conversation, EffectiveRole};
use crate::providers::base::Provider;
use crate::session::compaction_event::{CompactionEvent, CompactionTrigger};
use crate::session::Session;
use goose_providers::model::ModelConfig;

const COMPACTION_THINKING_TEXT: &str = "goose is compacting the conversation...";

pub(super) const MAX_CONTEXT_ERROR_COMPACTIONS: usize = 2;

fn compaction_part(
    total_tokens: Option<TokenCount>,
    context_limit: usize,
    threshold: f64,
    session_modification_permitted: bool,
) -> Option<String> {
    let total_tokens = total_tokens?;
    if total_tokens <= 0 || context_limit == 0 || threshold <= 0.0 || threshold >= 1.0 {
        return None;
    }

    let compaction_at = (context_limit as f64 * threshold) as TokenCount;
    if compaction_at <= 0 || (total_tokens as f64 / compaction_at as f64) < 0.5 {
        return None;
    }

    let mut line = format!(
        "~{}k tokens remaining",
        compaction_at.saturating_sub(total_tokens) / 1000
    );
    if !session_modification_permitted {
        line.push_str(
            " (you cannot compact this session yourself; the user can run /permit session-modification)",
        );
    }
    Some(format!("<compaction>{line}</compaction>"))
}

/// Reported usage stops at the inference that requested the tools, so the
/// results that answered it are not counted until the next request.
async fn unreported_tool_tokens(conversation: &Conversation) -> Result<TokenCount> {
    let messages = conversation.messages();
    if last_effective_role(messages)? != EffectiveRole::Tool {
        return Ok(0);
    }
    let after_request = messages
        .iter()
        .rposition(Message::is_tool_call)
        .map_or(0, |index| index + 1);
    count_context_tokens(&messages[after_request..]).await
}

pub struct CompactionOperation {
    provider: Arc<dyn Provider>,
    model_config: ModelConfig,
    context_limit: usize,
    threshold: f64,
    manages_own_context: bool,
}

impl CompactionOperation {
    pub fn new(
        provider: Arc<dyn Provider>,
        model_config: ModelConfig,
        context_limit: usize,
        threshold: f64,
    ) -> Self {
        let manages_own_context = provider.manages_own_context();
        Self {
            provider,
            model_config,
            context_limit,
            threshold,
            manages_own_context,
        }
    }

    fn over_threshold(&self, tokens: usize) -> bool {
        if self.threshold <= 0.0 || self.threshold >= 1.0 {
            return false;
        }
        (tokens as f64 / self.context_limit as f64) > self.threshold
    }

    async fn context_tokens(
        &self,
        session: &Session,
        conversation: &Conversation,
    ) -> Result<TokenCount> {
        match session.usage.total_tokens {
            Some(tokens) => Ok(tokens + unreported_tool_tokens(conversation).await?),
            None => count_context_tokens(conversation.messages()).await,
        }
    }

    async fn command_error(
        conversation: &Conversation,
        message: String,
        emit: &Emitter,
    ) -> Result<OperationResult<GooseEffect>> {
        let command = messages_since_kickoff(conversation)?
            .first()
            .cloned()
            .ok_or_else(|| anyhow!("compact command conversation has no kickoff message"))?;
        let message_id = command
            .id
            .clone()
            .ok_or_else(|| anyhow!("Persisted slash command message has no id"))?;
        let command = command.with_visibility(true, false);
        let response = Message::assistant()
            .with_text(message)
            .with_visibility(true, false);
        emit.message(command).await;
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

    async fn clear(
        session: &Session,
        conversation: &Conversation,
        destroy: bool,
        emit: &Emitter,
        before_tokens: Option<TokenCount>,
    ) -> Result<OperationResult<GooseEffect>> {
        let command = messages_since_kickoff(conversation)?
            .first()
            .cloned()
            .ok_or_else(|| anyhow!("clear command conversation has no kickoff message"))?
            .with_visibility(true, false);
        let response = Message::assistant()
            .with_text("Conversation cleared")
            .with_visibility(true, false);
        let command = emit.message(command).await;
        let response = emit.message(response).await;
        let event = CompactionEvent::new(
            CompactionTrigger::Clear,
            None,
            conversation,
            &Conversation::empty(),
            before_tokens,
            Some(0),
        );
        let _ = session;
        yielded_with([
            GooseEffect::ClearConversation { destroy, event },
            command.into(),
            response.into(),
        ])
    }
}

#[async_trait]
impl Operation<Session, GooseEffect> for CompactionOperation {
    fn name(&self) -> &'static str {
        "compaction"
    }

    async fn run_command(
        &self,
        command: &SlashCommand<'_>,
        session: &Session,
        conversation: &Conversation,
        emit: &Emitter,
    ) -> Result<OperationResult<GooseEffect>> {
        match command.command {
            "clear" => {
                let destroy = command.params_str == "--destroy";
                let before_tokens = self.context_tokens(session, conversation).await.ok();
                return Self::clear(session, conversation, destroy, emit, before_tokens).await;
            }
            "compact" => {}
            _ => return not_applicable(),
        }

        let span = chat_span(
            self.provider.as_ref(),
            &self.model_config,
            &session.id,
            "compaction",
        );
        let result = match compact_messages(
            self.provider.as_ref(),
            &self.model_config,
            &session.id,
            conversation,
            true,
        )
        .instrument(span.clone())
        .await
        {
            Ok(result) => result,
            Err(error) => {
                span.record("error.type", "compaction_error");
                return Self::command_error(conversation, error.to_string(), emit).await;
            }
        };
        let compacted = result.conversation;
        let usage = result.usage;
        record_chat_usage(&span, &usage);

        let command = messages_since_kickoff(conversation)?
            .first()
            .cloned()
            .ok_or_else(|| anyhow!("compact command conversation has no kickoff message"))?
            .with_visibility(true, false);
        let response = Message::assistant()
            .with_text("Compaction complete")
            .with_visibility(true, false);
        emit.message(command).await;
        let response = emit.message(response).await;
        let event = CompactionEvent::new(
            CompactionTrigger::Manual,
            None,
            conversation,
            &compacted,
            session.usage.total_tokens,
            Some(result.retained_context_tokens),
        );
        yielded_with([
            GooseEffect::CompactConversation {
                conversation: compacted,
                usage: Some(usage),
                event,
            },
            response.into(),
        ])
    }

    async fn moim_parts(
        &self,
        session: &Session,
        conversation: &Conversation,
    ) -> Result<Vec<String>> {
        if self.manages_own_context {
            return Ok(Vec::new());
        }
        let permitted = crate::capabilities::SessionPermissions::read(&session.extension_data)
            .is_granted(crate::capabilities::SESSION_MODIFICATION);
        Ok(compaction_part(
            Some(self.context_tokens(session, conversation).await?),
            self.context_limit,
            self.threshold,
            permitted,
        )
        .into_iter()
        .collect())
    }

    async fn run(
        &self,
        session: &Session,
        conversation: &Conversation,
        emit: &Emitter,
    ) -> Result<OperationResult<GooseEffect>> {
        if self.manages_own_context {
            return not_applicable();
        }

        let messages = messages_since_kickoff(conversation)?;
        let reactive_context_error = matches!(
            trailing_error(conversation),
            Some(MessageErrorKind::ContextLengthExceeded)
        );

        if reactive_context_error {
            let context_errors = messages
                .iter()
                .filter(|message| {
                    message.error_kind() == Some(MessageErrorKind::ContextLengthExceeded)
                        && !message.is_agent_visible()
                })
                .count();
            if context_errors > MAX_CONTEXT_ERROR_COMPACTIONS {
                return not_applicable();
            }
        } else {
            // Compact only ahead of an inference. An assistant tail ends the turn or
            // awaits tool responses, hiding an unanswered request orphans its result,
            // and RecipeOperation delivers a successful final output from a tool tail.
            let tail = last_effective_role(messages)?;
            if tail == EffectiveRole::Assistant
                || awaits_tool_responses(messages)
                || (tail == EffectiveRole::Tool
                    && FinalOutputTool::successful_output(messages).is_some())
            {
                return not_applicable();
            }
            let tokens = self.context_tokens(session, conversation).await?;
            if tokens <= 0 || !self.over_threshold(tokens as usize) {
                return not_applicable();
            }
        }

        let conversation_with_hidden_error;
        let conversation = if reactive_context_error {
            let mut messages = conversation.messages().to_vec();
            let Some(last) = messages.last_mut() else {
                return not_applicable();
            };
            last.metadata.agent_visible = false;
            conversation_with_hidden_error = Conversation::new_unvalidated(messages);
            &conversation_with_hidden_error
        } else {
            conversation
        };

        let threshold_percentage = (self.threshold * 100.0) as u32;
        emit.message(Message::assistant().with_system_notification(
            SystemNotificationType::InlineMessage,
            format!(
                "Exceeded auto-compact threshold of {threshold_percentage}%. \
                     Performing auto-compaction..."
            ),
        ))
        .await;
        emit.message(Message::assistant().with_system_notification(
            SystemNotificationType::ThinkingMessage,
            COMPACTION_THINKING_TEXT,
        ))
        .await;

        let span = chat_span(
            self.provider.as_ref(),
            &self.model_config,
            &session.id,
            "compaction",
        );
        match compact_messages(
            self.provider.as_ref(),
            &self.model_config,
            &session.id,
            conversation,
            false,
        )
        .instrument(span.clone())
        .await
        {
            Ok(result) => {
                let compacted = result.conversation;
                let usage = result.usage;
                record_chat_usage(&span, &usage);
                let event = CompactionEvent::new(
                    if reactive_context_error {
                        CompactionTrigger::Recovery
                    } else {
                        CompactionTrigger::Threshold
                    },
                    None,
                    conversation,
                    &compacted,
                    self.context_tokens(session, conversation).await.ok(),
                    Some(result.retained_context_tokens),
                );
                emit.message(Message::assistant().with_system_notification(
                    SystemNotificationType::InlineMessage,
                    "Compaction complete",
                ))
                .await;
                applied([GooseEffect::CompactConversation {
                    conversation: compacted,
                    usage: Some(usage),
                    event,
                }])
            }
            Err(e) => {
                span.record("error.type", "compaction_error");
                emit.message(Message::assistant().with_text(format!(
                    "Ran into this error trying to compact: {e}.\n\n\
                     Please try again or create a new session"
                )))
                .await;
                yielded()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_compaction_part_names_the_permission_when_it_is_denied() {
        let denied = compaction_part(Some(70_000), 100_000, 0.8, false).unwrap();
        assert!(denied.contains("~10k tokens remaining"));
        assert!(denied.contains("/permit session-modification"));

        let permitted = compaction_part(Some(70_000), 100_000, 0.8, true).unwrap();
        assert_eq!(permitted, "<compaction>~10k tokens remaining</compaction>");
    }

    #[test]
    fn the_compaction_part_stays_quiet_well_below_the_threshold() {
        assert!(compaction_part(Some(10_000), 100_000, 0.8, false).is_none());
    }
}
