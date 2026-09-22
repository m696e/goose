//! Tells the model what made its request too large, instead of compacting.
//!
//! A provider that refuses a request on its byte size has not run out of context
//! — the payload carries too much content. Summarizing the conversation would
//! hide the very attachments that caused the refusal, so the model would ask for
//! them again; describing them lets it decide what to shrink or drop.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;

use crate::agents::request_size;
use crate::agents::state_machine::{
    applied, messages_since_kickoff, not_applicable, trailing_error, ConversationEffect, Emitter,
    GooseEffect, Operation, OperationResult,
};
use crate::conversation::message::{Message, MessageErrorKind, SystemNotificationType};
use crate::conversation::Conversation;
use crate::providers::base::Provider;
use crate::session::Session;

pub struct RequestSizeOperation {
    provider: Arc<dyn Provider>,
    manages_own_context: bool,
}

impl RequestSizeOperation {
    pub fn new(provider: Arc<dyn Provider>) -> Self {
        let manages_own_context = provider.manages_own_context();
        Self {
            provider,
            manages_own_context,
        }
    }

    /// A refused request leaves an error message behind, and goose counts those
    /// to know how often it has already asked the model to shrink the payload.
    fn advisories(&self, messages: &[Message]) -> usize {
        messages
            .iter()
            .filter(|message| {
                message.error_kind() == Some(MessageErrorKind::RequestTooLarge)
                    && !message.is_agent_visible()
            })
            .count()
    }
}

#[async_trait]
impl Operation<Session, GooseEffect> for RequestSizeOperation {
    fn name(&self) -> &'static str {
        "request_size"
    }

    async fn run(
        &self,
        _session: &Session,
        conversation: &Conversation,
        emit: &Emitter,
    ) -> Result<OperationResult<GooseEffect>> {
        if self.manages_own_context {
            return not_applicable();
        }

        let Some(MessageErrorKind::RequestTooLarge) = trailing_error(conversation) else {
            return not_applicable();
        };

        let messages = messages_since_kickoff(conversation)?;
        if self.advisories(messages) <= request_size::MAX_REQUEST_SIZE_ADVISORIES {
            let details = conversation
                .last()
                .and_then(|message| {
                    message
                        .content
                        .iter()
                        .find_map(|content| content.as_error().map(|error| error.message.clone()))
                })
                .unwrap_or_default();

            tracing::warn!(
                "Provider refused the request as too large; asking the model to shrink it: {details}"
            );

            let advisory = Message::user()
                .with_text(request_size::oversized_request_message(
                    conversation,
                    &details,
                    self.provider.max_request_bytes(),
                ))
                .with_visibility(false, true);
            let notice = Message::assistant().with_system_notification(
                SystemNotificationType::InlineMessage,
                "The request exceeded the provider's size limit. Asking the model to reduce it...",
            );

            emit.message(notice).await;
            let advisory = emit.message(advisory).await;
            return applied([advisory.into()]);
        }

        // The model did not shrink the request itself. Take the largest content
        // out of it rather than summarizing the conversation, which would only
        // hide what made the request large and send the model looking for it
        // again.
        let evictions = messages
            .iter()
            .filter(|message| message.is_agent_visible())
            .filter(|message| {
                message
                    .content
                    .iter()
                    .filter_map(|content| content.as_text())
                    .collect::<Vec<_>>()
                    .join(" ")
                    .starts_with(request_size::EVICTION_NOTICE_PREFIX)
            })
            .count();
        if evictions >= request_size::MAX_REQUEST_SIZE_EVICTIONS {
            return not_applicable();
        }

        let removed = request_size::messages_to_evict(conversation);
        if removed.is_empty() {
            return not_applicable();
        }

        tracing::warn!(
            "Request is still too large; removing {} message(s) from the conversation",
            removed.len()
        );

        let mut effects: Vec<GooseEffect> = removed
            .iter()
            .map(|evicted| {
                ConversationEffect::SetMessageVisibility {
                    message_id: evicted.id.clone(),
                    user_visible: true,
                    agent_visible: false,
                }
                .into()
            })
            .collect();

        emit.message(Message::assistant().with_system_notification(
            SystemNotificationType::InlineMessage,
            "The request was still too large. Removed the largest content so the conversation can continue...",
        ))
        .await;

        let eviction = Message::user()
            .with_text(request_size::eviction_message(&removed))
            .with_visibility(false, true);
        effects.push(eviction.into());
        applied(effects)
    }
}
