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
    applied, messages_since_kickoff, not_applicable, trailing_error, Emitter, GooseEffect,
    Operation, OperationResult,
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
        if self.advisories(messages) > request_size::MAX_REQUEST_SIZE_ADVISORIES {
            // The budget is spent: let the error surface rather than asking
            // again, so a model that keeps resending the same request cannot
            // hold the turn open.
            return not_applicable();
        }

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
        applied([advisory.into()])
    }
}
