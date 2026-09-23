//! Applies a compaction the model asked for, at a turn boundary.
//!
//! The request arrives as a tool call and is recorded in the session's
//! extension data, so this runs on the next step with the tool pair already
//! complete: a compaction can never land between a `tool_use` and its
//! `tool_result`. What may happen is decided by `session_requests::decide`.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use tracing_futures::Instrument;

use crate::agents::session_requests::{
    applied_notice, carry_not_kept, decide, denied_notice, refusal_message, with_carry,
    CompactionDecision, SessionRequestState,
};
use crate::agents::state_machine::ops_llm::{chat_span, record_chat_usage};
use crate::agents::state_machine::{
    applied, not_applicable, Emitter, GooseEffect, Operation, OperationResult,
};
use crate::context_mgmt::compact_messages;
use crate::conversation::message::Message;
use crate::conversation::Conversation;
use crate::providers::base::Provider;
use crate::session::compaction_event::{CompactionEvent, CompactionTrigger};
use crate::session::Session;
use goose_providers::model::ModelConfig;

pub struct SessionRequestOperation {
    provider: Arc<dyn Provider>,
    model_config: ModelConfig,
}

impl SessionRequestOperation {
    pub fn new(provider: Arc<dyn Provider>, model_config: ModelConfig) -> Self {
        Self {
            provider,
            model_config,
        }
    }
}

#[async_trait]
impl Operation<Session, GooseEffect> for SessionRequestOperation {
    fn name(&self) -> &'static str {
        "session_request"
    }

    async fn run(
        &self,
        session: &Session,
        conversation: &Conversation,
        emit: &Emitter,
    ) -> Result<OperationResult<GooseEffect>> {
        let mut state = SessionRequestState::read(session);

        // A request that arrived while the capability was denied: the user is
        // told once, then it is dropped.
        if let Some(denied) = state.denied_compaction.take() {
            let notice = emit.message(denied_notice(&denied)).await;
            return Self::persist(session, state, notice).await;
        }

        let Some(request) = state.take_pending() else {
            return not_applicable();
        };

        match decide(session, conversation) {
            CompactionDecision::Denied => {
                state.defer_denied(request.clone());
                let notice = emit.message(denied_notice(&request)).await;
                Self::persist(session, state, notice).await
            }
            CompactionDecision::Refused(detail) => {
                let detail = format!("{detail}{}", carry_not_kept(&request));
                let notice = emit.message(refusal_message(&detail)).await;
                Self::persist(session, state, notice).await
            }
            CompactionDecision::Apply => {
                let before = conversation.clone();
                let before_tokens = session.usage.total_tokens;
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
                        let compacted = with_carry(result.conversation, request.carry.as_deref());
                        let usage = result.usage;
                        record_chat_usage(&span, &usage);
                        let event = CompactionEvent::new(
                            CompactionTrigger::Model,
                            Some(request.reason.clone()),
                            &before,
                            &compacted,
                            before_tokens,
                            Some(result.retained_context_tokens),
                        )
                        .with_carry(request.carry.clone());
                        let notice = emit
                            .message(applied_notice(
                                &request,
                                before_tokens,
                                Some(result.retained_context_tokens),
                                event.archived_message_count(),
                            ))
                            .await;
                        state.mark_applied(&request);

                        let mut extension_data = session.extension_data.clone();
                        state.write_into(&mut extension_data)?;
                        applied([
                            GooseEffect::CompactConversation {
                                conversation: compacted,
                                usage: Some(usage),
                                event,
                            },
                            GooseEffect::SetExtensionData(extension_data),
                            notice.into(),
                        ])
                    }
                    Err(error) => {
                        span.record("error.type", "compaction_error");
                        let notice = emit
                            .message(refusal_message(&format!(
                                "compaction failed ({error}); ask again if you still need it"
                            )))
                            .await;
                        Self::persist(session, state, notice).await
                    }
                }
            }
        }
    }
}

impl SessionRequestOperation {
    /// Clearing a request and telling someone about it go together: the notice
    /// is part of the outcome, so it is persisted rather than only streamed.
    async fn persist(
        session: &Session,
        state: SessionRequestState,
        notice: Message,
    ) -> Result<OperationResult<GooseEffect>> {
        let mut extension_data = session.extension_data.clone();
        state.write_into(&mut extension_data)?;
        applied([GooseEffect::SetExtensionData(extension_data), notice.into()])
    }
}
