//! `/permit <capability>` and `/deny <capability>`: the only way the model's
//! capabilities are turned on and off. A grant is session-scoped and is written
//! to the session's own extension data, a key no tool can write.

use anyhow::{anyhow, Result};
use async_trait::async_trait;

use crate::agents::state_machine::{
    messages_since_kickoff, not_applicable, yielded_with, ConversationEffect, Emitter, GooseEffect,
    Operation, OperationResult, SlashCommand,
};
use crate::capabilities::{
    capability_names, find_capability, list_capabilities, SessionPermissions,
};
use crate::conversation::message::Message;
use crate::conversation::Conversation;
use crate::session::Session;

pub struct CapabilityOperation;

impl CapabilityOperation {
    fn report(&self, session: &Session, verb: &str) -> String {
        let permissions = SessionPermissions::read(&session.extension_data);
        let mut lines = vec![format!("**Capabilities** (use `/{verb} <name>`)")];
        for def in list_capabilities() {
            lines.push(format!(
                "- `{}` — {} [{}]",
                def.name,
                def.description,
                if permissions.is_granted(def.name) {
                    "permitted"
                } else {
                    "denied"
                }
            ));
        }
        lines.join("\n")
    }
}

#[async_trait]
impl Operation<Session, GooseEffect> for CapabilityOperation {
    fn name(&self) -> &'static str {
        "capabilities"
    }

    async fn run_command(
        &self,
        command: &SlashCommand<'_>,
        session: &Session,
        conversation: &Conversation,
        emit: &Emitter,
    ) -> Result<OperationResult<GooseEffect>> {
        let grant = match command.command {
            "permit" => true,
            "deny" => false,
            _ => return not_applicable(),
        };

        let name = command.params_str.trim();
        let verb = if grant { "permit" } else { "deny" };

        if name.is_empty() {
            let text = self.report(session, verb);
            return Self::respond(conversation, text, emit).await;
        }

        if find_capability(name).is_none() {
            let text = format!(
                "`{name}` is not a capability. Available: {}.",
                capability_names()
                    .iter()
                    .map(|name| format!("`{name}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            return Self::respond(conversation, text, emit).await;
        }

        let mut session = session.clone();
        let mut permissions = SessionPermissions::read(&session.extension_data);
        let changed = if grant {
            permissions.grant(name)
        } else {
            permissions.revoke(name)
        };
        permissions.write_into(&mut session.extension_data)?;

        let text = format!(
            "`{name}` is {} for this session{}.",
            if grant { "permitted" } else { "denied" },
            if changed { "" } else { " (already in effect)" }
        );

        let command_message = messages_since_kickoff(conversation)?
            .first()
            .cloned()
            .ok_or_else(|| anyhow!("capability command conversation has no kickoff message"))?;
        let message_id = command_message
            .id
            .clone()
            .ok_or_else(|| anyhow!("Persisted slash command message has no id"))?;
        emit.message(command_message.with_visibility(true, false))
            .await;
        let response = emit
            .message(
                Message::assistant()
                    .with_text(text)
                    .with_visibility(true, false),
            )
            .await;

        yielded_with([
            ConversationEffect::SetMessageVisibility {
                message_id,
                user_visible: true,
                agent_visible: false,
            }
            .into(),
            GooseEffect::SetExtensionData(session.extension_data),
            response.into(),
        ])
    }
}

impl CapabilityOperation {
    async fn respond(
        conversation: &Conversation,
        text: String,
        emit: &Emitter,
    ) -> Result<OperationResult<GooseEffect>> {
        let command_message = messages_since_kickoff(conversation)?
            .first()
            .cloned()
            .ok_or_else(|| anyhow!("capability command conversation has no kickoff message"))?;
        let message_id = command_message
            .id
            .clone()
            .ok_or_else(|| anyhow!("Persisted slash command message has no id"))?;
        emit.message(command_message.with_visibility(true, false))
            .await;
        let response = emit
            .message(
                Message::assistant()
                    .with_text(text)
                    .with_visibility(true, false),
            )
            .await;

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
