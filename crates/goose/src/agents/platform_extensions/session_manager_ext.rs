//! Gives the model a read-only view of the session it is working in, so it can
//! plan around the context window instead of discovering the limit by hitting
//! it. Mutating tools arrive with the capability gate; this one only reports.

use anyhow::Result;
use async_trait::async_trait;
use indoc::indoc;
use rmcp::model::{
    CallToolResult, ContentBlock, Implementation, InitializeResult, JsonObject, ListToolsResult,
    ServerCapabilities, Tool, ToolAnnotations,
};
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::agents::extension::PlatformExtensionContext;
use crate::agents::mcp_client::{Error, McpClientTrait};
use crate::agents::session_requests::{CompactionRequest, SessionRequestState};
use crate::agents::tool_execution::ToolCallContext;
use crate::capabilities::SessionPermissions;
use crate::config::Config;
use crate::context_mgmt::DEFAULT_COMPACTION_THRESHOLD;
use crate::session::session_manager::archive_summary;

pub static EXTENSION_NAME: &str = "session-manager";
pub const SESSION_STATUS_TOOL_NAME: &str = "session_status";
pub const SESSION_STATUS_TOOL_NAME_COMPLETE: &str = "session-manager__session_status";
pub const REQUEST_COMPACTION_TOOL_NAME: &str = "request_compaction";
pub const REQUEST_COMPACTION_TOOL_NAME_COMPLETE: &str = "session-manager__request_compaction";

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct SessionStatusParams {}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct RequestCompactionParams {
    /// Why this is the right moment to compact, in one sentence. This is what
    /// the user reads, and it is kept as the record of the request.
    reason: String,
}

pub struct SessionManagerClient {
    info: InitializeResult,
    context: PlatformExtensionContext,
}

impl SessionManagerClient {
    pub fn new(context: PlatformExtensionContext) -> Result<Self> {
        let info = InitializeResult::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(
                Implementation::new(EXTENSION_NAME.to_string(), "1.0.0".to_string())
                    .with_title("Session Manager"),
            )
            .with_instructions(
                indoc! {r#"
                Session management tools report on the session you are working in.

                Use session_status to see how full your context is before you commit to a long
                stretch of work, and when you are deciding whether history still has to be kept.
            "#}
                .to_string(),
            );

        Ok(Self { info, context })
    }

    async fn session_status(&self, session_id: &str) -> Result<String, String> {
        let session = self
            .context
            .session_manager
            .get_session(session_id, true)
            .await
            .map_err(|error| format!("Failed to read the session: {error}"))?;
        let conversation = session.conversation.clone().unwrap_or_default();

        let provider = match self
            .context
            .extension_manager
            .as_ref()
            .and_then(|manager| manager.upgrade())
        {
            Some(manager) => manager.get_provider().lock().await.clone(),
            None => None,
        };
        let model_config = self
            .context
            .model_config_for_session(session_id)
            .await
            .map_err(|error| format!("Failed to resolve the model: {error}"))?;

        let context_limit = match provider.as_ref() {
            Some(provider) => {
                crate::context_limit::get_context_limit(provider.as_ref(), &model_config.model_name)
                    .await
                    .ok()
            }
            None => None,
        };

        let threshold = Config::global()
            .get_param::<f64>("GOOSE_AUTO_COMPACT_THRESHOLD")
            .unwrap_or(DEFAULT_COMPACTION_THRESHOLD);
        let auto_compaction_enabled = threshold > 0.0 && threshold < 1.0;

        let used = session.usage.total_tokens.filter(|tokens| *tokens >= 0);
        let counted = crate::context_mgmt::count_context_tokens(conversation.messages())
            .await
            .ok()
            .filter(|tokens| *tokens >= 0);

        let mut lines = vec![
            format!("session: {}", session.id),
            format!(
                "model: {} ({})",
                model_config.model_name,
                session.provider_name.as_deref().unwrap_or("unknown")
            ),
        ];

        match (used, context_limit) {
            (Some(used), Some(limit)) if limit > 0 => lines.push(format!(
                "context: {used} / {limit} tokens ({:.0}%) as of the last request; that is the whole request, including the system prompt and tool definitions",
                (used as f64 / limit as f64) * 100.0
            )),
            (Some(used), _) => lines.push(format!(
                "context: {used} tokens as of the last request (limit unknown)"
            )),
            _ => lines.push("context: not reported by the provider yet".to_string()),
        }

        if let Some(counted) = counted {
            let share = match context_limit {
                Some(limit) if limit > 0 => {
                    format!(" ({:.0}%)", (counted as f64 / limit as f64) * 100.0)
                }
                _ => String::new(),
            };
            lines.push(format!(
                "conversation: ~{counted} tokens{share} counted now; messages alone, excluding the system prompt and tool definitions"
            ));
        }

        if auto_compaction_enabled {
            match (used, context_limit) {
                (Some(used), Some(limit)) if limit > 0 => {
                    let threshold_tokens = (limit as f64 * threshold) as i32;
                    lines.push(format!(
                        "auto-compaction: at {:.0}% of the limit ({} tokens); {} tokens to go",
                        threshold * 100.0,
                        threshold_tokens,
                        threshold_tokens.saturating_sub(used)
                    ));
                }
                _ => lines.push(format!(
                    "auto-compaction: at {:.0}% of the limit",
                    threshold * 100.0
                )),
            }
        } else {
            lines.push("auto-compaction: disabled".to_string());
        }

        lines.push(format!(
            "messages: {} visible to you, {} on record",
            conversation.agent_visible_messages().len(),
            conversation.messages().len()
        ));

        let events = self
            .context
            .session_manager
            .list_compaction_events(session_id)
            .await
            .unwrap_or_default();
        lines.push(match archive_summary(&events) {
            Some(detail) => format!("archived: {detail}"),
            None => "archived: nothing yet".to_string(),
        });

        let permitted = SessionPermissions::read(&session.extension_data)
            .is_granted(crate::capabilities::SESSION_MODIFICATION);
        lines.push(if permitted {
            "session-modification: permitted".to_string()
        } else {
            "session-modification: denied (the user can run /permit session-modification)"
                .to_string()
        });

        Ok(lines.join("\n"))
    }

    async fn request_compaction(&self, session_id: &str, reason: &str) -> Result<String, String> {
        let reason = reason.trim();
        if reason.is_empty() {
            return Err("A reason is required: it is the record of the request.".to_string());
        }

        let manager = &self.context.session_manager;
        let mut session = manager
            .get_session(session_id, false)
            .await
            .map_err(|error| format!("Failed to read the session: {error}"))?;

        let request = CompactionRequest {
            reason: reason.to_string(),
            requested_at: chrono::Utc::now().timestamp(),
        };
        let mut state = SessionRequestState::read(&session);
        state.requested_compaction_count = state.requested_compaction_count.saturating_add(1);

        let permitted = SessionPermissions::read(&session.extension_data)
            .is_granted(crate::capabilities::SESSION_MODIFICATION);

        let text = if permitted {
            if state.pending_compaction.is_some() {
                return Ok(
                    "A compaction request is already pending; it will be applied before your next turn."
                        .to_string(),
                );
            }
            state.pending_compaction = Some(request);
            "Compaction requested. It will be applied at the next turn boundary, before your next request.".to_string()
        } else {
            state.denied_compaction = Some(request);
            "Compaction is not permitted in this session, so nothing was done. The user has been shown your request and can run /permit session-modification to allow it; ask them if you need it now.".to_string()
        };

        state
            .write_into(&mut session.extension_data)
            .map_err(|error| format!("Failed to record the request: {error}"))?;
        manager
            .update(session_id)
            .extension_data(session.extension_data)
            .apply()
            .await
            .map_err(|error| format!("Failed to record the request: {error}"))?;

        Ok(text)
    }

    fn get_tools() -> Vec<Tool> {
        let schema = schema_for!(SessionStatusParams);
        let schema_value =
            serde_json::to_value(schema).expect("Failed to serialize SessionStatusParams schema");
        let request_schema = schema_for!(RequestCompactionParams);
        let request_schema_value = serde_json::to_value(request_schema)
            .expect("Failed to serialize RequestCompactionParams schema");

        vec![
            Tool::new(
                SESSION_STATUS_TOOL_NAME.to_string(),
                indoc! {r#"
                Report how full your context is: tokens used and the model's context limit, the
                auto-compaction threshold and how much room is left before it triggers, how many
                messages are visible to you versus kept on record, and what history has been
                archived.

                Read-only. Use it before starting a long run of tool calls, and when you are
                deciding whether earlier history still needs to be in context.
            "#}
                .to_string(),
                schema_value.as_object().unwrap().clone(),
            )
            .annotate(ToolAnnotations::from_raw(
                Some("Session status".to_string()),
                Some(true),
                Some(false),
                Some(true),
                Some(false),
            )),
            Tool::new(
                REQUEST_COMPACTION_TOOL_NAME.to_string(),
                indoc! {r#"
                Ask for the session to compact its conversation history now.

                Compaction replaces the conversation so far with a summary, so anything you still
                need in detail will be gone from your context; the messages stay on record and
                /archive can show them. Ask for it when the work has moved on and earlier history
                is no longer worth its space — a long tool output you have finished with, a
                sub-task you have closed out — rather than waiting for the automatic threshold.

                It only happens if the user has permitted session modification in this session, and
                it is applied at the next turn boundary, before your next request.
            "#}
                .to_string(),
                request_schema_value.as_object().unwrap().clone(),
            )
            .annotate(ToolAnnotations::from_raw(
                Some("Request compaction".to_string()),
                Some(false),
                Some(false),
                Some(false),
                Some(true),
            )),
        ]
    }
}

#[async_trait]
impl McpClientTrait for SessionManagerClient {
    async fn list_tools(
        &self,
        _session_id: &str,
        _next_cursor: Option<String>,
        _cancellation_token: CancellationToken,
    ) -> Result<ListToolsResult, Error> {
        Ok(ListToolsResult {
            tools: Self::get_tools(),
            next_cursor: None,
            meta: None,
            ..Default::default()
        })
    }

    async fn call_tool(
        &self,
        ctx: &ToolCallContext,
        name: &str,
        arguments: Option<JsonObject>,
        _cancellation_token: CancellationToken,
    ) -> Result<CallToolResult, Error> {
        let result = match name {
            SESSION_STATUS_TOOL_NAME => self.session_status(&ctx.session_id).await,
            REQUEST_COMPACTION_TOOL_NAME => {
                let reason = arguments
                    .as_ref()
                    .and_then(|arguments| arguments.get("reason"))
                    .and_then(|value| value.as_str())
                    .unwrap_or("");
                self.request_compaction(&ctx.session_id, reason).await
            }
            _ => Err(format!("Unknown tool: {name}")),
        };

        match result {
            Ok(text) => Ok(CallToolResult::success(vec![ContentBlock::text(text)])),
            Err(error) => Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "Error: {error}"
            ))])),
        }
    }

    fn get_info(&self) -> Option<&InitializeResult> {
        Some(&self.info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GooseMode;
    use crate::conversation::message::Message;
    use crate::session::session_manager::SessionType;
    use std::sync::Arc;

    fn client(
        manager: &Arc<crate::session::SessionManager>,
        session: &crate::session::Session,
    ) -> SessionManagerClient {
        SessionManagerClient::new(PlatformExtensionContext {
            extension_manager: None,
            session_manager: manager.clone(),
            scheduler: None,
            session: Some(Arc::new(session.clone())),
            use_login_shell_path: false,
        })
        .unwrap()
    }

    async fn session_with_history() -> (
        crate::session::Session,
        Arc<crate::session::SessionManager>,
        tempfile::TempDir,
    ) {
        let temp_dir = tempfile::tempdir().unwrap();
        let manager = Arc::new(crate::session::SessionManager::new(
            temp_dir.path().to_path_buf(),
        ));
        let session = manager
            .create_session(
                temp_dir.path().to_path_buf(),
                "status".to_string(),
                SessionType::User,
                GooseMode::default(),
            )
            .await
            .unwrap();
        manager
            .add_message(
                &session.id,
                &Message::user().with_id("m1").with_text("hello"),
            )
            .await
            .unwrap();
        manager
            .update(&session.id)
            .usage(goose_providers::conversation::token_usage::Usage::new(
                Some(500),
                Some(0),
                Some(500),
            ))
            .apply()
            .await
            .unwrap();
        (session, manager, temp_dir)
    }

    #[tokio::test]
    async fn status_reports_usage_and_the_message_counts() {
        let (session, manager, _tmp) = session_with_history().await;
        let client = client(&manager, &session);

        let status = client.session_status(&session.id).await.unwrap();

        assert!(status.contains("context: 500 tokens as of the last request"));
        assert!(status.contains("messages: 1 visible to you, 1 on record"));
        assert!(status.contains("archived: nothing yet"));
        assert!(status.contains("session-modification: denied"));
        // The provider's number covers the whole request; the estimate does not,
        // and the output says so rather than showing two unlabelled numbers.
        assert!(status.contains("conversation: ~"));
        assert!(status.contains("messages alone, excluding the system prompt and tool definitions"));
    }

    #[tokio::test]
    async fn status_separates_the_request_total_from_the_conversation_estimate() {
        let (session, manager, _tmp) = session_with_history().await;
        manager
            .update(&session.id)
            .model_config(goose_providers::model::ModelConfig::new("some-model"))
            .apply()
            .await
            .unwrap();
        let client = client(&manager, &session);

        let status = client.session_status(&session.id).await.unwrap();

        // Without a provider the limit is unknown, but the two numbers are still
        // labelled so they cannot be read as the same measurement.
        assert!(status.contains("as of the last request"));
        assert!(status.contains("conversation: ~"));
        assert!(status.contains("excluding the system prompt and tool definitions"));
    }

    #[tokio::test]
    async fn status_reports_the_archive_once_history_is_taken_away() {
        let (session, manager, _tmp) = session_with_history().await;
        let client = client(&manager, &session);

        let conversation = manager
            .get_session(&session.id, true)
            .await
            .unwrap()
            .conversation
            .unwrap();
        manager
            .archive_conversation(
                &session.id,
                &crate::session::CompactionEvent::new(
                    crate::session::CompactionTrigger::Clear,
                    None,
                    &conversation,
                    &crate::conversation::Conversation::empty(),
                    Some(500),
                    Some(0),
                ),
            )
            .await
            .unwrap();

        let status = client.session_status(&session.id).await.unwrap();

        assert!(status.contains("archived: 1 message(s) in 1 event(s) (last: clear at"));
        assert!(status.contains("messages: 0 visible to you, 1 on record"));
    }

    #[tokio::test]
    async fn the_extension_is_registered_with_its_tool() {
        assert!(crate::agents::extension::PLATFORM_EXTENSIONS.contains_key(EXTENSION_NAME));

        let tools = SessionManagerClient::get_tools();
        let names: Vec<&str> = tools.iter().map(|tool| tool.name.as_ref()).collect();
        assert_eq!(
            names,
            vec![SESSION_STATUS_TOOL_NAME, REQUEST_COMPACTION_TOOL_NAME]
        );
    }

    #[tokio::test]
    async fn a_request_needs_a_reason() {
        let (session, manager, _tmp) = session_with_history().await;
        let client = client(&manager, &session);

        assert!(client.request_compaction(&session.id, "  ").await.is_err());
    }

    #[tokio::test]
    async fn a_denied_request_is_recorded_so_the_user_is_told() {
        use crate::agents::session_requests::SessionRequestState;
        use crate::capabilities::SESSION_MODIFICATION;

        let (session, manager, _tmp) = session_with_history().await;
        let client = client(&manager, &session);

        let text = client
            .request_compaction(&session.id, "the logs are done with")
            .await
            .unwrap();

        assert!(text.contains("/permit session-modification"));
        let session = manager.get_session(&session.id, false).await.unwrap();
        let state = SessionRequestState::read(&session);
        assert!(state.pending_compaction.is_none());
        assert_eq!(
            state
                .denied_compaction
                .as_ref()
                .map(|request| request.reason.as_str()),
            Some("the logs are done with")
        );
        assert_eq!(state.requested_compaction_count, 1);
        assert!(
            !crate::capabilities::SessionPermissions::read(&session.extension_data)
                .is_granted(SESSION_MODIFICATION)
        );
    }

    #[tokio::test]
    async fn a_permitted_request_is_queued_and_asks_once() {
        use crate::agents::session_requests::SessionRequestState;
        use crate::capabilities::{SessionPermissions, SESSION_MODIFICATION};

        let (session, manager, _tmp) = session_with_history().await;
        let mut session_data = manager.get_session(&session.id, false).await.unwrap();
        let mut permissions = SessionPermissions::default();
        permissions.grant(SESSION_MODIFICATION);
        permissions
            .write_into(&mut session_data.extension_data)
            .unwrap();
        manager
            .update(&session.id)
            .extension_data(session_data.extension_data)
            .apply()
            .await
            .unwrap();
        let client = client(&manager, &session);

        let text = client
            .request_compaction(&session.id, "the logs are done with")
            .await
            .unwrap();
        assert!(text.contains("Compaction requested"));

        let session = manager.get_session(&session.id, false).await.unwrap();
        let state = SessionRequestState::read(&session);
        assert_eq!(
            state
                .pending_compaction
                .as_ref()
                .map(|request| request.reason.as_str()),
            Some("the logs are done with")
        );

        let again = client
            .request_compaction(&session.id, "again")
            .await
            .unwrap();
        assert!(again.contains("already pending"));
    }

    #[tokio::test]
    async fn unknown_tools_are_rejected() {
        let (session, manager, _tmp) = session_with_history().await;
        let client = client(&manager, &session);
        let ctx = ToolCallContext::new(session.id.clone(), None, None);

        let result = client
            .call_tool(&ctx, "not_a_tool", None, CancellationToken::new())
            .await
            .unwrap();

        assert!(result.is_error.unwrap_or(false));
    }
}
