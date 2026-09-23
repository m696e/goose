use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;
use goose::agents::{Agent, AgentEvent, LlmStage, SessionConfig};
use goose::config::GooseMode;
use goose::conversation::message::{Message, MessageContent};
use goose::conversation::Conversation;
use goose::providers::base::{
    stream_from_single_message, MessageStream, Provider, ProviderDef, ProviderMetadata,
};
use goose::session::session_manager::SessionType;
use goose::session::Session;
use goose_providers::conversation::token_usage::{ProviderUsage, Usage};
use goose_providers::errors::ProviderError;
use goose_providers::model::ModelConfig;
use rmcp::model::Tool;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tempfile::TempDir;

struct MockCompactionProvider {
    /// Tracks whether compaction has occurred (for context limit recovery case)
    has_compacted: Arc<AtomicBool>,
    manages_own_context: bool,
}

impl MockCompactionProvider {
    fn new() -> Self {
        Self {
            has_compacted: Arc::new(AtomicBool::new(false)),
            manages_own_context: false,
        }
    }

    fn context_owning() -> Self {
        Self {
            has_compacted: Arc::new(AtomicBool::new(false)),
            manages_own_context: true,
        }
    }

    /// Calculate input tokens based on system prompt and messages
    /// Simulates realistic token counts for different scenarios
    fn calculate_input_tokens(&self, system_prompt: &str, messages: &[Message]) -> i32 {
        // Check if this is a compaction call
        let is_compaction_call = messages.len() == 1
            && messages[0].content.iter().any(|c| {
                if let MessageContent::Text(text) = c {
                    text.text.to_lowercase().contains("summarize")
                } else {
                    false
                }
            });

        if is_compaction_call {
            // For compaction: system prompt length is a good proxy for conversation size
            // Base: 6000 (system) + conversation content embedded in prompt
            6000 + (system_prompt.len() as i32 / 4).max(400)
        } else {
            // Regular call: system prompt + messages
            let system_tokens = if system_prompt.is_empty() { 0 } else { 6000 };

            let message_tokens: i32 = messages
                .iter()
                .map(|msg| {
                    let mut tokens = 100;
                    for content in &msg.content {
                        if let MessageContent::Text(text) = content {
                            if text.text.contains("long_tool_call") {
                                tokens += 15000;
                            }
                        }
                    }
                    tokens
                })
                .sum();

            system_tokens + message_tokens
        }
    }

    /// Calculate output tokens based on response type
    fn calculate_output_tokens(&self, is_compaction: bool, messages: &[Message]) -> i32 {
        if is_compaction {
            // Compaction produces a compact summary
            200
        } else {
            // Regular responses vary by content
            let has_hello = messages.iter().any(|msg| {
                msg.content.iter().any(|c| {
                    if let MessageContent::Text(text) = c {
                        text.text.to_lowercase().contains("hello")
                    } else {
                        false
                    }
                })
            });

            if has_hello {
                50 // Simple greeting response
            } else {
                100 // Default response
            }
        }
    }
}

#[async_trait]
impl Provider for MockCompactionProvider {
    fn manages_own_context(&self) -> bool {
        self.manages_own_context
    }

    async fn stream(
        &self,
        _model_config: &ModelConfig,
        system_prompt: &str,
        messages: &[Message],
        _tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        // Check if this is a compaction call (message contains "summarize")
        let is_compaction = messages.iter().any(|msg| {
            msg.content.iter().any(|content| {
                if let MessageContent::Text(text) = content {
                    text.text.to_lowercase().contains("summarize")
                } else {
                    false
                }
            })
        });

        // Calculate realistic token counts based on actual content
        let input_tokens = self.calculate_input_tokens(system_prompt, messages);
        let output_tokens = self.calculate_output_tokens(is_compaction, messages);

        // Simulate context limit: if input > 20k tokens and we haven't compacted yet, fail
        const CONTEXT_LIMIT: i32 = 20000;
        if !is_compaction
            && input_tokens > CONTEXT_LIMIT
            && !self.has_compacted.load(Ordering::SeqCst)
        {
            return Err(ProviderError::ContextLengthExceeded(format!(
                "Context limit exceeded: {} > {}",
                input_tokens, CONTEXT_LIMIT
            )));
        }

        // If this is a compaction call, mark that we've compacted
        if is_compaction {
            self.has_compacted.store(true, Ordering::SeqCst);
        }

        // Generate response
        let message = if is_compaction {
            Message::assistant().with_text("<mock summary of conversation>")
        } else {
            let response_text = if messages.iter().any(|msg| {
                msg.content.iter().any(|c| {
                    if let MessageContent::Text(text) = c {
                        text.text.to_lowercase().contains("hello")
                    } else {
                        false
                    }
                })
            }) {
                "Hi there! How can I help you?"
            } else {
                "This is a mock response."
            };
            Message::assistant().with_text(response_text)
        };

        let usage = ProviderUsage::new(
            "mock-model".to_string(),
            Usage::new(
                Some(input_tokens),
                Some(output_tokens),
                Some(input_tokens + output_tokens),
            ),
        );

        Ok(stream_from_single_message(message, usage))
    }

    fn get_name(&self) -> &str {
        "mock-compaction"
    }
}

impl goose::providers::base::ProviderDescriptor for MockCompactionProvider {
    fn metadata() -> ProviderMetadata {
        ProviderMetadata {
            name: "mock".to_string(),
            display_name: "Mock Compaction Provider".to_string(),
            description: "Mock provider for compaction testing".to_string(),
            default_model: "mock-model".to_string(),
            known_models: vec![],
            model_doc_link: "".to_string(),
            config_keys: vec![],
            setup_steps: vec![],
            setup: None,
            deprecated: None,
        }
    }
}

impl ProviderDef for MockCompactionProvider {
    type Provider = Self;

    fn from_env(
        _extensions: Vec<goose::config::ExtensionConfig>,
        _tls_config: Option<goose::providers::api_client::TlsConfig>,
    ) -> futures::future::BoxFuture<'static, anyhow::Result<Self>> {
        Box::pin(async { Ok(Self::new()) })
    }
}

/// Helper: Set up a test session with initial messages and token counts
async fn setup_test_session(
    agent: &Agent,
    temp_dir: &TempDir,
    session_name: &str,
    messages: Vec<Message>,
) -> Result<Session> {
    let session = agent
        .config
        .session_manager
        .create_session(
            temp_dir.path().to_path_buf(),
            session_name.to_string(),
            SessionType::Hidden,
            GooseMode::default(),
        )
        .await?;

    let conversation = Conversation::new_unvalidated(messages);
    agent
        .config
        .session_manager
        .replace_conversation(&session.id, &conversation)
        .await?;

    // Set initial token counts
    agent
        .config
        .session_manager
        .update(&session.id)
        .usage(Usage::new(Some(600), Some(400), Some(1000)))
        .accumulated_usage(Usage::new(Some(600), Some(400), Some(1000)))
        .apply()
        .await?;

    Ok(session)
}

#[tokio::test]
async fn context_owning_provider_rejects_clear_and_compact_without_changing_session() -> Result<()>
{
    let temp_dir = TempDir::new()?;
    let agent = Agent::new();
    let messages = vec![
        Message::user().with_text("Remember this"),
        Message::assistant().with_text("I will"),
    ];
    let session = setup_test_session(
        &agent,
        &temp_dir,
        "context-owning-provider",
        messages.clone(),
    )
    .await?;
    let before = agent
        .config
        .session_manager
        .get_session(&session.id, true)
        .await?;
    let conversation_before = before.conversation.unwrap();
    let usage_before = before.usage;
    let provider = Arc::new(MockCompactionProvider::context_owning());
    agent
        .update_provider(provider, ModelConfig::new("mock-model"), &session.id)
        .await?;

    for command in ["clear", "compact"] {
        let error = agent
            .execute_command(&format!("/{command}"), &session.id)
            .await
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            format!(
                "/{command} is not available for provider 'mock-compaction' because it manages its own conversation context"
            )
        );
    }

    let unchanged = agent
        .config
        .session_manager
        .get_session(&session.id, true)
        .await?;
    assert_eq!(unchanged.conversation.unwrap(), conversation_before);
    assert_eq!(unchanged.usage, usage_before);

    Ok(())
}

/// Helper: Assert conversation has been compacted with proper message visibility
fn assert_conversation_compacted(conversation: &Conversation) {
    let messages = conversation.messages();
    assert!(!messages.is_empty(), "Conversation should not be empty");

    // Find the summary message (contains "mock summary")
    let summary_index = messages
        .iter()
        .position(|msg| {
            msg.content.iter().any(|content| {
                if let MessageContent::Text(text) = content {
                    text.text.contains("mock summary")
                } else {
                    false
                }
            })
        })
        .expect("Conversation should contain the summary message");

    let summary_msg = &messages[summary_index];

    // Assert summary message visibility
    assert!(
        summary_msg.is_agent_visible(),
        "Summary message should be agent visible"
    );
    assert!(
        !summary_msg.is_user_visible(),
        "Summary message should NOT be user visible"
    );

    // Check messages BEFORE the summary (the compacted original messages)
    // These should be made agent-invisible
    for (idx, msg) in messages.iter().enumerate() {
        if idx < summary_index {
            // Old messages before summary: agent can't see them
            assert!(
                !msg.is_agent_visible(),
                "Message before summary at index {} should be agent-invisible",
                idx
            );
        }
    }

    // Check for continuation message after summary
    // (Should exist and be agent-only)
    if summary_index + 1 < messages.len() {
        let continuation_msg = &messages[summary_index + 1];
        // Continuation message should contain instructions about not mentioning summary
        let has_continuation_text = continuation_msg.content.iter().any(|content| {
            if let MessageContent::Text(text) = content {
                text.text.contains("previous message contains a summary")
                    || text.text.contains("summarization occurred")
            } else {
                false
            }
        });

        if has_continuation_text {
            assert!(
                continuation_msg.is_agent_visible(),
                "Continuation message should be agent visible"
            );
            assert!(
                !continuation_msg.is_user_visible(),
                "Continuation message should NOT be user visible"
            );
        }
    }

    // The projected replay of the preserved user message is agent-only. Any
    // ordinary messages appended after it should remain visible to both sides.
    let continuation_end = summary_index + 2;
    for (idx, msg) in messages.iter().enumerate() {
        if idx >= continuation_end {
            assert!(
                msg.is_agent_visible(),
                "Message after compaction at index {} should be agent visible",
                idx
            );
            if msg.is_turn_context() {
                assert!(
                    !msg.is_user_visible(),
                    "Carried turn-context event should be user-invisible"
                );
            } else if idx == continuation_end && matches!(msg.role, rmcp::model::Role::User) {
                assert!(
                    !msg.is_user_visible(),
                    "Projected preserved user message should be user-invisible"
                );
            } else {
                assert!(
                    msg.is_user_visible(),
                    "Ordinary message after compaction at index {} should be user visible",
                    idx
                );
            }
        }
    }
}

#[tokio::test]
async fn test_manual_compaction_updates_token_counts_and_conversation() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let agent = Agent::new();

    // Setup session with initial messages
    // Each message ~100 tokens, so 4 messages = ~400 tokens in conversation
    let messages = vec![
        Message::user().with_text("Hello, can you help me with something?"),
        Message::assistant().with_text("Of course! What do you need help with?"),
        Message::user().with_text("I need to understand how compaction works."),
        Message::assistant()
            .with_text("Compaction is a process that summarizes conversation history."),
    ];

    let session = setup_test_session(&agent, &temp_dir, "manual-compact-test", messages).await?;

    // Setup mock provider
    let provider = Arc::new(MockCompactionProvider::new());
    agent
        .update_provider(provider, ModelConfig::new("mock-model"), &session.id)
        .await?;

    // Execute manual compaction
    let result = agent.execute_command("/compact", &session.id).await?;
    assert!(result.is_some(), "Compaction should return a result");

    // Verify token counts
    let updated_session = agent
        .config
        .session_manager
        .get_session(&session.id, true)
        .await?;

    // Expected token calculation for compaction:
    // During compaction, the 4 messages are embedded in the system prompt template
    // - Input: system prompt with embedded conversation + "Please summarize" message
    // - Output: summary (200 tokens)
    //
    // From mock provider calculation:
    // - System prompt (with 4 embedded messages): varies based on template + content
    // - Single "summarize" message: 100 tokens
    // - Total input observed: ~6100 tokens
    //
    // After compaction the baseline is the estimated retained conversation
    // (summary + continuation), not the provider-reported output count
    let input_after = updated_session
        .usage
        .input_tokens
        .expect("Input tokens should be set after compaction");
    assert!(
        input_after > 0 && input_after < 200,
        "Input tokens should be the estimated retained context (smaller than the mock's claimed 200 output tokens). Got: {}",
        input_after
    );
    assert_eq!(
        updated_session.usage.output_tokens, None,
        "Output tokens should be None after compaction (no new assistant output)"
    );
    assert_eq!(
        updated_session.usage.total_tokens,
        Some(input_after),
        "Total should equal input after compaction"
    );

    // Accumulated tokens increased by the compaction cost
    // Initial: 1000
    // Compaction input: ~6700 (system 6000 + compaction prompt + 4 messages;
    // the mock derives input tokens from the rendered prompt length, so the
    // band must absorb compaction.md wording changes)
    // Compaction output: 200
    let accumulated = updated_session.accumulated_usage.total_tokens.unwrap();
    assert!(
        (7300..=8600).contains(&accumulated),
        "Accumulated should be ~7900 (1000 initial + ~6700 input + 200 output). Got: {}",
        accumulated
    );

    // Verify conversation has been compacted
    let compacted_conversation = updated_session
        .conversation
        .expect("Session should have conversation");

    assert_conversation_compacted(&compacted_conversation);

    Ok(())
}

async fn run_command(agent: &Agent, session: &Session, command: &str) -> Result<Vec<AgentEvent>> {
    let session_config = SessionConfig {
        id: session.id.clone(),
        schedule_id: None,
        max_turns: None,
        retry_config: None,
    };
    let stream = agent
        .reply(
            Message::user().with_text(command),
            session_config,
            goose::agents::state_machine::enabled(),
            None,
        )
        .await?;
    Ok(stream
        .filter_map(|event| async move { event.ok() })
        .collect()
        .await)
}

fn text_of(events: &[AgentEvent]) -> String {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::Message(message) => Some(message.as_concat_text()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The text of every system notification in a turn: notices are what the user
/// is told, and they carry no text content.
fn notifications_of(events: &[AgentEvent]) -> String {
    use goose::conversation::message::MessageContentBlock;

    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::Message(message) => Some(message),
            _ => None,
        })
        .flat_map(|message| message.content.iter())
        .filter_map(|content| match content {
            MessageContentBlock::SystemNotification(notification) => Some(notification.msg.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The notices stored in a session, which is where they have to be for a
/// resumed session to explain what happened to history.
async fn stored_notices(agent: &Agent, session: &Session) -> String {
    use goose::conversation::message::MessageContentBlock;

    agent
        .config
        .session_manager
        .get_session(&session.id, true)
        .await
        .unwrap()
        .conversation
        .unwrap()
        .messages()
        .iter()
        .flat_map(|message| message.content.iter())
        .filter_map(|content| match content {
            MessageContentBlock::SystemNotification(notification) => Some(notification.msg.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

async fn set_pending_compaction_request(
    agent: &Agent,
    session: &Session,
    reason: &str,
    granted: bool,
) {
    set_pending_compaction_request_carrying(agent, session, reason, granted, None).await;
}

async fn set_pending_compaction_request_carrying(
    agent: &Agent,
    session: &Session,
    reason: &str,
    granted: bool,
    carry: Option<&str>,
) {
    use goose::agents::session_requests::{CompactionRequest, SessionRequestState};
    use goose::capabilities::{SessionPermissions, SESSION_MODIFICATION};

    let manager = &agent.config.session_manager;
    let mut session_data = manager.get_session(&session.id, false).await.unwrap();

    let mut permissions = SessionPermissions::read(&session_data.extension_data);
    if granted {
        permissions.grant(SESSION_MODIFICATION);
    } else {
        permissions.revoke(SESSION_MODIFICATION);
    }
    permissions
        .write_into(&mut session_data.extension_data)
        .unwrap();

    let state = SessionRequestState {
        pending_compaction: Some(CompactionRequest {
            reason: reason.to_string(),
            carry: carry.map(str::to_string),
            requested_at: 0,
        }),
        ..Default::default()
    };
    state.write_into(&mut session_data.extension_data).unwrap();

    manager
        .update(&session.id)
        .extension_data(session_data.extension_data)
        .apply()
        .await
        .unwrap();
}

fn four_message_history() -> Vec<Message> {
    vec![
        Message::user().with_id("m1").with_text("first question"),
        Message::assistant().with_id("m2").with_text("first answer"),
        Message::user().with_id("m3").with_text("second question"),
        Message::assistant()
            .with_id("m4")
            .with_text("second answer"),
    ]
}

#[tokio::test]
async fn a_denied_compaction_request_is_reported_and_changes_nothing() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let agent = Agent::new();
    let session =
        setup_test_session(&agent, &temp_dir, "denied-request", four_message_history()).await?;
    agent
        .update_provider(
            Arc::new(MockCompactionProvider::new()),
            ModelConfig::new("mock-model"),
            &session.id,
        )
        .await?;

    set_pending_compaction_request(&agent, &session, "the logs are no longer needed", false).await;
    let events = run_command(&agent, &session, "carry on").await?;

    let notices = notifications_of(&events);
    assert!(notices.contains("goose asked to compact this conversation"));
    assert!(notices.contains("the logs are no longer needed"));
    assert!(notices.contains("/permit session-modification"));

    let on_record = stored_notices(&agent, &session).await;
    assert!(
        on_record.contains("goose asked to compact this conversation"),
        "the notice must be on record, not only on screen"
    );

    assert!(agent
        .config
        .session_manager
        .list_compaction_events(&session.id)
        .await?
        .is_empty());

    let state = goose::agents::session_requests::SessionRequestState::read(
        &agent
            .config
            .session_manager
            .get_session(&session.id, false)
            .await?,
    );
    assert!(state.pending_compaction.is_none());
    assert!(state.denied_compaction.is_none());

    Ok(())
}

#[tokio::test]
async fn a_permitted_compaction_request_compacts_at_the_next_boundary() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let agent = Agent::new();
    let session = setup_test_session(
        &agent,
        &temp_dir,
        "permitted-request",
        four_message_history(),
    )
    .await?;
    agent
        .update_provider(
            Arc::new(MockCompactionProvider::new()),
            ModelConfig::new("mock-model"),
            &session.id,
        )
        .await?;

    set_pending_compaction_request(&agent, &session, "the first exchange is done", true).await;
    let events = run_command(&agent, &session, "carry on").await?;

    let notices = notifications_of(&events);
    assert!(notices.contains("Compacted at goose's request"));
    assert!(notices.contains("the first exchange is done"));

    let on_record = stored_notices(&agent, &session).await;
    assert!(
        on_record.contains("Compacted at goose's request"),
        "the notice must be on record, not only on screen"
    );

    let recorded = agent
        .config
        .session_manager
        .list_compaction_events(&session.id)
        .await?;
    assert_eq!(recorded.len(), 1);
    assert_eq!(
        recorded[0].event.trigger,
        goose::session::CompactionTrigger::Model
    );
    assert_eq!(
        recorded[0].event.reason.as_deref(),
        Some("the first exchange is done")
    );

    Ok(())
}

#[tokio::test]
async fn a_carried_note_survives_the_compaction_verbatim() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let agent = Agent::new();
    let session =
        setup_test_session(&agent, &temp_dir, "carried-note", four_message_history()).await?;
    agent
        .update_provider(
            Arc::new(MockCompactionProvider::new()),
            ModelConfig::new("mock-model"),
            &session.id,
        )
        .await?;

    let note = "step 3 of 7: resume with /tmp/step3.json, key abc123";
    set_pending_compaction_request_carrying(&agent, &session, "the step is done", true, Some(note))
        .await;
    run_command(&agent, &session, "carry on").await?;

    let stored = agent
        .config
        .session_manager
        .get_session(&session.id, true)
        .await?;
    let in_context = stored
        .conversation
        .clone()
        .unwrap_or_default()
        .agent_visible_messages()
        .iter()
        .map(|message| message.as_concat_text())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        in_context.contains(note),
        "the note must survive verbatim, not only inside the summary: {in_context}"
    );

    let recorded = agent
        .config
        .session_manager
        .list_compaction_events(&session.id)
        .await?;
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].event.carry.as_deref(), Some(note));

    let listing = text_of(&run_command(&agent, &session, "/archive").await?);
    assert!(listing.contains("kept a note:"));
    let dump = text_of(&run_command(&agent, &session, "/archive last").await?);
    assert!(dump.contains("**Kept across this compaction**"));
    assert!(dump.contains(note));

    Ok(())
}

#[tokio::test]
async fn a_refused_request_drops_its_note() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let agent = Agent::new();
    let session = setup_test_session(
        &agent,
        &temp_dir,
        "refused-note",
        vec![Message::user().with_id("m1").with_text("only one message")],
    )
    .await?;
    agent
        .update_provider(
            Arc::new(MockCompactionProvider::new()),
            ModelConfig::new("mock-model"),
            &session.id,
        )
        .await?;

    set_pending_compaction_request_carrying(
        &agent,
        &session,
        "too early",
        true,
        Some("must not survive"),
    )
    .await;
    let events = run_command(&agent, &session, "carry on").await?;

    let text = text_of(&events);
    assert!(text.contains("No compaction this time"));
    assert!(text.contains("was not kept"));
    assert!(!text.contains("must not survive"));
    assert!(agent
        .config
        .session_manager
        .list_compaction_events(&session.id)
        .await?
        .is_empty());

    Ok(())
}

#[tokio::test]
async fn a_granted_request_on_a_short_conversation_is_refused() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let agent = Agent::new();
    let session = setup_test_session(
        &agent,
        &temp_dir,
        "short-request",
        vec![Message::user().with_id("m1").with_text("only one message")],
    )
    .await?;
    agent
        .update_provider(
            Arc::new(MockCompactionProvider::new()),
            ModelConfig::new("mock-model"),
            &session.id,
        )
        .await?;

    set_pending_compaction_request(&agent, &session, "too early", true).await;
    let events = run_command(&agent, &session, "carry on").await?;

    assert!(text_of(&events).contains("No compaction this time"));
    assert!(agent
        .config
        .session_manager
        .list_compaction_events(&session.id)
        .await?
        .is_empty());

    Ok(())
}

async fn agent_visible_message_ids(agent: &Agent, session: &Session) -> Vec<String> {
    agent
        .config
        .session_manager
        .get_session(&session.id, true)
        .await
        .unwrap()
        .conversation
        .unwrap()
        .messages()
        .iter()
        .filter(|message| message.is_agent_visible())
        .filter_map(|message| message.id.clone())
        .collect()
}

#[tokio::test]
async fn clear_archives_the_conversation_and_keeps_the_messages() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let agent = Agent::new();
    let session = setup_test_session(
        &agent,
        &temp_dir,
        "clear-archives",
        vec![
            Message::user().with_id("m1").with_text("Remember this"),
            Message::assistant().with_id("m2").with_text("I will"),
        ],
    )
    .await?;
    agent
        .update_provider(
            Arc::new(MockCompactionProvider::new()),
            ModelConfig::new("mock-model"),
            &session.id,
        )
        .await?;

    let events = run_command(&agent, &session, "/clear").await?;
    assert!(events
        .iter()
        .any(|event| matches!(event, AgentEvent::HistoryReplaced(_))));

    let stored = agent
        .config
        .session_manager
        .get_session(&session.id, true)
        .await?
        .conversation
        .unwrap();
    for id in ["m1", "m2"] {
        let message = stored
            .messages()
            .iter()
            .find(|message| message.id.as_deref() == Some(id))
            .expect("cleared messages must stay on record");
        assert!(!message.is_agent_visible());
        assert!(message.is_user_visible());
    }
    assert!(agent_visible_message_ids(&agent, &session).await.is_empty());

    let events = agent
        .config
        .session_manager
        .list_compaction_events(&session.id)
        .await?;
    assert_eq!(events.len(), 1);
    assert_eq!(
        events[0].event.trigger,
        goose::session::CompactionTrigger::Clear
    );
    for id in ["m1", "m2"] {
        assert!(
            events[0]
                .event
                .archived_message_ids
                .contains(&id.to_string()),
            "the cleared messages must be recorded as archived"
        );
    }

    Ok(())
}

#[tokio::test]
async fn clear_destroy_deletes_the_messages() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let agent = Agent::new();
    let session = setup_test_session(
        &agent,
        &temp_dir,
        "clear-destroys",
        vec![
            Message::user().with_id("m1").with_text("Remember this"),
            Message::assistant().with_id("m2").with_text("I will"),
        ],
    )
    .await?;
    agent
        .update_provider(
            Arc::new(MockCompactionProvider::new()),
            ModelConfig::new("mock-model"),
            &session.id,
        )
        .await?;

    run_command(&agent, &session, "/clear --destroy").await?;

    let stored = agent
        .config
        .session_manager
        .get_session(&session.id, true)
        .await?
        .conversation
        .unwrap();
    assert!(stored
        .messages()
        .iter()
        .all(|message| !matches!(message.id.as_deref(), Some("m1") | Some("m2"))));

    Ok(())
}

#[tokio::test]
async fn status_and_archive_report_the_archived_history() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let agent = Agent::new();
    let session = setup_test_session(
        &agent,
        &temp_dir,
        "archive-report",
        vec![
            Message::user().with_id("m1").with_text("Remember this"),
            Message::assistant().with_id("m2").with_text("I will"),
        ],
    )
    .await?;
    agent
        .update_provider(
            Arc::new(MockCompactionProvider::new()),
            ModelConfig::new("mock-model"),
            &session.id,
        )
        .await?;

    run_command(&agent, &session, "/clear").await?;

    let status = text_of(&run_command(&agent, &session, "/status").await?);
    assert!(status.contains("Archived history:"));
    assert!(status.contains("in 1 event(s)"));

    let listing = text_of(&run_command(&agent, &session, "/archive").await?);
    assert!(listing.contains("**Archived history**: 1 event(s)"));

    let dump = text_of(&run_command(&agent, &session, "/archive last").await?);
    assert!(dump.contains("Remember this"));
    assert!(dump.contains("I will"));

    Ok(())
}

#[tokio::test]
async fn permit_and_deny_are_the_only_way_to_grant_a_session_capability() -> Result<()> {
    use goose::capabilities::{SessionPermissions, SESSION_MODIFICATION};

    let temp_dir = TempDir::new()?;
    let agent = Agent::new();
    let session = setup_test_session(
        &agent,
        &temp_dir,
        "capability-gate",
        vec![Message::user().with_id("m1").with_text("Remember this")],
    )
    .await?;
    agent
        .update_provider(
            Arc::new(MockCompactionProvider::new()),
            ModelConfig::new("mock-model"),
            &session.id,
        )
        .await?;

    async fn granted(agent: &Agent, id: &str) -> bool {
        let session = agent
            .config
            .session_manager
            .get_session(id, false)
            .await
            .unwrap();
        SessionPermissions::read(&session.extension_data).is_granted(SESSION_MODIFICATION)
    }

    let listing = text_of(&run_command(&agent, &session, "/permit").await?);
    assert!(listing.contains("session-modification"));
    assert!(listing.contains("[denied]"));
    assert!(!granted(&agent, &session.id).await);

    let unknown = text_of(&run_command(&agent, &session, "/permit nonsense").await?);
    assert!(unknown.contains("is not a capability"));

    let granted_text =
        text_of(&run_command(&agent, &session, "/permit session-modification").await?);
    assert!(granted_text.contains("`session-modification` is permitted for this session"));
    assert!(granted(&agent, &session.id).await);

    let again = text_of(&run_command(&agent, &session, "/permit session-modification").await?);
    assert!(again.contains("already in effect"));

    let denied_text = text_of(&run_command(&agent, &session, "/deny session-modification").await?);
    assert!(denied_text.contains("`session-modification` is denied for this session"));
    assert!(!granted(&agent, &session.id).await);

    Ok(())
}

#[tokio::test]
async fn grants_do_not_leak_into_another_session() -> Result<()> {
    use goose::capabilities::{SessionPermissions, SESSION_MODIFICATION};

    let temp_dir = TempDir::new()?;
    let agent = Agent::new();
    let first = setup_test_session(
        &agent,
        &temp_dir,
        "granted-session",
        vec![Message::user().with_id("m1").with_text("Remember this")],
    )
    .await?;
    let second = setup_test_session(
        &agent,
        &temp_dir,
        "other-session",
        vec![Message::user().with_id("m1").with_text("Remember this")],
    )
    .await?;
    agent
        .update_provider(
            Arc::new(MockCompactionProvider::new()),
            ModelConfig::new("mock-model"),
            &first.id,
        )
        .await?;

    run_command(&agent, &first, "/permit session-modification").await?;

    let other = agent
        .config
        .session_manager
        .get_session(&second.id, false)
        .await?;
    assert!(!SessionPermissions::read(&other.extension_data).is_granted(SESSION_MODIFICATION));

    Ok(())
}

#[tokio::test]
async fn test_auto_compaction_during_reply() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let agent = Agent::new();

    // Setup session with many messages to have substantial context
    // 20 exchanges = 40 messages * 100 tokens = ~4000 tokens in conversation
    let mut messages = vec![];
    for i in 0..20 {
        messages.push(Message::user().with_text(format!("User message {}", i)));
        messages.push(Message::assistant().with_text(format!("Assistant response {}", i)));
    }

    let session = setup_test_session(&agent, &temp_dir, "auto-compact-test", messages).await?;

    // Capture initial context size before triggering reply
    // Should be: system (6000) + 40 messages (4000) = ~10000 tokens
    let initial_session = agent
        .config
        .session_manager
        .get_session(&session.id, true)
        .await?;
    let initial_input_tokens = initial_session.usage.input_tokens.unwrap_or(0);

    // Setup mock provider (no context limit enforcement)
    let provider = Arc::new(MockCompactionProvider::new());
    agent
        .update_provider(provider, ModelConfig::new("mock-model"), &session.id)
        .await?;

    // Trigger a reply
    // Expected tokens for reply:
    // - Input: system (6000) + 40 messages (4000) + new user message (100) = 10100 tokens
    // - Output: regular response (100 tokens)
    let user_message = Message::user().with_text("Tell me more about compaction");

    let session_config = SessionConfig {
        id: session.id.clone(),
        schedule_id: None,
        max_turns: None,
        retry_config: None,
    };

    let reply_stream = agent
        .reply(
            user_message,
            session_config,
            goose::agents::state_machine::enabled(),
            None,
        )
        .await?;
    tokio::pin!(reply_stream);

    // Track compaction and context size changes
    let mut compaction_occurred = false;
    let mut input_tokens_after_compaction: Option<i32> = None;

    while let Some(event_result) = reply_stream.next().await {
        match event_result {
            Ok(AgentEvent::HistoryReplaced(_)) => {
                compaction_occurred = true;

                // Capture the input tokens immediately after compaction
                let session_after_compact = agent
                    .config
                    .session_manager
                    .get_session(&session.id, true)
                    .await?;
                input_tokens_after_compaction = session_after_compact.usage.input_tokens;
            }
            Ok(_) => {}
            Err(e) => return Err(e),
        }
    }

    let updated_session = agent
        .config
        .session_manager
        .get_session(&session.id, true)
        .await?;

    if compaction_occurred {
        // Verify that current input context decreased after compaction
        let tokens_after =
            input_tokens_after_compaction.expect("Should have captured tokens after compaction");

        // Before compaction: system (6000) + 40 messages (4000) = 10,000 tokens
        // After compaction: only the summary (200 tokens) - this becomes the new input
        assert!(
            tokens_after < initial_input_tokens,
            "Input tokens should decrease after compaction. Before: {}, After: {}",
            initial_input_tokens,
            tokens_after
        );

        // After compaction, input should be exactly the summary: 200 tokens
        assert_eq!(
            tokens_after, 200,
            "Input tokens after compaction should be exactly 200 (summary). Got: {}",
            tokens_after
        );

        // After the subsequent reply, the current window includes:
        // - system (6000) + summary (200) + new user message (100) + reply (100) = 6400
        let final_input = updated_session.usage.input_tokens.unwrap();
        let final_output = updated_session.usage.output_tokens.unwrap();
        let final_total = updated_session.usage.total_tokens.unwrap();

        assert!(
            final_input >= 6000,
            "Final input should include at least system prompt (6000). Got: {}",
            final_input
        );
        assert_eq!(
            final_output, 100,
            "Final output should be 100 tokens (default response). Got: {}",
            final_output
        );
        assert_eq!(
            final_total,
            final_input + final_output,
            "Final total should equal input + output"
        );

        // Accumulated tokens should include:
        // - Initial: 1000
        // - Compaction: ~10,400 input + 200 output = 10,600
        // - Reply: ~6,300 input + 100 output = 6,400
        // Total: 1000 + 10,600 + 6,400 = 18,000
        let accumulated = updated_session.accumulated_usage.total_tokens.unwrap();
        assert!(
            (17000..=19000).contains(&accumulated),
            "Accumulated should be ~18,000 (initial + compaction + reply). Got: {}",
            accumulated
        );
    } else {
        // If no compaction, accumulated should include reply cost
        // - Initial: 1000
        // - Reply: system (6000) + 40 messages (4000) + new message (100) = 10,100 input
        // - Reply output: 100
        // Total: 1000 + 10,100 + 100 = 11,200
        let accumulated = updated_session.accumulated_usage.total_tokens.unwrap();
        assert!(
            (11000..=11500).contains(&accumulated),
            "Accumulated should be ~11,200 (initial + reply). Got: {}",
            accumulated
        );

        // Current window should be: 10,100 input + 100 output = 10,200
        let final_input = updated_session.usage.input_tokens.unwrap();
        let final_output = updated_session.usage.output_tokens.unwrap();

        assert!(
            (10000..=10500).contains(&final_input),
            "Input should be ~10,100. Got: {}",
            final_input
        );
        assert_eq!(
            final_output, 100,
            "Output should be 100. Got: {}",
            final_output
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_context_limit_recovery_compaction() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let agent = Agent::new();

    // Setup session with messages that will push context over the limit
    // Each message = 100 tokens, but we'll add a large one
    let messages = vec![
        Message::user().with_text("Hello"),
        Message::assistant().with_text("Hi there"),
        Message::user().with_text("Can you process this long_tool_call result?"),
        Message::assistant().with_text("Processing..."),
    ];
    // Token calculation:
    // - 3 regular messages: 300 tokens
    // - 1 message with "long_tool_call": 100 + 15000 = 15100 tokens
    // - Total conversation: ~15400 tokens
    // - With system prompt (6000): 21400 tokens

    let session = setup_test_session(&agent, &temp_dir, "context-limit-test", messages).await?;

    // Setup mock provider with context limit of 20000 tokens
    // Initial context (6000 system + 15400 messages = 21400) exceeds this limit
    let provider = Arc::new(MockCompactionProvider::new());
    agent
        .update_provider(provider, ModelConfig::new("mock-model"), &session.id)
        .await?;

    // Try to send a message - should trigger context limit, then recover via compaction
    let session_config = SessionConfig {
        id: session.id.clone(),
        schedule_id: None,
        max_turns: None,
        retry_config: None,
    };

    let reply_stream = agent
        .reply(
            Message::user().with_text("Tell me more"),
            session_config,
            goose::agents::state_machine::enabled(),
            None,
        )
        .await?;
    tokio::pin!(reply_stream);

    // Track compaction and context size changes
    let mut compaction_occurred = false;
    let mut got_response = false;
    let mut input_tokens_after_compaction: Option<i32> = None;

    while let Some(event_result) = reply_stream.next().await {
        match event_result {
            Ok(AgentEvent::HistoryReplaced(_)) => {
                compaction_occurred = true;

                // Capture the input tokens immediately after compaction
                let session_after_compact = agent
                    .config
                    .session_manager
                    .get_session(&session.id, true)
                    .await?;
                input_tokens_after_compaction = session_after_compact.usage.input_tokens;
            }
            Ok(AgentEvent::Message(msg)) => {
                // Check if we got a real response (not just a notification)
                if msg
                    .content
                    .iter()
                    .any(|c| matches!(c, MessageContent::Text(_)))
                {
                    got_response = true;
                }
            }
            Ok(_) => {}
            Err(e) => return Err(e),
        }
    }

    // Verify recovery occurred
    assert!(
        compaction_occurred,
        "Compaction should have occurred due to context limit (>20000 tokens)"
    );
    assert!(
        got_response,
        "Should have received a response after recovery"
    );

    // Verify token counts
    let updated_session = agent
        .config
        .session_manager
        .get_session(&session.id, true)
        .await?;

    // Expected token flow:
    // 1. Initial attempt: >20000 tokens -> Context limit exceeded
    // 2. Compaction triggered:
    //    - Input: system prompt + messages (including long_tool_call with 15k tokens)
    //    - Output: 200 tokens (summary, as claimed by the mock)
    //    - New context size: estimated tokens of the retained conversation
    // 3. Retry with compacted context:
    //    - Input: system prompt + summary + new message
    //    - Output: 100 tokens (response)

    // Verify that current input context is dramatically reduced after compaction
    let tokens_after =
        input_tokens_after_compaction.expect("Should have captured tokens after compaction");

    // Before: system (6000) + long_tool_call messages (~15,400) = 21,400 (exceeded limit!)
    assert!(
        tokens_after > 0 && tokens_after < 200,
        "Input tokens after compaction should be the estimated retained context (under the mock's claimed 200). Got: {}",
        tokens_after
    );

    // The compacted context is now well under the 20k limit
    assert!(
        tokens_after < 20000,
        "Compacted context should be under 20k limit. Got: {}",
        tokens_after
    );

    // Check the final token state after recovery
    // Note: The session state reflects the RETRY call (after compaction),
    // which only sees agent-visible messages (summary + continuation + user message)
    let final_input = updated_session.usage.input_tokens.unwrap();
    let final_output = updated_session.usage.output_tokens;
    let final_total = updated_session.usage.total_tokens.unwrap();

    // After compaction, the retry only sees agent-visible messages:
    // Input: system (6000) + summary (~100) + continuation (~100) + user message (~100) = ~6300
    // Output: 200 (mock detects "summarized" in continuation as compaction)
    // Total: ~6500
    assert!(
        (6000..=6600).contains(&final_input),
        "Final input should reflect retry with agent-visible messages (~6300). Got: {}",
        final_input
    );

    assert_eq!(
        final_output,
        Some(200),
        "Final output should be 200 (mock detects continuation as compaction). Got: {:?}",
        final_output
    );

    assert_eq!(
        final_total,
        final_input + final_output.unwrap(),
        "Final total should equal input + output"
    );

    // Accumulated tokens should include all operations:
    // - Initial: 1000
    // - Compaction: ~6400 input (mock uses system_prompt.len()/4) + 200 output = ~6600
    // - Reply: ~6500 input + 200 output = ~6700
    // Total: 1000 + 6600 + 6700 = ~14300
    let accumulated = updated_session.accumulated_usage.total_tokens.unwrap();
    assert!(
        (13000..=16000).contains(&accumulated),
        "Accumulated should be ~14300 (initial + compaction + reply). Got: {}",
        accumulated
    );

    // Verify that the conversation was compacted
    let updated_conversation = updated_session
        .conversation
        .expect("Session should have conversation");
    assert_conversation_compacted(&updated_conversation);

    Ok(())
}

/// A compaction request reports its stage through the task-local sink the
/// caller installed: prefilling while the summarizer request is in flight, then
/// rewriting the context once generation starts. The sink must cover stream
/// consumption, not just the `reply` call, because the state-machine loop
/// compacts while its stream is being polled.
#[tokio::test]
async fn reports_compaction_stages_on_both_loops() -> Result<()> {
    for use_state_machine in [false, true] {
        let temp_dir = TempDir::new()?;
        let agent = Agent::new();
        let messages = vec![
            Message::user().with_text("Hello, can you help me with something?"),
            Message::assistant().with_text("Of course! What do you need help with?"),
        ];
        let session = setup_test_session(&agent, &temp_dir, "compact-stages", messages).await?;
        let provider = Arc::new(MockCompactionProvider::new());
        agent
            .update_provider(provider, ModelConfig::new("mock-model"), &session.id)
            .await?;

        let stages = Arc::new(Mutex::new(Vec::new()));
        let sink = {
            let stages = Arc::clone(&stages);
            Arc::new(move |stage| stages.lock().unwrap().push(stage))
        };
        let session_config = SessionConfig {
            id: session.id.clone(),
            schedule_id: None,
            max_turns: None,
            retry_config: None,
        };

        goose::session_context::with_stage_sink(Some(sink), async {
            let mut stream = agent
                .reply(
                    Message::user().with_text("/compact"),
                    session_config,
                    use_state_machine,
                    None,
                )
                .await?;
            while let Some(event) = stream.next().await {
                event?;
            }
            Ok::<(), anyhow::Error>(())
        })
        .await?;

        assert_eq!(
            *stages.lock().unwrap(),
            vec![LlmStage::Prefilling, LlmStage::RewritingContext],
            "state_machine={use_state_machine}"
        );
    }

    Ok(())
}
