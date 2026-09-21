use goose_provider_types::conversation::{
    message::{Message, MessageUsage},
    token_usage::ProviderUsage,
    Conversation,
};
use rmcp::model::ServerNotification;

/// The phase an in-flight LLM request is in. Reported so a client can show a
/// status indicator that reflects the request rather than inferring it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LlmStage {
    /// The request was sent and the model has not started streaming yet
    /// (time-to-first-token). Nothing is written to the output during this
    /// stage, so it needs a client-side indicator.
    Prefilling,
}

#[derive(Clone, Debug)]
pub enum AgentEvent {
    Message(Message),
    Usage(ProviderUsage),
    MessageUsage {
        message_id: Option<String>,
        usage: MessageUsage,
    },
    McpNotification((String, ServerNotification)),
    HistoryReplaced(Conversation),
    Stage(LlmStage),
}
