//! What a decision provider answered for each tool request, and what the
//! permission judge said about the same request.
//!
//! These rows exist so a shadow classifier can be compared against the judge
//! on real traffic; they are never read back to make a permission decision.

/// One recorded classification. `read_only` is the gated verdict the shadow
/// classifier would have acted on, where the row came from the permission
/// shadow; it is `None` for rows written by a tool that steers instead of
/// gating. `probability` and `confidence` are the raw answer, so any other
/// threshold can be replayed from the log.
#[derive(Debug, Clone)]
pub struct JevDecisionRecord {
    pub session_id: String,
    pub request_id: String,
    pub tool_name: String,
    pub arguments: String,
    pub read_only: Option<bool>,
    pub probability: f64,
    pub confidence: f64,
    pub model: String,
    pub latency_ms: i64,
    pub input_tokens: Option<i64>,
    pub cost: Option<f64>,
    pub judge_read_only: Option<bool>,
    pub outcome: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StoredJevDecision {
    pub id: i64,
    pub created_at: String,
    pub request_id: String,
    pub tool_name: String,
    pub read_only: Option<bool>,
    pub probability: f64,
    pub confidence: f64,
    pub model: String,
    pub judge_read_only: Option<bool>,
    pub outcome: String,
}
