//! Lets the model ask a decision model for a typed answer.
//!
//! The decision model answers questions; it does not converse. Everything that
//! makes it useful — a typed answer, a probability, a confidence — is also what
//! makes it easy to misuse, so the caveats live in the tool description where
//! the model reads them before deciding whether to call it.

use anyhow::Result;
use async_trait::async_trait;
use goose_providers::decision::{
    DecisionAnswer, DecisionQuestion, DecisionRequest, DecisionResponse,
};
use indoc::indoc;
use rmcp::model::{
    CallToolResult, ContentBlock, Implementation, InitializeResult, JsonObject, ListToolsResult,
    ServerCapabilities, Tool, ToolAnnotations,
};
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::agents::extension::PlatformExtensionContext;
use crate::agents::mcp_client::{Error, McpClientTrait};
use crate::agents::tool_execution::ToolCallContext;
use crate::providers::decision_provider::decision_provider_from_config;
use crate::session::JevDecisionRecord;

pub static EXTENSION_NAME: &str = "jev";
pub const ASK_TOOL_NAME: &str = "ask_jev";
pub const STEER_TOOL_NAME: &str = "steer";

/// Below this, the question is not well enough posed for the answer to mean much.
pub const MIN_USEFUL_CONFIDENCE: f64 = 0.5;

/// How far ahead the leader must be before a steer is worth giving. The spread
/// is what tells a real ordering from a coin toss, and it never reaches the model.
pub const MIN_STEER_MARGIN: f64 = 0.20;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
enum AnswerKind {
    /// Yes or no, returned as the probability that the answer is true.
    Noul,
    /// One of a small set of options, returned with the full distribution.
    Choice,
    /// One level of an ordered rubric, returned with the full distribution.
    Score,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ChoiceOption {
    /// A short label for the option.
    label: String,
    /// What choosing this option means.
    meaning: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct AskJevParams {
    /// The question, in the words of whoever asked it. Pass it through as close
    /// to verbatim as you can.
    question: String,
    kind: AnswerKind,
    /// For kind "noul": what it means for the answer to be true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    true_meaning: Option<String>,
    /// For kind "noul": what it means for the answer to be false.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    false_meaning: Option<String>,
    /// For kind "choice": the options, two or more.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    options: Option<Vec<ChoiceOption>>,
    /// For kind "score": the ordered levels, lowest first.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    levels: Option<Vec<String>>,
    /// The facts the question should be judged against. Everything the answer
    /// needs must be here: the model cannot ask follow-up questions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    context: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct SteerCandidate {
    /// A short label for the direction.
    label: String,
    /// What pursuing this direction means.
    description: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct SteerParams {
    /// What is being decided: the research question these directions answer.
    question: String,
    /// The directions you are choosing between, in your own words. At least two.
    candidates: Vec<SteerCandidate>,
    /// The label of the direction you would pursue on your own judgement, before
    /// asking. Required: it is what makes the reply a change to your plan rather
    /// than an instruction, and what lets the steer be checked afterwards.
    prior: String,
    /// Facts the ordering should be judged against. Everything it depends on must
    /// be here; the model cannot ask follow-up questions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    context: Option<String>,
}

/// What a distribution is allowed to do: move the agent off its stated prior, or
/// nothing. There is deliberately no variant that returns the numbers.
#[derive(Debug, PartialEq)]
enum Steer {
    Move(String),
    NoSteer(NoSteerReason),
}

#[derive(Debug, PartialEq)]
enum NoSteerReason {
    /// Nothing separated the candidates.
    NotDistinguishable,
    /// The leader was already the stated prior.
    AgreedWithPrior,
}

impl NoSteerReason {
    fn outcome(&self) -> &'static str {
        match self {
            Self::NotDistinguishable => "no_steer_indistinguishable",
            Self::AgreedWithPrior => "no_steer_agreed",
        }
    }
}

/// Consumes the distribution. The margin and the confidence stay here: they
/// decide whether a steer happens and are never part of what the agent is told.
fn steer_from(ranked: &[(String, f64)], confidence: f64, prior: &str) -> Result<Steer, String> {
    let (leader, leader_probability) = ranked
        .first()
        .ok_or_else(|| "the decision model returned no distribution".to_string())?;
    let runner_up = ranked
        .get(1)
        .map(|(_, probability)| *probability)
        .unwrap_or(0.0);

    if confidence < MIN_USEFUL_CONFIDENCE || leader_probability - runner_up < MIN_STEER_MARGIN {
        return Ok(Steer::NoSteer(NoSteerReason::NotDistinguishable));
    }
    if leader == prior {
        return Ok(Steer::NoSteer(NoSteerReason::AgreedWithPrior));
    }
    Ok(Steer::Move(leader.clone()))
}

fn ranked(probabilities: &HashMap<String, f64>) -> Vec<(String, f64)> {
    let mut entries: Vec<(String, f64)> = probabilities
        .iter()
        .map(|(label, probability)| (label.clone(), *probability))
        .collect();
    entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    entries
}

pub struct JevClient {
    info: InitializeResult,
    context: PlatformExtensionContext,
}

impl JevClient {
    pub fn new(context: PlatformExtensionContext) -> Result<Self> {
        let info = InitializeResult::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(
                Implementation::new(EXTENSION_NAME.to_string(), "1.0.0".to_string())
                    .with_title("Jev Decisions"),
            )
            .with_instructions(
                indoc! {r#"
                ask_jev asks a decision model for a typed answer: a yes/no
                probability, one of a set of options, or a level from an ordered
                rubric. It answers questions; it does not converse, generate
                prose, or explain itself.

                Use it when the user asks you to consult it, or when a decision
                turns on a judgement you cannot make from the material you have.
                Pass the question through in the user's own words, and put every
                fact the answer depends on in context — it cannot ask you
                anything back.

                The answer is advisory. Say where it came from when you relay it.

                steer is different and is the one to prefer for research: you
                supply the directions you are choosing between and the one you
                would take on your own judgement, and what comes back is either a
                direction to consider instead or nothing at all. It never returns
                a number, so there is nothing in it to mistake for evidence.
            "#}
                .to_string(),
            );

        Ok(Self { info, context })
    }

    fn get_tools() -> Vec<Tool> {
        let schema = schema_for!(AskJevParams);
        let schema_value =
            serde_json::to_value(schema).expect("Failed to serialize AskJevParams schema");
        let steer_schema = schema_for!(SteerParams);
        let steer_schema_value =
            serde_json::to_value(steer_schema).expect("Failed to serialize SteerParams schema");

        vec![
            Tool::new(
                ASK_TOOL_NAME.to_string(),
                indoc! {r#"
                Ask a decision model for a typed answer. It returns a number, not
                prose: a yes/no probability, one of a set of options, or a level
                from an ordered rubric.

                Read this before calling it.

                - It cannot explain itself and it cannot ask you a question back.
                  Everything the answer depends on has to be in the question or in
                  context.
                - You choose the shape of the answer with 'kind'. There is no
                  default and no free-form mode.
                - Pass the question through in the user's own words. Do not
                  reword, narrow or tidy it: the wording is the answer. Measured
                  on the same underlying facts, rewording the criteria moved the
                  answer from 0.75 to 0.95.
                - The answer is advisory. It is not calibrated for this project
                  and it does not know your code, your history or your
                  constraints beyond what you put in context.
                - Do not use it to decide whether a tool call is safe, and do not
                  use it to justify a decision you have already made. Ask when
                  you would genuinely act on either answer.
                - 'noul' answers carry a probability but no confidence. 'choice'
                  and 'score' carry both.
                - Below about 0.5 confidence the question is not well posed.
                  Sharpen or split it and ask again rather than acting on the
                  number.

                Tell the user what you asked and what came back, including the
                confidence.
            "#}
                .to_string(),
                schema_value.as_object().unwrap().clone(),
            )
            .annotate(ToolAnnotations::from_raw(
                Some("Ask a decision model".to_string()),
                Some(true),
                Some(false),
                Some(true),
                Some(true),
            )),
            Tool::new(
                STEER_TOOL_NAME.to_string(),
                indoc! {r#"
                Ask which of the directions you proposed to pursue next.

                You supply the question, the candidate directions, and — required —
                the direction you would take on your own judgement. The ordering is
                consumed inside goose; what comes back is either a direction to
                consider instead of your prior, or an explicit no-steer.

                What comes back is deliberately never a number, a probability or a
                confidence. There is nothing in it to treat as evidence: it is one
                model's ordering of options you wrote, it cannot explain itself, it
                is calibrated against nothing in this project, and you may override
                it. When the ordering cannot move you off your stated prior you are
                told nothing at all — that is not a confirmation of your prior.

                Do not use this to settle a factual question, and never to decide
                whether a tool call is safe. Use it when you have several directions
                you could investigate next and no strong reason to prefer one.
            "#}
                .to_string(),
                steer_schema_value.as_object().unwrap().clone(),
            )
            .annotate(ToolAnnotations::from_raw(
                Some("Steer research".to_string()),
                Some(true),
                Some(false),
                Some(true),
                Some(true),
            )),
        ]
    }

    fn build_request(params: &AskJevParams) -> Result<(DecisionRequest, String), String> {
        let question = params.question.trim();
        if question.is_empty() {
            return Err("the question is empty".to_string());
        }

        let model = decision_provider_from_config(None)
            .ok_or_else(|| {
                "no decision provider is configured; set GOOSE_DECISION_PROVIDER with \
                 OPENROUTER_API_KEY or TYPESAFE_API_KEY"
                    .to_string()
            })?
            .model;

        let state = json!({
            "background": params.context.clone().unwrap_or_default(),
        });

        let decision_question = match params.kind {
            AnswerKind::Noul => {
                let (true_meaning, false_meaning) =
                    match (&params.true_meaning, &params.false_meaning) {
                        (Some(true_meaning), Some(false_meaning)) => {
                            (true_meaning.clone(), false_meaning.clone())
                        }
                        _ => {
                            return Err(
                                "kind 'noul' needs both true_meaning and false_meaning".to_string()
                            )
                        }
                    };
                DecisionQuestion::Noul {
                    instructions: question.to_string(),
                    criteria: Some(goose_providers::decision::NoulCriteria {
                        true_description: true_meaning,
                        false_description: false_meaning,
                    }),
                }
            }
            AnswerKind::Choice => {
                let options = params
                    .options
                    .as_ref()
                    .filter(|options| options.len() >= 2)
                    .ok_or_else(|| "kind 'choice' needs at least two options".to_string())?;
                DecisionQuestion::Choice {
                    instructions: question.to_string(),
                    criteria: options
                        .iter()
                        .map(|option| (option.label.clone(), option.meaning.clone()))
                        .collect::<HashMap<_, _>>(),
                }
            }
            AnswerKind::Score => {
                let levels = params
                    .levels
                    .as_ref()
                    .filter(|levels| !levels.is_empty())
                    .ok_or_else(|| "kind 'score' needs at least one level".to_string())?;
                DecisionQuestion::Score {
                    instructions: question.to_string(),
                    criteria: levels.clone(),
                }
            }
        };

        let echo = format!(
            "question sent: {question}\nanswer shape: {}",
            match params.kind {
                AnswerKind::Noul => "noul (probability that it is true)",
                AnswerKind::Choice => "choice",
                AnswerKind::Score => "score",
            }
        );

        Ok((
            DecisionRequest {
                model,
                state,
                questions: HashMap::from([("answer".to_string(), decision_question)]),
            },
            echo,
        ))
    }

    async fn ask(&self, params: &AskJevParams) -> Result<String, String> {
        let (request, echo) = Self::build_request(params)?;

        let spec = decision_provider_from_config(None)
            .ok_or_else(|| "no decision provider is configured".to_string())?;

        let response = spec
            .provider
            .create_decision(&request)
            .await
            .map_err(|error| format!("the decision provider failed: {error}"))?;

        Ok(render(&echo, spec.name, &response))
    }

    /// Orders the agent's own candidate directions and returns a move off its
    /// stated prior, or nothing at all. The numbers stay in here.
    async fn steer(&self, session_id: &str, params: &SteerParams) -> Result<String, String> {
        let question = params.question.trim();
        if question.is_empty() {
            return Err("the question is empty".to_string());
        }
        let prior = params.prior.trim();
        let labels: Vec<String> = params
            .candidates
            .iter()
            .map(|candidate| candidate.label.trim().to_string())
            .collect();
        if labels.len() < 2 {
            return Err("give at least two candidate directions".to_string());
        }
        if labels.iter().any(|label| label.is_empty()) {
            return Err("candidate labels cannot be empty".to_string());
        }
        if labels.iter().collect::<HashSet<_>>().len() != labels.len() {
            return Err("candidate labels must be distinct".to_string());
        }
        if !labels.iter().any(|label| label == prior) {
            return Err(format!("prior {prior:?} is not one of the candidates"));
        }

        let spec = decision_provider_from_config(None)
            .ok_or_else(|| "no decision provider is configured".to_string())?;

        let request = DecisionRequest {
            model: spec.model.clone(),
            state: json!({ "background": params.context.clone().unwrap_or_default() }),
            questions: HashMap::from([(
                "answer".to_string(),
                DecisionQuestion::Choice {
                    instructions: question.to_string(),
                    criteria: params
                        .candidates
                        .iter()
                        .map(|candidate| {
                            (
                                candidate.label.trim().to_string(),
                                candidate.description.clone(),
                            )
                        })
                        .collect(),
                },
            )]),
        };

        let started = Instant::now();
        let response = spec
            .provider
            .create_decision(&request)
            .await
            .map_err(|error| format!("the decision provider failed: {error}"))?;
        let latency_ms = started.elapsed().as_millis() as i64;

        let Some(DecisionAnswer::Choice {
            confidence,
            probabilities,
            ..
        }) = response.answers.get("answer")
        else {
            return Err("the decision provider did not return a choice".to_string());
        };

        let ordered = ranked(probabilities);
        let steer = steer_from(&ordered, *confidence, prior)?;

        let record = JevDecisionRecord {
            session_id: session_id.to_string(),
            request_id: format!(
                "steer-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|elapsed| elapsed.as_millis())
                    .unwrap_or(0)
            ),
            tool_name: STEER_TOOL_NAME.to_string(),
            arguments: truncate_json(json!({
                "question": question,
                "candidates": labels,
                "prior": prior,
                "distribution": probabilities,
            })),
            read_only: None,
            probability: ordered
                .first()
                .map(|(_, probability)| *probability)
                .unwrap_or(0.0),
            confidence: *confidence,
            model: response.model.clone(),
            latency_ms,
            input_tokens: response.usage.input_tokens.map(|tokens| tokens as i64),
            cost: response.usage.cost,
            judge_read_only: None,
            outcome: match &steer {
                Steer::Move(_) => "steered".to_string(),
                Steer::NoSteer(reason) => reason.outcome().to_string(),
            },
        };
        if let Err(error) = self
            .context
            .session_manager
            .record_jev_decision(&record)
            .await
        {
            tracing::warn!("could not record the steer: {error}");
        }

        Ok(match steer {
            Steer::Move(chosen) => render_steer(prior, &chosen),
            Steer::NoSteer(_) => NO_STEER.to_string(),
        })
    }
}

fn render(echo: &str, provider: &str, response: &DecisionResponse) -> String {
    let mut out = String::new();
    out.push_str(echo);
    out.push_str(&format!("\nanswered by: {} ({provider})\n", response.model));

    match response.answers.get("answer") {
        Some(DecisionAnswer::Noul { noul }) => {
            out.push_str(&format!(
                "\nprobability true: {noul:.3}\n(no confidence is available for a noul answer)"
            ));
        }
        Some(DecisionAnswer::Choice {
            choice,
            confidence,
            probabilities,
        }) => {
            out.push_str(&format!("\nchoice: {choice}\nconfidence: {confidence:.3}"));
            out.push_str(&distribution(probabilities));
            out.push_str(&confidence_note(*confidence));
        }
        Some(DecisionAnswer::Score {
            score,
            confidence,
            probabilities,
            ..
        }) => {
            out.push_str(&format!("\nscore: {score:.3}\nconfidence: {confidence:.3}"));
            out.push_str(&distribution(probabilities));
            out.push_str(&confidence_note(*confidence));
        }
        None => {
            out.push_str("\nthe provider returned no answer for this question");
            return out;
        }
    }

    out.push_str(
        "\n\nadvisory: relay this with its confidence, and say it is a decision model's answer.",
    );
    out
}

const NO_STEER: &str = "No steer: there is no direction to offer. Continue on your own judgement.";

const MAX_LOGGED_ARGUMENTS: usize = 4096;

fn render_steer(prior: &str, chosen: &str) -> String {
    format!(
        "You proposed: {prior}\nConsider instead: {chosen}\n\n\
         This is one decision model's ordering of options you wrote. It cannot explain itself, \
         it is not evidence, and you may override it.\n\
         Before acting on it, say what you would expect to see if this direction is wrong."
    )
}

fn truncate_json(value: serde_json::Value) -> String {
    let text = value.to_string();
    if text.len() <= MAX_LOGGED_ARGUMENTS {
        return text;
    }
    let mut end = MAX_LOGGED_ARGUMENTS;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", text.get(..end).unwrap_or(&text))
}

fn distribution(probabilities: &HashMap<String, f64>) -> String {
    let mut entries: Vec<_> = probabilities.iter().collect();
    entries.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
    entries
        .into_iter()
        .map(|(label, probability)| format!("\n  {probability:.3}  {label}"))
        .collect()
}

fn confidence_note(confidence: f64) -> String {
    if confidence < MIN_USEFUL_CONFIDENCE {
        format!(
            "\n\nconfidence is below {MIN_USEFUL_CONFIDENCE:.1}: the question is probably not well \
             posed. Sharpen or split it and ask again rather than acting on this."
        )
    } else {
        String::new()
    }
}

#[async_trait]
impl McpClientTrait for JevClient {
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
            ASK_TOOL_NAME => match arguments {
                Some(arguments) => {
                    match serde_json::from_value::<AskJevParams>(serde_json::Value::Object(
                        arguments,
                    )) {
                        Ok(params) => self.ask(&params).await,
                        Err(error) => Err(format!("invalid arguments: {error}")),
                    }
                }
                None => Err("no arguments given".to_string()),
            },
            STEER_TOOL_NAME => match arguments {
                Some(arguments) => {
                    match serde_json::from_value::<SteerParams>(serde_json::Value::Object(
                        arguments,
                    )) {
                        Ok(params) => self.steer(&ctx.session_id, &params).await,
                        Err(error) => Err(format!("invalid arguments: {error}")),
                    }
                }
                None => Err("no arguments given".to_string()),
            },
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

    fn params(kind: AnswerKind) -> AskJevParams {
        AskJevParams {
            question: "Should we ship it?".to_string(),
            kind,
            true_meaning: Some("ship now".to_string()),
            false_meaning: Some("wait".to_string()),
            options: None,
            levels: None,
            context: Some("we have no users yet".to_string()),
        }
    }

    fn choice_params() -> AskJevParams {
        AskJevParams {
            options: Some(vec![
                ChoiceOption {
                    label: "read_only".to_string(),
                    meaning: "changes nothing".to_string(),
                },
                ChoiceOption {
                    label: "mutating".to_string(),
                    meaning: "changes something".to_string(),
                },
            ]),
            ..params(AnswerKind::Choice)
        }
    }

    fn response(answer: DecisionAnswer) -> DecisionResponse {
        DecisionResponse {
            model: "typesafe/jev-1.13".to_string(),
            answers: HashMap::from([("answer".to_string(), answer)]),
            usage: goose_providers::decision::DecisionUsage {
                input_tokens: None,
                output_tokens: None,
                cost: None,
            },
            id: None,
            provider: None,
        }
    }

    #[test]
    fn the_tool_description_carries_the_caveats() {
        let description = JevClient::get_tools()[0]
            .description
            .as_ref()
            .expect("the tool has a description")
            .to_string();
        for caveat in [
            "cannot explain itself",
            "user's own words",
            "the wording is the answer",
            "advisory",
            "not calibrated",
            "justify a decision you have already made",
            "no confidence",
        ] {
            assert!(
                description.contains(caveat),
                "the tool description should mention {caveat:?}"
            );
        }
    }

    #[test]
    fn a_noul_question_keeps_the_two_meanings() {
        let (request, echo) = JevClient::build_request(&params(AnswerKind::Noul)).unwrap();
        assert!(echo.contains("Should we ship it?"));
        assert!(matches!(
            request.questions.get("answer"),
            Some(DecisionQuestion::Noul {
                criteria: Some(_),
                ..
            })
        ));
        assert_eq!(request.state["background"], "we have no users yet");
    }

    #[test]
    fn a_noul_without_both_meanings_is_rejected() {
        let mut broken = params(AnswerKind::Noul);
        broken.false_meaning = None;
        assert!(JevClient::build_request(&broken).is_err());
    }

    #[test]
    fn a_choice_needs_two_options() {
        let mut broken = choice_params();
        broken.options = Some(vec![ChoiceOption {
            label: "only".to_string(),
            meaning: "alone".to_string(),
        }]);
        assert!(JevClient::build_request(&broken).is_err());
        assert!(JevClient::build_request(&choice_params()).is_ok());
    }

    #[test]
    fn a_score_needs_a_rubric() {
        let mut broken = params(AnswerKind::Score);
        broken.levels = Some(vec![]);
        assert!(JevClient::build_request(&broken).is_err());
        broken.levels = Some(vec!["low".to_string(), "high".to_string()]);
        assert!(JevClient::build_request(&broken).is_ok());
    }

    #[test]
    fn an_empty_question_is_rejected() {
        let mut broken = params(AnswerKind::Noul);
        broken.question = "   ".to_string();
        assert!(JevClient::build_request(&broken).is_err());
    }

    #[test]
    fn the_rendered_answer_echoes_the_question_and_the_confidence() {
        let echoed = render(
            "question sent: Should we ship it?\nanswer shape: choice",
            "openrouter",
            &response(DecisionAnswer::Choice {
                choice: "read_only".to_string(),
                confidence: 0.91,
                probabilities: HashMap::from([
                    ("read_only".to_string(), 0.91),
                    ("mutating".to_string(), 0.09),
                ]),
            }),
        );
        assert!(echoed.contains("question sent: Should we ship it?"));
        assert!(echoed.contains("choice: read_only"));
        assert!(echoed.contains("confidence: 0.910"));
        assert!(echoed.contains("0.910  read_only"));
        assert!(echoed.contains("advisory"));
        assert!(!echoed.contains("below 0.5"));
    }

    #[test]
    fn a_low_confidence_answer_says_the_question_was_poorly_posed() {
        let echoed = render(
            "echo",
            "openrouter",
            &response(DecisionAnswer::Choice {
                choice: "read_only".to_string(),
                confidence: 0.22,
                probabilities: HashMap::new(),
            }),
        );
        assert!(echoed.contains("not well posed"));
    }

    #[test]
    fn a_noul_answer_says_it_has_no_confidence() {
        let echoed = render(
            "echo",
            "openrouter",
            &response(DecisionAnswer::Noul { noul: 0.77 }),
        );
        assert!(echoed.contains("probability true: 0.770"));
        assert!(echoed.contains("no confidence is available"));
    }

    fn ordered(pairs: &[(&str, f64)]) -> Vec<(String, f64)> {
        pairs
            .iter()
            .map(|(label, probability)| (label.to_string(), *probability))
            .collect()
    }

    #[test]
    fn a_clear_leader_that_is_not_the_prior_becomes_a_move() {
        let ranked = ordered(&[("analyse-logs", 0.80), ("rewrite-parser", 0.12)]);
        assert_eq!(
            steer_from(&ranked, 0.90, "rewrite-parser").unwrap(),
            Steer::Move("analyse-logs".to_string())
        );
    }

    #[test]
    fn a_clear_leader_that_is_the_prior_is_not_a_steer() {
        let ranked = ordered(&[("analyse-logs", 0.80), ("rewrite-parser", 0.12)]);
        assert_eq!(
            steer_from(&ranked, 0.90, "analyse-logs").unwrap(),
            Steer::NoSteer(NoSteerReason::AgreedWithPrior)
        );
    }

    #[test]
    fn a_narrow_margin_is_not_a_steer() {
        let ranked = ordered(&[("analyse-logs", 0.42), ("rewrite-parser", 0.38)]);
        assert_eq!(
            steer_from(&ranked, 0.95, "rewrite-parser").unwrap(),
            Steer::NoSteer(NoSteerReason::NotDistinguishable)
        );
    }

    #[test]
    fn low_confidence_is_not_a_steer_however_wide_the_margin() {
        let ranked = ordered(&[("analyse-logs", 0.95), ("rewrite-parser", 0.02)]);
        assert_eq!(
            steer_from(&ranked, 0.30, "rewrite-parser").unwrap(),
            Steer::NoSteer(NoSteerReason::NotDistinguishable)
        );
    }

    #[test]
    fn the_confidence_bound_is_inclusive() {
        let ranked = ordered(&[("a", 0.75), ("b", 0.10)]);
        assert_eq!(
            steer_from(&ranked, MIN_USEFUL_CONFIDENCE, "b").unwrap(),
            Steer::Move("a".to_string())
        );
        assert_eq!(
            steer_from(&ranked, MIN_USEFUL_CONFIDENCE - 0.01, "b").unwrap(),
            Steer::NoSteer(NoSteerReason::NotDistinguishable)
        );
    }

    #[test]
    fn a_margin_that_only_looks_like_the_boundary_is_not_one() {
        // 0.60 - 0.40 is 0.19999999999999996, so a pair that "is" exactly the
        // margin falls just under it. The margin is a policy knob, not an exact
        // threshold, so this is the right outcome — but it is worth knowing.
        let ranked = ordered(&[("a", 0.60), ("b", 0.40)]);
        assert!(ranked[0].1 - ranked[1].1 < MIN_STEER_MARGIN);
        assert_eq!(
            steer_from(&ranked, 0.95, "b").unwrap(),
            Steer::NoSteer(NoSteerReason::NotDistinguishable)
        );
    }

    #[test]
    fn an_empty_distribution_is_an_error() {
        assert!(steer_from(&[], 0.9, "a").is_err());
    }

    #[test]
    fn a_steer_carries_no_number() {
        let text = render_steer("rewrite-parser", "analyse-logs");
        assert!(text.contains("rewrite-parser"));
        assert!(text.contains("analyse-logs"));
        assert!(
            !text.chars().any(|character| character.is_ascii_digit()),
            "a steer must not carry a number: {text}"
        );
        assert!(text.contains("not evidence"));
        assert!(text.contains("what you would expect to see"));
    }

    #[test]
    fn the_no_steer_reply_does_not_confirm_the_prior() {
        assert!(!NO_STEER.contains("confirm"));
        assert!(NO_STEER.contains("Continue on your own judgement"));
    }

    #[test]
    fn the_steer_tool_description_carries_the_caveats() {
        let description = JevClient::get_tools()[1]
            .description
            .as_ref()
            .expect("the tool has a description")
            .to_string();
        for caveat in [
            "never a number",
            "nothing in it to treat as evidence",
            "cannot explain itself",
            "may override",
            "not a confirmation",
            "never to decide",
        ] {
            assert!(
                description.contains(caveat),
                "the steer description should mention {caveat:?}"
            );
        }
    }

    #[test]
    fn the_distribution_is_ordered_by_probability() {
        let ranked = ranked(&HashMap::from([
            ("low".to_string(), 0.1),
            ("high".to_string(), 0.7),
            ("mid".to_string(), 0.2),
        ]));
        assert_eq!(ranked[0].0, "high");
        assert_eq!(ranked[1].0, "mid");
        assert_eq!(ranked[2].0, "low");
    }
}
