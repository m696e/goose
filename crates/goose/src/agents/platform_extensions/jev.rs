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
use std::collections::HashMap;
use tokio_util::sync::CancellationToken;

use crate::agents::extension::PlatformExtensionContext;
use crate::agents::mcp_client::{Error, McpClientTrait};
use crate::agents::tool_execution::ToolCallContext;
use crate::providers::decision_provider::decision_provider_from_config;

pub static EXTENSION_NAME: &str = "jev";
pub const ASK_TOOL_NAME: &str = "ask_jev";

/// Below this, the question is not well enough posed for the answer to mean much.
pub const MIN_USEFUL_CONFIDENCE: f64 = 0.5;

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

pub struct JevClient {
    info: InitializeResult,
}

impl JevClient {
    pub fn new(_context: PlatformExtensionContext) -> Result<Self> {
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
            "#}
                .to_string(),
            );

        Ok(Self { info })
    }

    fn get_tools() -> Vec<Tool> {
        let schema = schema_for!(AskJevParams);
        let schema_value =
            serde_json::to_value(schema).expect("Failed to serialize AskJevParams schema");

        vec![Tool::new(
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
        ))]
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
        _ctx: &ToolCallContext,
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
}
