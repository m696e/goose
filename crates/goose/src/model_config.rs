use crate::config::{Config, ConfigError};
use crate::conversation::message::{LlmStage, Message};
use crate::providers::base::Provider;
use anyhow::{anyhow, Result};
use futures::StreamExt;
use goose_providers::base::collect_stream;
use goose_providers::conversation::token_usage::ProviderUsage;
use goose_providers::errors::ProviderError;
use goose_providers::model::ModelConfig;
use goose_providers::thinking::ThinkingEffort;
use rmcp::model::Tool;
use serde_json::Value;
use std::collections::HashMap;

pub fn model_config_from_user_config(
    provider_name: &str,
    model_name: impl AsRef<str>,
) -> Result<ModelConfig> {
    let model = base_model_config_from_user_config(provider_name, model_name.as_ref())?;
    materialize_model_config(provider_name, model)
}

pub fn model_config_from_user_config_with_session_settings(
    provider_name: &str,
    model_name: impl AsRef<str>,
    previous: Option<&ModelConfig>,
    request_params: Option<HashMap<String, Value>>,
    _context_limit: Option<usize>,
) -> Result<ModelConfig> {
    let config = Config::global();
    let model = base_model_config_from_user_config(provider_name, model_name.as_ref())?;
    let model = materialize_model_config_inner(model, provider_name, false)?
        .with_inherited_session_settings_from(previous, request_params)
        .with_default_thinking_effort(config.get_goose_thinking_effort());

    Ok(apply_canonical_limits(provider_name, model))
}

pub fn materialize_model_config(provider_name: &str, model: ModelConfig) -> Result<ModelConfig> {
    let model = materialize_model_config_inner(model, provider_name, true)?;
    Ok(apply_canonical_limits(provider_name, model))
}

fn apply_canonical_limits(provider_name: &str, model: ModelConfig) -> ModelConfig {
    if provider_name == goose_providers::azure_foundry::AZURE_FOUNDRY_PROVIDER_NAME {
        return model;
    }
    let model = with_declarative_vision(
        provider_name,
        model
            .with_canonical_limits(provider_name)
            .with_canonical_vision_support(provider_name),
    );
    with_declarative_image_dimension(provider_name, model)
}

/// Declarative and custom providers may serve models the canonical registry
/// does not know, which leaves `supports_vision` unset. Formatters read unset
/// as "no vision" and drop image content, so fall back to the provider's own
/// declared model list when the model states a value there.
fn with_declarative_vision(provider_name: &str, mut model: ModelConfig) -> ModelConfig {
    if model.supports_vision.is_none() {
        if let Some(supports_vision) = declared_supports_vision(provider_name, &model.model_name) {
            model = model.with_vision_support(supports_vision);
        }
    }
    model
}

fn declared_supports_vision(provider_name: &str, model_name: &str) -> Option<bool> {
    crate::config::declarative_providers::load_provider(provider_name)
        .ok()?
        .config
        .models
        .iter()
        .find(|model| model.name == model_name)?
        .supports_vision
}

/// Re-derive the longest image side the endpoint accepts from the provider's
/// current declaration, discarding any value stored on the model config. The
/// limit is configuration state, not session state: a resumed session must
/// reflect what the provider declares now, including a declaration it did not
/// have when the session was saved.
pub fn with_rederived_image_dimension(provider_name: &str, mut model: ModelConfig) -> ModelConfig {
    model.max_image_dimension = None;
    with_declarative_image_dimension(provider_name, model)
}

/// The longest image side the endpoint accepts, as the provider declares it.
/// A model entry that states a value wins over the provider's own, since the
/// limit belongs to the endpoint serving that model. Unstated stays `None`:
/// the numbers differ between endpoints, and a guess would either refuse an
/// image the provider takes or claim a limit goose has not been told.
fn with_declarative_image_dimension(provider_name: &str, mut model: ModelConfig) -> ModelConfig {
    if model.max_image_dimension.is_none() {
        if let Some(dimension) = declared_max_image_dimension(provider_name, &model.model_name) {
            model = model.with_max_image_dimension(dimension);
        }
    }
    model
}

fn declared_max_image_dimension(provider_name: &str, model_name: &str) -> Option<u32> {
    let provider = crate::config::declarative_providers::load_provider(provider_name)
        .ok()?
        .config;

    provider
        .models
        .iter()
        .find(|model| model.name == model_name)
        .and_then(|model| model.max_image_dimension)
        .or(provider.max_image_dimension)
}

fn materialize_model_config_inner(
    mut model: ModelConfig,
    provider_name: &str,
    include_default_thinking_effort: bool,
) -> Result<ModelConfig> {
    let config = Config::global();

    if model.temperature.is_none() {
        model = model.with_temperature(get_goose_temperature(config)?);
    }

    if model.toolshim && model.toolshim_model.is_none() {
        model = model.with_toolshim_model(get_goose_toolshim_model(config)?);
    }

    model = model.with_default_max_tokens(config.get_goose_max_tokens()?);

    if include_default_thinking_effort {
        model = model.with_default_thinking_effort(config.get_goose_thinking_effort());
    }

    if model.cache_ttl().is_none() {
        if let Some(ttl) = get_goose_cache_ttl(config)? {
            model = model.with_cache_ttl(&ttl);
        }
    }

    if provider_name == goose_providers::openai::OPEN_AI_PROVIDER_NAME {
        model = apply_openai_request_params(model);
    }

    Ok(model)
}

fn one_shot_model_config(model_config: ModelConfig) -> ModelConfig {
    model_config
        .with_thinking_effort(ThinkingEffort::Off)
        .with_prompt_cache_disabled()
}

/// Run a completion for a one-shot auxiliary task on the main session model.
/// Thinking is disabled and prompt-cache writes are skipped because this prompt
/// will not recur.
pub async fn complete_one_shot(
    provider: &dyn Provider,
    model_config: &ModelConfig,
    session_id: &str,
    system: &str,
    messages: &[Message],
    tools: &[Tool],
) -> Result<(Message, ProviderUsage), ProviderError> {
    let one_shot_model_config = one_shot_model_config(model_config.clone());

    crate::session_context::with_session_id(
        Some(session_id.to_string()),
        provider.complete(&one_shot_model_config, system, messages, tools),
    )
    .await
}

/// Run a one-shot completion that reports the request's stage, for prompts
/// whose output is collected rather than streamed to the user. Stage reporting
/// is what tells a client that generation has started; without it the request
/// looks stalled for its whole duration.
pub async fn complete_one_shot_with_stage(
    provider: &dyn Provider,
    model_config: &ModelConfig,
    session_id: &str,
    system: &str,
    messages: &[Message],
    tools: &[Tool],
) -> Result<(Message, ProviderUsage), ProviderError> {
    let one_shot_model_config = one_shot_model_config(model_config.clone());

    crate::session_context::with_session_id(Some(session_id.to_string()), async {
        crate::session_context::report_stage(LlmStage::Prefilling);

        let stream = provider
            .stream(&one_shot_model_config, system, messages, tools)
            .await?;

        let mut generation_reported = false;
        let stream = stream.inspect(move |item| {
            if !generation_reported {
                if let Ok((Some(_), _)) = item {
                    generation_reported = true;
                    crate::session_context::report_stage(LlmStage::RewritingContext);
                }
            }
        });

        collect_stream(Box::pin(stream)).await
    })
    .await
}

fn apply_openai_request_params(mut model: ModelConfig) -> ModelConfig {
    let config = Config::global();
    if let Some(store) = config.get_openai_store() {
        model = model.with_merged_request_params(HashMap::from([(
            "store".to_string(),
            serde_json::json!(store),
        )]));
    }
    model
}

fn base_model_config_from_user_config(
    provider_name: &str,
    model_name: &str,
) -> Result<ModelConfig> {
    let config = Config::global();
    let mut model = ModelConfig {
        model_name: model_name.to_string(),
        context_limit: None,
        temperature: get_goose_temperature(config)?,
        max_tokens: None,
        toolshim: get_goose_toolshim(config)?.unwrap_or(false),
        toolshim_model: get_goose_toolshim_model(config)?,
        request_params: None,
        reasoning: None,
        supports_vision: None,
        max_image_dimension: None,
        request_headers: None,
    };
    if provider_name != goose_providers::azure_foundry::AZURE_FOUNDRY_PROVIDER_NAME {
        model.normalize_effort_suffix();
    }
    Ok(model)
}

/// Re-derive the prompt-cache TTL from the current configuration, discarding
/// any value stored on the model config. The TTL is configuration state, not
/// session state: a resumed session must reflect the user's current opt-in,
/// not a value persisted by an earlier (possibly clamped) run.
pub fn with_rederived_cache_ttl(model: ModelConfig) -> Result<ModelConfig> {
    let mut model = model.without_cache_ttl();
    if let Some(ttl) = get_goose_cache_ttl(Config::global())? {
        model = model.with_cache_ttl(&ttl);
    }
    Ok(model)
}

fn get_goose_cache_ttl(config: &Config) -> Result<Option<String>> {
    match config.get_param::<String>("GOOSE_CACHE_TTL") {
        Ok(ttl) => {
            let ttl = ttl.trim().to_lowercase();
            match ttl.as_str() {
                "5m" | "1h" => Ok(Some(ttl)),
                other => Err(anyhow!(
                    "GOOSE_CACHE_TTL must be '5m' or '1h', got '{other}'"
                )),
            }
        }
        Err(ConfigError::NotFound(_)) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

fn get_goose_temperature(config: &Config) -> Result<Option<f32>> {
    match config.get_param::<f32>("GOOSE_TEMPERATURE") {
        Ok(temp) if temp < 0.0 => Err(anyhow!(
            "Value for 'GOOSE_TEMPERATURE' is out of valid range: {temp}"
        )),
        Ok(temp) => Ok(Some(temp)),
        Err(ConfigError::NotFound(_)) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

fn get_goose_toolshim(config: &Config) -> Result<Option<bool>> {
    match config.get_param::<serde_yaml::Value>("GOOSE_TOOLSHIM") {
        Ok(value) => parse_yaml_bool_config("GOOSE_TOOLSHIM", value).map(Some),
        Err(ConfigError::NotFound(_)) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Resolve the global toolshim setting, defaulting to false when unset.
pub fn global_toolshim() -> bool {
    get_goose_toolshim(Config::global())
        .ok()
        .flatten()
        .unwrap_or(false)
}

fn get_goose_toolshim_model(config: &Config) -> Result<Option<String>> {
    match config.get_param::<String>("GOOSE_TOOLSHIM_OLLAMA_MODEL") {
        Ok(value) if value.trim().is_empty() => Err(anyhow!(
            "Invalid value for 'GOOSE_TOOLSHIM_OLLAMA_MODEL': '{value}' - cannot be empty if set"
        )),
        Ok(value) => Ok(Some(value)),
        Err(ConfigError::NotFound(_)) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

fn parse_bool_config(key: &str, value: &str) -> Result<bool> {
    match value.to_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(anyhow!(
            "Invalid value for '{key}': '{value}' - must be one of: 1, true, yes, on, 0, false, no, off"
        )),
    }
}

fn parse_yaml_bool_config(key: &str, value: serde_yaml::Value) -> Result<bool> {
    match value {
        serde_yaml::Value::Bool(value) => Ok(value),
        serde_yaml::Value::Number(value) => parse_bool_config(key, &value.to_string()),
        serde_yaml::Value::String(value) => parse_bool_config(key, &value),
        other => {
            Err(anyhow!(
            "Invalid value for '{key}': '{}' - must be one of: 1, true, yes, on, 0, false, no, off",
            serde_yaml::to_string(&other).unwrap_or_else(|_| "<unprintable>".to_string()).trim()
        ))
        }
    }
}

#[cfg(test)]
mod declarative_vision_tests {
    use super::*;
    use crate::config::declarative_providers::{
        create_custom_provider, CreateCustomProviderParams,
    };
    use goose_providers::base::ModelInfo;

    fn create_provider_with_model(model: ModelInfo) -> String {
        create_custom_provider(CreateCustomProviderParams {
            engine: "openai".to_string(),
            display_name: "Vision Probe".to_string(),
            api_url: "https://example.invalid/v1".to_string(),
            api_key: None,
            models: vec![model],
            supports_streaming: Some(true),
            headers: None,
            requires_auth: false,
            catalog_provider_id: None,
            base_path: None,
            toolshim: false,
            preserves_thinking: None,
            auth: None,
        })
        .unwrap()
        .name
    }

    #[test]
    fn fills_vision_from_declarative_model_when_registry_has_no_entry() {
        let temp_dir = tempfile::tempdir().unwrap();
        let temp_root = temp_dir.path().display().to_string();
        let _guard = env_lock::lock_env([("GOOSE_PATH_ROOT", Some(temp_root.as_str()))]);

        let provider = create_provider_with_model(
            ModelInfo::new("unknown-vision-model").with_vision_support(true),
        );

        let config = model_config_from_user_config(&provider, "unknown-vision-model").unwrap();
        assert_eq!(config.supports_vision, Some(true));
    }

    #[test]
    fn declarative_vision_false_is_honored() {
        let temp_dir = tempfile::tempdir().unwrap();
        let temp_root = temp_dir.path().display().to_string();
        let _guard = env_lock::lock_env([("GOOSE_PATH_ROOT", Some(temp_root.as_str()))]);

        let provider = create_provider_with_model(
            ModelInfo::new("unknown-text-model").with_vision_support(false),
        );

        let config = model_config_from_user_config(&provider, "unknown-text-model").unwrap();
        assert_eq!(config.supports_vision, Some(false));
    }

    fn declare_provider_image_dimension(provider: &str, dimension: u32) {
        let path =
            crate::config::declarative_providers::custom_provider_file_path(provider).unwrap();
        let mut json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        json["max_image_dimension"] = serde_json::json!(dimension);
        std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap()).unwrap();
    }

    #[test]
    fn fills_the_image_dimension_a_model_declares() {
        let temp_dir = tempfile::tempdir().unwrap();
        let temp_root = temp_dir.path().display().to_string();
        let _guard = env_lock::lock_env([("GOOSE_PATH_ROOT", Some(temp_root.as_str()))]);

        let provider = create_provider_with_model(
            ModelInfo::new("limited-vision-model")
                .with_vision_support(true)
                .with_max_image_dimension(4096),
        );

        let config = model_config_from_user_config(&provider, "limited-vision-model").unwrap();
        assert_eq!(config.max_image_dimension, Some(4096));
    }

    #[test]
    fn fills_the_image_dimension_a_provider_declares() {
        let temp_dir = tempfile::tempdir().unwrap();
        let temp_root = temp_dir.path().display().to_string();
        let _guard = env_lock::lock_env([("GOOSE_PATH_ROOT", Some(temp_root.as_str()))]);

        let provider = create_provider_with_model(ModelInfo::new("unlimited-vision-model"));
        declare_provider_image_dimension(&provider, 8192);

        let config = model_config_from_user_config(&provider, "unlimited-vision-model").unwrap();
        assert_eq!(config.max_image_dimension, Some(8192));
    }

    #[test]
    fn a_model_image_dimension_wins_over_the_providers() {
        let temp_dir = tempfile::tempdir().unwrap();
        let temp_root = temp_dir.path().display().to_string();
        let _guard = env_lock::lock_env([("GOOSE_PATH_ROOT", Some(temp_root.as_str()))]);

        let provider = create_provider_with_model(
            ModelInfo::new("narrow-vision-model").with_max_image_dimension(4096),
        );
        declare_provider_image_dimension(&provider, 8192);

        let config = model_config_from_user_config(&provider, "narrow-vision-model").unwrap();
        assert_eq!(config.max_image_dimension, Some(4096));
    }

    #[test]
    fn undeclared_image_dimension_stays_none() {
        let temp_dir = tempfile::tempdir().unwrap();
        let temp_root = temp_dir.path().display().to_string();
        let _guard = env_lock::lock_env([("GOOSE_PATH_ROOT", Some(temp_root.as_str()))]);

        let provider = create_provider_with_model(ModelInfo::new("silent-vision-model"));

        let config = model_config_from_user_config(&provider, "silent-vision-model").unwrap();
        assert_eq!(config.max_image_dimension, None);
    }

    #[test]
    fn rederivation_replaces_a_stored_dimension_with_the_declared_one() {
        let temp_dir = tempfile::tempdir().unwrap();
        let temp_root = temp_dir.path().display().to_string();
        let _guard = env_lock::lock_env([("GOOSE_PATH_ROOT", Some(temp_root.as_str()))]);

        let provider = create_provider_with_model(
            ModelInfo::new("restated-vision-model").with_max_image_dimension(4096),
        );
        let stored = ModelConfig::new("restated-vision-model").with_max_image_dimension(9999);

        let rederived = with_rederived_image_dimension(&provider, stored);

        assert_eq!(rederived.max_image_dimension, Some(4096));
    }

    #[test]
    fn rederivation_drops_a_stored_dimension_no_longer_declared() {
        let temp_dir = tempfile::tempdir().unwrap();
        let temp_root = temp_dir.path().display().to_string();
        let _guard = env_lock::lock_env([("GOOSE_PATH_ROOT", Some(temp_root.as_str()))]);

        let provider = create_provider_with_model(ModelInfo::new("undefined-vision-model"));
        let stored = ModelConfig::new("undefined-vision-model").with_max_image_dimension(4096);

        let rederived = with_rederived_image_dimension(&provider, stored);

        assert_eq!(rederived.max_image_dimension, None);
    }

    #[test]
    fn undeclared_vision_stays_none() {
        let temp_dir = tempfile::tempdir().unwrap();
        let temp_root = temp_dir.path().display().to_string();
        let _guard = env_lock::lock_env([("GOOSE_PATH_ROOT", Some(temp_root.as_str()))]);

        let provider =
            create_provider_with_model(ModelInfo::new("unknown-plain-model").with_context_limit(1));

        let config = model_config_from_user_config(&provider, "unknown-plain-model").unwrap();
        assert_eq!(config.supports_vision, None);
    }
}

#[cfg(test)]
mod one_shot_tests {
    use super::*;

    #[test]
    fn thinking_and_prompt_cache_are_disabled() {
        let config = one_shot_model_config(
            ModelConfig::new("claude-haiku-4-5").with_thinking_effort(ThinkingEffort::High),
        );

        assert_eq!(config.thinking_effort(), Some(ThinkingEffort::Off));
        assert!(config.prompt_cache_disabled());
    }
}

#[cfg(test)]
mod cache_ttl_tests {
    use super::*;

    #[test]
    fn env_var_populates_cache_ttl() {
        let _guard = env_lock::lock_env([("GOOSE_CACHE_TTL", Some("1h"))]);
        let model = materialize_model_config_inner(
            ModelConfig::new("claude-sonnet-4-5"),
            "anthropic",
            false,
        )
        .unwrap();
        assert_eq!(model.cache_ttl().as_deref(), Some("1h"));
    }

    #[test]
    fn absent_env_var_leaves_cache_ttl_unset() {
        let _guard = env_lock::lock_env([("GOOSE_CACHE_TTL", None::<&str>)]);
        let model = materialize_model_config_inner(
            ModelConfig::new("claude-sonnet-4-5"),
            "anthropic",
            false,
        )
        .unwrap();
        assert!(model.cache_ttl().is_none());
    }

    #[test]
    fn invalid_env_var_is_rejected() {
        let _guard = env_lock::lock_env([("GOOSE_CACHE_TTL", Some("2h"))]);
        let result = materialize_model_config_inner(
            ModelConfig::new("claude-sonnet-4-5"),
            "anthropic",
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn rederive_replaces_stored_ttl_with_configured_value() {
        let _guard = env_lock::lock_env([("GOOSE_CACHE_TTL", Some("1h"))]);
        let model =
            with_rederived_cache_ttl(ModelConfig::new("claude-sonnet-4-5").with_cache_ttl("5m"))
                .unwrap();
        assert_eq!(model.cache_ttl().as_deref(), Some("1h"));
    }

    #[test]
    fn rederive_drops_stored_ttl_when_config_absent() {
        let _guard = env_lock::lock_env([("GOOSE_CACHE_TTL", None::<&str>)]);
        let model =
            with_rederived_cache_ttl(ModelConfig::new("claude-sonnet-4-5").with_cache_ttl("1h"))
                .unwrap();
        assert!(model.cache_ttl().is_none());
    }

    #[test]
    fn explicit_model_ttl_wins_over_env_var() {
        let _guard = env_lock::lock_env([("GOOSE_CACHE_TTL", Some("1h"))]);
        let model = materialize_model_config_inner(
            ModelConfig::new("claude-sonnet-4-5").with_cache_ttl("5m"),
            "anthropic",
            false,
        )
        .unwrap();
        assert_eq!(model.cache_ttl().as_deref(), Some("5m"));
    }
}

#[cfg(test)]
mod azure_foundry_tests {
    use super::*;

    #[test]
    fn deployment_name_survives_thinking_effort_changes() {
        let config = base_model_config_from_user_config("azure_foundry", "gpt-5-high")
            .unwrap()
            .with_thinking_effort(ThinkingEffort::Off);

        assert_eq!(config.model_name, "gpt-5-high");
        assert_eq!(config.context_limit, None);
        assert_eq!(config.thinking_effort(), Some(ThinkingEffort::Off));
    }

    #[test]
    fn none_suffixed_deployment_name_is_preserved() {
        let config = base_model_config_from_user_config("azure_foundry", "gpt-5-none").unwrap();

        assert_eq!(config.model_name, "gpt-5-none");
        assert_eq!(config.thinking_effort(), None);
    }
}

#[cfg(test)]
mod canonical_vision_tests {
    use super::*;

    #[test]
    fn apply_canonical_limits_resolves_vision_support() {
        let model = apply_canonical_limits("alibaba-token-plan", ModelConfig::new("qwen3.8-flash"));

        assert_eq!(model.supports_vision, Some(true));
    }

    #[test]
    fn apply_canonical_limits_leaves_unknown_providers_untouched() {
        let model = apply_canonical_limits("my-custom-provider", ModelConfig::new("some-model"));

        assert_eq!(model.supports_vision, None);
    }
}
