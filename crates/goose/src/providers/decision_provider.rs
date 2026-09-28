//! Resolves a [`DecisionProvider`] from goose configuration.
//!
//! Decisions are a separate axis from chat completions: a decision provider
//! answers typed questions (`noul`/`choice`/`score`) and never generates prose.
//! Which upstream serves them is chosen by `GOOSE_DECISION_PROVIDER`, falling
//! back to whichever decision-capable API key is configured.

use std::sync::Arc;

use goose_providers::api_client::{ApiClient, AuthMethod, TlsConfig};
use goose_providers::decision::DecisionProvider;
use goose_providers::openrouter::{
    OpenRouterProvider, OPENROUTER_DECISION_DEFAULT_MODEL, OPENROUTER_PROVIDER_NAME,
};
use goose_providers::typesafe::{TypeSafeProvider, TYPESAFE_DEFAULT_HOST, TYPESAFE_DEFAULT_MODEL};
use tracing::warn;

use crate::config::Config;

pub const DECISION_PROVIDER_CONFIG_KEY: &str = "GOOSE_DECISION_PROVIDER";
pub const DECISION_MODEL_CONFIG_KEY: &str = "GOOSE_DECISION_MODEL";

const OPENROUTER_API_KEY: &str = "OPENROUTER_API_KEY";
const OPENROUTER_HOST: &str = "OPENROUTER_HOST";
const OPENROUTER_DEFAULT_HOST: &str = "https://openrouter.ai";

const TYPESAFE_API_KEY: &str = "TYPESAFE_API_KEY";
const TYPESAFE_HOST: &str = "TYPESAFE_HOST";

/// A usable decision provider and the model id to ask it for.
pub struct DecisionProviderSpec {
    pub name: &'static str,
    pub model: String,
    pub provider: Arc<dyn DecisionProvider>,
}

impl std::fmt::Debug for DecisionProviderSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecisionProviderSpec")
            .field("name", &self.name)
            .field("model", &self.model)
            .finish()
    }
}

/// Resolves the configured decision provider, or `None` when none is usable.
///
/// An explicit `GOOSE_DECISION_PROVIDER` (`openrouter` or `typesafe`) wins.
/// Otherwise the first available key decides: OpenRouter, then TypeSafe.
pub fn decision_provider_from_config(
    tls_config: Option<TlsConfig>,
) -> Option<DecisionProviderSpec> {
    let config = Config::global();
    let model_override = config.get_param::<String>(DECISION_MODEL_CONFIG_KEY).ok();
    let explicit = config
        .get_param::<String>(DECISION_PROVIDER_CONFIG_KEY)
        .ok()
        .map(|value| value.trim().to_lowercase());

    match explicit.as_deref() {
        Some("openrouter") => openrouter(config, tls_config, model_override),
        Some("typesafe") => typesafe(config, tls_config, model_override),
        Some(other) => {
            warn!("{DECISION_PROVIDER_CONFIG_KEY} is {other:?}, expected openrouter or typesafe");
            None
        }
        None => openrouter(config, tls_config.clone(), model_override.clone())
            .or_else(|| typesafe(config, tls_config, model_override)),
    }
}

fn openrouter(
    config: &Config,
    tls_config: Option<TlsConfig>,
    model_override: Option<String>,
) -> Option<DecisionProviderSpec> {
    let api_key: String = config.get_secret(OPENROUTER_API_KEY).ok()?;
    let host: String = config
        .get_param(OPENROUTER_HOST)
        .unwrap_or_else(|_| OPENROUTER_DEFAULT_HOST.to_string());

    let api_client = ApiClient::new_with_tls(host, AuthMethod::BearerToken(api_key), tls_config)
        .ok()?
        .with_header("HTTP-Referer", "https://goose-docs.ai")
        .ok()?
        .with_header("X-Title", "goose")
        .ok()?
        .with_header("X-OpenRouter-Categories", "cli-agent,productivity")
        .ok()?;

    Some(DecisionProviderSpec {
        name: OPENROUTER_PROVIDER_NAME,
        model: model_override.unwrap_or_else(|| OPENROUTER_DECISION_DEFAULT_MODEL.to_string()),
        provider: Arc::new(OpenRouterProvider::new(api_client, None, None)),
    })
}

fn typesafe(
    config: &Config,
    tls_config: Option<TlsConfig>,
    model_override: Option<String>,
) -> Option<DecisionProviderSpec> {
    let api_key: String = config.get_secret(TYPESAFE_API_KEY).ok()?;
    let host: String = config
        .get_param(TYPESAFE_HOST)
        .unwrap_or_else(|_| TYPESAFE_DEFAULT_HOST.to_string());

    let api_client =
        ApiClient::new_with_tls(host, AuthMethod::BearerToken(api_key), tls_config).ok()?;

    Some(DecisionProviderSpec {
        name: "typesafe",
        model: model_override.unwrap_or_else(|| TYPESAFE_DEFAULT_MODEL.to_string()),
        provider: Arc::new(TypeSafeProvider::new(api_client)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_openrouter_decision_model_is_jev() {
        assert_eq!(OPENROUTER_DECISION_DEFAULT_MODEL, "typesafe/jev-1.13");
    }
}
