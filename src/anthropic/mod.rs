mod api;
mod tools;

use reqwest::Client;

use crate::anthropic::api::ClaudeModel;
use crate::core::llm::{Hyperparams, Model, Provider};
use crate::core::tool::ProviderTool;

/// An implementation of the `Provider` trait for Anthropic's models.
#[derive(Clone, Debug)]
pub struct Anthropic {
    client: Client,
    api_key: String,
}

impl Anthropic {
    /// Create a new Anthropic client with the given API key.
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: Client::new(),
        }
    }
}

/// An implementation of the `Provider` trait for Anthropic's models.
/// 
/// Note that this will yield a refined `AnthropicModel` implementation, which adds
/// additional functionality.
impl Provider<Claude> for Anthropic {
    #[allow(refining_impl_trait)]
    async fn obtain(
        &self,
        model: Claude,
        system_prompt: Option<impl AsRef<str>>,
        hyperparams: Hyperparams,
    ) -> impl AnthropicModel {
        ClaudeModel::new(
            self.client.clone(),
            self.api_key.clone(),
            model,
            system_prompt.map(|s| s.as_ref().to_string()),
            hyperparams,
        )
    }
}

/// A trait that adds additional functionality to the `Model` trait for Anthropic's models.
/// 
/// Notably, this trait provides tool implementations provided by Anthropic's API.
pub trait AnthropicModel: Model {
    fn editor<'a, 'b>(&'a self) -> impl ProviderTool + 'b;
}

/// Claude, Anthropic's flagship LLM.
#[derive(Clone, Copy, Debug)]
pub enum Claude {
    /// Claude 3.5 Sonnet.
    ThreeDotFiveSonnet,
    /// Claude 3.7 Sonnet.
    ThreeDotSevenSonnet,
}

impl ToString for Claude {
    fn to_string(&self) -> String {
        match self {
            Claude::ThreeDotFiveSonnet => "claude-3-5-sonnet-20241022".to_string(),
            Claude::ThreeDotSevenSonnet => "claude-3-7-sonnet-20250219".to_string(),
        }
    }
}
