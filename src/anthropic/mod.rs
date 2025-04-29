mod api;
mod tools;

use reqwest::Client;

use crate::anthropic::api::ClaudeModel;
use crate::core::llm::{Hyperparams, Model, Provider};
use crate::core::tool::ProviderTool;

#[derive(Clone, Debug)]
pub struct Anthropic {
    client: Client,
    api_key: String,
}

impl Anthropic {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: Client::new(),
        }
    }
}

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

pub trait AnthropicModel: Model {
    fn editor<'a, 'b>(&'a self) -> impl ProviderTool + 'b;
}

#[derive(Clone, Copy, Debug)]
pub enum Claude {
    ThreeDotFiveSonnet,
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
