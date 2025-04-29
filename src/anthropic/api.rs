use crate::anthropic::tools::editor::Editor;
use crate::core::{
    Error,
    llm::{
        self, AssistantContent, Content as LlmContent, Function, Hyperparams,
        Message as LlmMessage, Model, Usage as LlmUsage, UserContent,
    },
    tool::ProviderTool,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{AnthropicModel, Claude};

#[derive(Clone)]
pub struct ClaudeModel {
    client: Client,
    api_key: String,
    model: Claude,
    system_prompt: Option<String>,
    hyperparams: Hyperparams,
}

impl ClaudeModel {
    pub fn new(
        client: Client,
        api_key: String,
        model: Claude,
        system_prompt: Option<String>,
        hyperparams: Hyperparams,
    ) -> Self {
        Self {
            client,
            api_key,
            model,
            system_prompt,
            hyperparams,
        }
    }
}

impl Model for ClaudeModel {
    async fn call(
        &self,
        messages: impl AsRef<[LlmMessage]>,
        functions: impl AsRef<[Function]>,
    ) -> Result<llm::Completion, Error> {
        let anthropic_messages = messages
            .as_ref()
            .iter()
            .map(map_llm_message_to_anthropic)
            .collect::<Vec<_>>();

        let anthropic_tools = functions
            .as_ref()
            .iter()
            .map(|f| match f {
                Function::Local {
                    name,
                    description,
                    input_schema,
                } => Tool {
                    r#type: None,
                    name: name.clone(),
                    description: Some(description.clone()),
                    input_schema: Some(input_schema.clone()),
                },
                Function::Provider { id, name } => Tool {
                    r#type: Some(id.clone()),
                    name: name.clone(),
                    description: None,
                    input_schema: None,
                },
            })
            .collect::<Vec<_>>();

        let payload = NewMessages {
            model: self.model.to_string(),
            max_tokens: self.hyperparams.max_tokens,
            temperature: Some(self.hyperparams.temperature),
            system: self.system_prompt.clone(),
            messages: anthropic_messages,
            tools: anthropic_tools,
        };

        let body = serde_json::to_string(&payload)?;
        let req = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .body(body)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json");
        let resp = req.send().await?.text().await?;
        let completion: Completion = serde_json::from_str(&resp)?;

        match completion {
            Completion::Message {
                content,
                id: _,
                model: _,
                stop_reason: _,
                stop_sequence: _,
                usage,
            } => {
                let llm_content = content
                    .into_iter()
                    .map(map_anthropic_content_to_llm)
                    .collect();
                Ok(llm::Completion {
                    usage: LlmUsage {
                        input_tokens: usage.input_tokens,
                        output_tokens: usage.output_tokens,
                    },
                    content: llm_content,
                })
            }
            Completion::Error { error } => Err(Error::Provider(error.message)),
        }
    }
}

impl AnthropicModel for ClaudeModel {
    fn editor<'a, 'b>(&'a self) -> impl ProviderTool + 'b {
        Editor::new(self.model)
    }
}

fn map_llm_message_to_anthropic(msg: &LlmMessage) -> Message {
    match msg {
        LlmMessage::User(content) => Message::User {
            content: content
                .iter()
                .map(map_llm_user_content_to_anthropic)
                .collect(),
        },
        LlmMessage::Assistant(content) => Message::Assistant {
            content: content
                .iter()
                .map(map_llm_assistant_content_to_anthropic)
                .collect(),
        },
    }
}

fn map_llm_user_content_to_anthropic(content: &UserContent) -> Content {
    match content {
        UserContent::Input(LlmContent::Text(text)) => Content::Text { text: text.clone() },
        UserContent::FunctionResult { id, result } => Content::ToolResult {
            tool_use_id: id.clone(),
            is_error: result.is_err(),
            content: match result {
                Ok(texts) => texts
                    .iter()
                    .map(|t| {
                        let inner_text = match t {
                            LlmContent::Text(s) => s.clone(),
                        };
                        Box::new(Some(Content::Text { text: inner_text }))
                    })
                    .collect(),
                Err(LlmContent::Text(text)) => {
                    vec![Box::new(Some(Content::Text { text: text.clone() }))]
                }
            },
        },
    }
}

fn map_llm_assistant_content_to_anthropic(content: &AssistantContent) -> Content {
    match content {
        AssistantContent::Output(LlmContent::Text(text)) => Content::Text { text: text.clone() },
        AssistantContent::FunctionCall { id, name, input } => Content::ToolUse {
            id: id.clone(),
            name: name.clone(),
            input: input.clone(),
        },
    }
}

fn map_anthropic_content_to_llm(content: Content) -> AssistantContent {
    match content {
        Content::Text { text } => AssistantContent::Output(LlmContent::Text(text)),
        Content::ToolUse { id, name, input } => AssistantContent::FunctionCall { id, name, input },
        Content::ToolResult {
            tool_use_id,
            is_error,
            content,
        } => AssistantContent::Output(LlmContent::Text(format!(
            "[ToolResult for {}: is_error={}, content={:?}]",
            tool_use_id, is_error, content
        ))),
    }
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub struct NewMessages {
    pub model: String,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    pub messages: Vec<Message>,
    pub tools: Vec<Tool>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "role")]
pub enum Message {
    User { content: Vec<Content> },
    Assistant { content: Vec<Content> },
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum Completion {
    Message {
        content: Vec<Content>,
        id: String,
        model: String,
        stop_reason: String,
        stop_sequence: Option<String>,
        usage: Usage,
    },
    Error {
        error: ErrorInfo,
    },
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub struct ErrorInfo {
    message: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
pub enum Content {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        is_error: bool,
        content: Vec<Box<Option<Content>>>,
    },
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub struct Usage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Serialize)]
pub struct Tool {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<serde_json::Value>,
}
