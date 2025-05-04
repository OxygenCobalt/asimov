use serde_json::Value;

/// A provider of LLM models.
pub trait Provider<T> {
    /// Obtain a new model from the provider with the provided system prompt and hyperparams.
    async fn obtain(
        &self,
        model: T,
        system_prompt: Option<impl AsRef<str>>,
        hyperparams: Hyperparams,
    ) -> impl Model;
}

/// Hyperparameters for an LLM.
#[derive(Debug, Clone, Copy)]
pub struct Hyperparams {
    /// The maximum number of tokens to generate.
    pub max_tokens: u32,
    /// The temperature to use for the model.
    pub temperature: f64,
}

/// A LLM model.
pub trait Model {
    /// Call the model with the provided messages and functions.
    async fn call(
        &self,
        messages: impl AsRef<[Message]>,
        functions: impl AsRef<[Function]>,
    ) -> Result<Completion, super::Error>;
}

/// A message to the LLM.
#[derive(Debug, Clone)]
pub enum Message {
    /// A user message.
    User(Vec<UserContent>),
    /// An assistant message.
    Assistant(Vec<AssistantContent>),
}

/// A function to be called by the LLM.
#[derive(Debug, Clone)]
pub enum Function {
    /// A local function, provided in the codebase. Compatible with all models.
    Local {
        /// The name of the function. This is used by the LLM to identify the function when calling it.
        name: String,
        /// The description of the function. This allows the LLM to understand the function's purpose.
        description: String,
        /// The input schema of the function.
        input_schema: Value,
    },
    /// A function provided by a model provider. Provider-specific functions may work better for their models. May not be supported.
    Provider {
        /// The ID of the function. For example, anthropic's editor function has the "ID" of `text_editor_<date>`
        id: String,
        /// The name of the function. This is used by the LLM to identify the function when calling it.
        name: String,
    },
}

/// The content of a message.
/// 
/// Note that some LLMs may not support all possible modalities in this enum.
#[derive(Debug, Clone)]
pub enum Content {
    /// Text content.
    Text(String),
}

/// The content of a user message.
#[derive(Debug, Clone)]
pub enum UserContent {
    /// Content that the user has input.
    Input(Content),
    /// The result of a function call sent by the LLM.
    FunctionResult {
        /// The ID of the function call that these results are in response to.
        id: String,
        /// The result of the function call.
        result: Result<Vec<Content>, Content>,
    },
}

/// The content of an assistant message sent by the LLM.
#[derive(Debug, Clone)]
pub enum AssistantContent {
    /// The output of the LLM.
    Output(Content),
    /// A function call sent by the LLM.
    FunctionCall {
        /// The unique ID of this particular function call.
        /// When sending the result of this call, the result content should have the same ID.
        id: String,
        /// The name of the function that the LLM wants to call.
        name: String,
        /// The input to the function.
        input: Value,
    },
}

/// The completion of a message.
#[derive(Debug, Clone)]
pub struct Completion {
    /// Model usage statistics.
    pub usage: Usage,
    /// The content of the message.
    pub content: Vec<AssistantContent>,
}

/// Model usage statistics.
#[derive(Debug, Clone)]
pub struct Usage {
    /// The number of input tokens used.
    pub input_tokens: u32,
    /// The number of output tokens used.
    pub output_tokens: u32,
}
