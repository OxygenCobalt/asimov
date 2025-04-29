use serde_json::Value;

pub trait Provider<T> {
    async fn obtain(
        &self,
        model: T,
        system_prompt: Option<impl AsRef<str>>,
        hyperparams: Hyperparams,
    ) -> impl Model;
}

#[derive(Debug, Clone, Copy)]
pub struct Hyperparams {
    pub max_tokens: u32,
    pub temperature: f64,
}

pub trait Model {
    async fn call(
        &self,
        messages: impl AsRef<[Message]>,
        functions: impl AsRef<[Function]>,
    ) -> Result<Completion, super::Error>;
}

#[derive(Debug, Clone)]
pub enum Message {
    User(Vec<UserContent>),
    Assistant(Vec<AssistantContent>),
}
#[derive(Debug, Clone)]
pub enum Function {
    Local {
        name: String,
        description: String,
        input_schema: Value,
    },
    Provider {
        id: String,
        name: String,
    },
}

#[derive(Debug, Clone)]
pub enum Content {
    Text(String),
}

#[derive(Debug, Clone)]
pub enum UserContent {
    Input(Content),
    FunctionResult {
        id: String,
        result: Result<Vec<Content>, Content>,
    },
}

#[derive(Debug, Clone)]
pub enum AssistantContent {
    Output(Content),
    FunctionCall {
        id: String,
        name: String,
        input: Value,
    },
}

#[derive(Debug, Clone)]
pub struct Completion {
    pub usage: Usage,
    pub content: Vec<AssistantContent>,
}

#[derive(Debug, Clone)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}
