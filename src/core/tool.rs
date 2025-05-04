use super::llm::Function;
use super::{Error, llm::Content};
use schemars::{JsonSchema, schema_for};
use serde::de::DeserializeOwned;
use serde_json::Value;

/// A local tool, defined in the codebase.
pub trait LocalTool {
    /// The input type of the tool. This should be a deserializable struct with well-annotated fields
    /// that the LLM can easily understand.
    type Input: DeserializeOwned + JsonSchema;
    /// The name of the tool. The LLM will use this name to identify the tool when calling it.
    /// Must be unique within the toolbox.
    fn name(&self) -> &'static str;
    /// The description of the tool. This will be used by the LLM to understand the tool's purpose
    /// and when to call it. Be detailed!
    fn description(&self) -> &'static str;
    /// The function that the tool will call.
    fn call(&self, input: Self::Input) -> Result<Vec<Content>, Content>;
}

/// A tool provided by a model provider.
pub trait ProviderTool {
    type Input: DeserializeOwned;
    /// The unique ID of the tool as provided by the model provider. For example, Anthropic's editor
    /// tool has the ID `text_editor_<date>`.
    fn id(&self) -> String;
    /// The name of the tool. The LLM will use this name to identify the tool when calling it.
    /// Must be unique within the toolbox.
    fn name(&self) -> String;
    /// The function that the tool will call.
    fn call(&self, input: Self::Input) -> Result<Vec<Content>, Content>;
}

/// A collection of tools that can be used by the agent.
pub struct Toolbox<'a> {
    tools: Vec<Box<dyn DynTool + 'a>>,
}

impl<'a> Toolbox<'a> {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    /// Add a local tool to the toolbox. The tool must live for the lifetime of the toolbox.
    pub fn local<T: LocalTool + 'a>(mut self, tool: T) -> Self {
        self.tools.push(Box::new(LocalDynTool(tool)));
        self
    }

    /// Add a provider tool to the toolbox. The tool must live for the lifetime of the toolbox.
    pub fn provided<T: ProviderTool + 'a>(mut self, tool: T) -> Self {
        self.tools.push(Box::new(ProviderDynTool(tool)));
        self
    }

    pub(crate) fn call(&self, name: &str, input: Value) -> Result<Vec<Content>, Content> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.is(name))
            .ok_or(Content::Text(format!(
                "Cannot use '{}' because it was not found.",
                name
            )))?;
        tool.call(input)
    }

    pub(super) fn functions(&self) -> Result<Vec<Function>, Error> {
        self.tools.iter().map(|t| t.function()).collect()
    }
}

// The plain tool trait is great for implementations but can't be used for trait objects,
// so we create some wrapper traits here that are dyn-compatible at the cost of having
// no type safety. This is okay since we have everything we need to validate arguments
// based on the tool's type information.

trait DynTool {
    fn is(&self, name: &str) -> bool;
    fn function(&self) -> Result<Function, Error>;
    fn call(&self, input: Value) -> Result<Vec<Content>, Content>;
}

struct LocalDynTool<T: LocalTool>(T);

impl<T: LocalTool> DynTool for LocalDynTool<T> {
    fn is(&self, name: &str) -> bool {
        self.0.name() == name
    }

    fn function(&self) -> Result<Function, Error> {
        Ok(Function::Local {
            name: self.0.name().to_string(),
            description: self.0.description().to_string(),
            input_schema: serde_json::to_value(schema_for!(T::Input))?,
        })
    }

    fn call(&self, input: Value) -> Result<Vec<Content>, Content> {
        let value =
            serde_json::from_value::<T::Input>(input).map_err(|e| Content::Text(e.to_string()))?;
        self.0.call(value)
    }
}

struct ProviderDynTool<T: ProviderTool>(T);

impl<T: ProviderTool> DynTool for ProviderDynTool<T> {
    fn is(&self, name: &str) -> bool {
        self.0.name() == name
    }

    fn function(&self) -> Result<Function, Error> {
        Ok(Function::Provider {
            id: self.0.id(),
            name: self.0.name(),
        })
    }

    fn call(&self, input: Value) -> Result<Vec<Content>, Content> {
        let value =
            serde_json::from_value::<T::Input>(input).map_err(|e| Content::Text(e.to_string()))?;
        self.0.call(value)
    }
}
