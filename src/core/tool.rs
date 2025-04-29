use super::llm::Function;
use super::{Error, llm::Content};
use schemars::{JsonSchema, schema_for};
use serde::de::DeserializeOwned;
use serde_json::Value;

pub trait LocalTool {
    type Input: DeserializeOwned + JsonSchema;
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn call(&self, input: Self::Input) -> Result<Vec<Content>, Content>;
}

pub trait ProviderTool {
    type Input: DeserializeOwned;
    fn id(&self) -> String;
    fn name(&self) -> String;
    fn call(&self, input: Self::Input) -> Result<Vec<Content>, Content>;
}

pub struct Toolbox<'a> {
    tools: Vec<Box<dyn DynTool + 'a>>,
}

impl<'a> Toolbox<'a> {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    pub fn local<T: LocalTool + 'a>(mut self, tool: T) -> Self {
        self.tools.push(Box::new(LocalDynTool(tool)));
        self
    }

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
