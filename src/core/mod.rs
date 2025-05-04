pub mod agent;
pub mod llm;
pub mod tool;

/// Possible errors that can occur when interacting with the agent.
#[derive(Debug)]
pub enum Error {
    /// An error occurred when reading or writing to a file.
    IO(std::io::Error),
    /// An error occurred when sending a request to the LLM provider.
    Reqwest(reqwest::Error),
    /// An error occurred when parsing JSON.
    Serde(serde_json::Error),
    /// An internal error occurred in the LLM provider.
    Provider(String),
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Error {
        Error::IO(error)
    }
}

impl From<reqwest::Error> for Error {
    fn from(error: reqwest::Error) -> Error {
        Error::Reqwest(error)
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Error {
        Error::Serde(error)
    }
}
