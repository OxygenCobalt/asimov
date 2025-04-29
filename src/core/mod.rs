pub mod agent;
pub mod llm;
pub mod tool;

#[derive(Debug)]
pub enum Error {
    IO(std::io::Error),
    Reqwest(reqwest::Error),
    Serde(serde_json::Error),
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
