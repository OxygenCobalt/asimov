mod anthropic;
mod core;

use anthropic::{Anthropic, AnthropicModel};
use colored::*;
use core::{
    agent::Agent,
    llm::{Hyperparams, Provider},
    tool::Toolbox,
};
use std::io::Write;

fn get_system_prompt() -> String {
    let os_name = std::env::consts::OS;
    let shell = std::env::var("SHELL").unwrap_or_else(|_| String::from("unknown"));
    let home_dir = dirs::home_dir()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let current_dir = std::env::current_dir()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    // This is just cline's system prompt, minus the custom tool calling (we assume the LLM can call tools without any coaxing)
    format!(
        "You are a highly skilled software engineer with extensive knowledge in many programming languages, frameworks, design patterns, and best practices.

SYSTEM INFORMATION:

Operating System: {}
Default Shell: {}
Home Directory: {}
Current Working Directory: {}

OBJECTIVE

You accomplish a given task iteratively, breaking it down into clear steps and working through them methodically.

1. Analyze the user's task and set clear, achievable goals to accomplish it. Prioritize these goals in a logical order.
2. Work through these goals sequentially, utilizing available tools one at a time as necessary. Each goal should correspond to a distinct step in your problem-solving process. You will be informed on the work completed and what's remaining as you go.
3. The user may provide feedback, which you can use to make improvements and try again. But DO NOT continue in pointless back and forth conversations, i.e. don't end your responses with questions or offers for further assistance.",
        os_name,
        shell,
        home_dir,
        current_dir
    )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::any::Any>> {
    dotenv::dotenv().unwrap();
    env_logger::init();
    let anthropic = Anthropic::new(std::env::var("ANTHROPIC_API_KEY").unwrap());
    let model = anthropic
        .obtain(
            anthropic::Claude::ThreeDotSevenSonnet,
            Some(get_system_prompt()),
            Hyperparams {
                max_tokens: 1024,
                temperature: 0.6,
            },
        )
        .await;
    let toolbox = Toolbox::new().provided(model.editor());
    let mut agent = Agent::new(model, toolbox);
    loop {
        print!("{} ", "you:".blue());
        std::io::stdout().flush().unwrap();
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        agent.go(input.to_string()).await.unwrap();
    }
}
