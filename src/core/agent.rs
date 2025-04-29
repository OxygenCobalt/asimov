use super::llm::{AssistantContent, Content, Message, Model, UserContent};
use super::tool::Toolbox;
use colored::*;

pub struct Agent<'a, M: Model> {
    model: M,
    toolbox: Toolbox<'a>,
    history: Vec<Message>,
}

impl<'a, M: Model> Agent<'a, M> {
    pub fn new(model: M, toolbox: Toolbox<'a>) -> Self {
        Self {
            model,
            toolbox,
            history: Vec::new(),
        }
    }

    pub async fn go(&mut self, and: String) -> Result<(), super::Error> {
        let mut send = vec![UserContent::Input(Content::Text(and))];
        while !send.is_empty() {
            self.history.push(Message::User(send.drain(..).collect()));
            let completion = self
                .model
                .call(&self.history, &self.toolbox.functions()?)
                .await?;
            for content in &completion.content {
                match content {
                    AssistantContent::Output(content) => {
                        let Content::Text(s) = content;
                        println!("{}: {}", "agent".green(), s);
                    }

                    AssistantContent::FunctionCall { id, name, input } => {
                        print!("{}: {}", "tool".red(), name);
                        let function_result = self.toolbox.call(name, input.clone());
                        match &function_result {
                            Ok(_) => {
                                println!(" -> {}", "ok".green());
                            }
                            Err(e) => {
                                let Content::Text(s) = e;
                                println!(" -> {}: {}", "err".red(), s);
                            }
                        }
                        let result = UserContent::FunctionResult {
                            id: id.clone(),
                            result: function_result,
                        };
                        send.push(result);
                    }
                }
            }
            self.history.push(Message::Assistant(completion.content));
        }
        Ok(())
    }
}
