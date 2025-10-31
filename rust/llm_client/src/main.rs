use dotenvy::dotenv;
use reqwest;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use std::env;
use tokio;

#[derive(Serialize, Deserialize, Debug)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
}

#[derive(Serialize, Deserialize, Debug)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Serialize, Deserialize, Debug)]
struct Choice {
    message: Message,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

    let base_url = env::var("OPENAI_BASE_URL").unwrap_or(String::from(DEFAULT_BASE_URL));
    let api_key = env::var("OPENAI_API_KEY").unwrap();
    let model_name = env::var("MODEL_NAME").unwrap();

    let chat_request = ChatRequest {
        model: model_name,
        messages: vec![Message {
            role: String::from("user"),
            content: String::from("Hello, how are you?"),
        }],
    };
    println!(
        "{}: {}",
        chat_request.messages[0].role, chat_request.messages[0].content
    );

    let client = reqwest::Client::new();
    let response = client
        .post(base_url + "/chat/completions")
        .json(&chat_request)
        .header(AUTHORIZATION, format!("Bearer {}", api_key))
        .header(CONTENT_TYPE, "application/json")
        .send()
        .await?;

    let chat_response: ChatResponse = response.json().await?;
    println!(
        "{}: {}",
        chat_response.choices[0].message.role, chat_response.choices[0].message.content
    );

    Ok(())
}
