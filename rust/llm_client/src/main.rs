use dotenvy::dotenv;
use futures::StreamExt;
use reqwest;
use reqwest::header::{HeaderMap, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use std::env;
use std::io::{stdout, Write};
use tokio;

#[derive(Serialize, Deserialize, Debug)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    stream: Option<bool>,
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

#[derive(Serialize, Deserialize, Debug)]
struct ChunkResponse {
    choices: Vec<DeltaChoice>,
}

#[derive(Serialize, Deserialize, Debug)]
struct DeltaChoice {
    delta: DeltaMessage,
}

#[derive(Serialize, Deserialize, Debug)]
struct DeltaMessage {
    role: Option<String>,
    content: Option<String>,
}

async fn chat(
    client: &reqwest::Client,
    api_url: &String,
    chat_request: &ChatRequest,
) -> Result<(), Box<dyn std::error::Error>> {
    let response = client.post(api_url).json(chat_request).send().await?;

    if !response.status().is_success() {
        return Err(format!(
            "Error: {}. Info: {}",
            response.status(),
            response.text().await?
        )
        .into());
    }

    // sync chat
    if !chat_request.stream.unwrap_or(false) {
        let chat_response: ChatResponse = response.json().await?;
        chat_response.choices.iter().for_each(|choice| {
            println!("{}: {}", choice.message.role, choice.message.content);
        });
        return Ok(());
    }

    // streaming
    let mut stream = response.bytes_stream();
    let mut buffer = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let chunk_str = String::from_utf8_lossy(&chunk);
        buffer.push_str(&chunk_str);

        while let Some(pos) = buffer.find("\n\n") {
            let line = buffer[..pos].to_string();
            buffer = buffer[pos + 2..].to_string();

            if line.starts_with("data: ") {
                let data = &line[6..];

                // println!("data: {:#?}", data);

                if data == "[DONE]" {
                    println!(); // New line after streaming
                    return Ok(());
                }

                match serde_json::from_str::<ChunkResponse>(data) {
                    Ok(chunk) => {
                        chunk
                            .choices
                            .into_iter()
                            .filter_map(|choice| choice.delta.content)
                            .for_each(|content| {
                                print!("{}", content);
                                stdout().flush().unwrap();
                            });
                    }
                    Err(e) => {
                        eprintln!("Error parsing JSON: {} - Data: {}", e, data);
                    }
                }
            }
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

    let base_url = env::var("OPENAI_BASE_URL").unwrap_or(String::from(DEFAULT_BASE_URL));
    let api_url = base_url + "/chat/completions";
    let api_key = env::var("OPENAI_API_KEY").unwrap();
    let model_name = env::var("MODEL_NAME").unwrap();

    let mut chat_request = ChatRequest {
        model: model_name,
        messages: vec![Message {
            role: String::from("user"),
            content: String::from("Write a quick sort function in rust"),
        }],
        stream: None,
    };
    println!(
        "{}: {}",
        chat_request.messages[0].role, chat_request.messages[0].content
    );

    let mut headers = HeaderMap::new();
    headers.insert(
        AUTHORIZATION,
        format!("Bearer {}", api_key).parse().unwrap(),
    );
    headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());

    let client = reqwest::Client::builder()
        .default_headers(headers)
        .build()?;

    // sync
    // chat(&client, &api_url, &chat_request).await?;

    // streaming
    chat_request.stream = Some(true);
    chat(&client, &api_url, &chat_request).await?;

    Ok(())
}
