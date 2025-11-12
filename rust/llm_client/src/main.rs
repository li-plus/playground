use dotenvy::dotenv;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use reqwest::{
    header::{HeaderMap, AUTHORIZATION, CONTENT_TYPE},
    Response,
};
use serde::{Deserialize, Serialize};
use std::env;
use std::io::{stdout, Write};
use tokio;
use tracing_subscriber::EnvFilter;

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

async fn naive_streaming(response: Response) -> Result<(), Box<dyn std::error::Error>> {
    let mut stream = response.bytes_stream();
    let mut buffer = String::new();

    while let Some(Ok(chunk)) = stream.next().await {
        let chunk_str = String::from_utf8_lossy(&chunk);
        buffer.push_str(&chunk_str);

        while let Some(pos) = buffer.find("\n\n") {
            let line = buffer[..pos].to_string();
            buffer = buffer[pos + 2..].to_string();

            if line.starts_with("data: ") {
                let data = &line[6..];

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

async fn eventsource_streaming(response: Response) -> Result<(), Box<dyn std::error::Error>> {
    let mut event_stream = response.bytes_stream().eventsource();

    while let Some(event) = event_stream.next().await {
        match event {
            Ok(event) => {
                if event.data == "[DONE]" {
                    println!();
                    return Ok(());
                }

                // Parse the JSON data from the event
                match serde_json::from_str::<ChunkResponse>(&event.data) {
                    Ok(chunk) => {
                        chunk
                            .choices
                            .into_iter()
                            .filter_map(|choice| choice.delta.content)
                            .for_each(|content| {
                                print!("{}", content);
                                let _ = stdout().flush();
                            });
                    }
                    Err(e) => {
                        eprintln!("Error parsing JSON: {} - Data: {}", e, event.data);
                    }
                }
            }
            Err(e) => {
                eprintln!("Error reading SSE event: {}", e);
            }
        }
    }

    Ok(())
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

    let use_eventsource = true;
    if use_eventsource {
        eventsource_streaming(response).await
    } else {
        naive_streaming(response).await
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

    let base_url = env::var("OPENAI_BASE_URL").unwrap_or(String::from(DEFAULT_BASE_URL));
    let api_url = base_url + "/chat/completions";
    let api_key = env::var("OPENAI_API_KEY").unwrap();
    let model_name = env::var("MODEL_NAME").unwrap();

    let mut chat_request = ChatRequest {
        model: model_name,
        messages: vec![Message {
            role: String::from("user"),
            content: String::from("Count from 1 to 100 in a line separated by comma."),
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
    println!("--- Sync Response ---");
    chat(&client, &api_url, &chat_request).await?;

    // streaming
    println!("--- Streaming Response ---");
    chat_request.stream = Some(true);
    chat(&client, &api_url, &chat_request).await?;

    Ok(())
}
