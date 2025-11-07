use axum::{
    body::Body,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use config::{Config, File};
use mime::APPLICATION_JSON;
use reqwest::{
    header::{HeaderMap, AUTHORIZATION, CONTENT_TYPE},
    Client,
};
use serde::Deserialize;
use serde_json::Value as JsonValue;
use std::{collections::HashMap, sync::Arc};
use tower_http::trace::TraceLayer;
use tracing::info;

struct AppState {
    client: Client,
    config: AppConfig,
}

#[derive(Deserialize)]
struct ProviderConfig {
    api_url: String,
    api_key: String,
    model: String,
}

#[derive(Deserialize)]
struct AppConfig {
    providers: HashMap<String, ProviderConfig>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .init();

    let config: AppConfig = Config::builder()
        .add_source(File::with_name("config/default"))
        .build()?
        .try_deserialize()?;

    let addr = "0.0.0.0:8080";

    let client = reqwest::Client::new();

    let state = Arc::new(AppState { client, config });

    let app = Router::new()
        .route("/chat/completions", post(chat_completions))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();

    Ok(())
}

async fn chat_completions(
    State(app_state): State<Arc<AppState>>,
    request: Json<JsonValue>,
) -> Response {
    let mut request = request.0;

    let Some(JsonValue::String(model_ref)) = request.get_mut("model") else {
        return StatusCode::BAD_REQUEST.into_response();
    };

    let Some(provider) = app_state.config.providers.get(model_ref) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    *model_ref = provider.model.clone();

    let mut headers = HeaderMap::new();
    headers.insert(
        AUTHORIZATION,
        format!("Bearer {}", provider.api_key).parse().unwrap(),
    );
    headers.insert(CONTENT_TYPE, APPLICATION_JSON.as_ref().parse().unwrap());

    // info!("Headers: {:#?}", headers);

    // info!("Received payload: {:#?}", request);

    let Ok(upstream_response) = app_state
        .client
        .post(&provider.api_url)
        .headers(headers)
        .json(&request)
        .send()
        .await
    else {
        return StatusCode::BAD_GATEWAY.into_response();
    };

    // info!("Response: {:#?}", upstream_response);

    let status = upstream_response.status();
    let headers = upstream_response.headers().clone();

    let Ok(mut response) = Response::builder()
        .status(status)
        .body(Body::from_stream(upstream_response.bytes_stream()))
    else {
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    };

    *response.headers_mut() = headers;

    response
}
