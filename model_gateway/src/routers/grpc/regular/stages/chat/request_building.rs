//! Chat request building stage: Build proto GenerateRequest for chat requests

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;
use uuid::Uuid;

use openai_protocol::chat::ChatCompletionRequest;

use crate::routers::{
    error,
    grpc::{
        common::stages::{helpers, PipelineStage},
        context::{ClientSelection, PreparationOutput, RequestContext},
        multimodal::assemble_multimodal_data,
        proto_wrapper::ProtoRequest,
        utils::stop_strings_to_token_ids,
    },
};

/// Chat request building stage
///
/// Extracts chat-specific request building logic from the old unified RequestBuildingStage.
pub(crate) struct ChatRequestBuildingStage {
    inject_pd_metadata: bool,
}

impl ChatRequestBuildingStage {
    pub fn new(inject_pd_metadata: bool) -> Self {
        Self { inject_pd_metadata }
    }
}

#[async_trait]
impl PipelineStage for ChatRequestBuildingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Take preparation state (last consumer — worker_selection already ran)
        let prep = ctx.state.preparation.take().ok_or_else(|| {
            error!(
                function = "ChatRequestBuildingStage::execute",
                "Preparation not completed"
            );
            error::internal_error("preparation_not_completed", "Preparation not completed")
        })?;

        let clients = ctx.state.clients.as_ref().ok_or_else(|| {
            error!(
                function = "ChatRequestBuildingStage::execute",
                "Client acquisition not completed"
            );
            error::internal_error(
                "client_acquisition_not_completed",
                "Client acquisition not completed",
            )
        })?;

        let chat_request = ctx.chat_request_arc();

        // Get client for building request (use prefill client if PD mode)
        let builder_client = match clients {
            ClientSelection::Single { client } => client,
            ClientSelection::Dual { prefill, .. } => prefill,
        };

        let PreparationOutput::Chat {
            token_ids,
            processed_messages,
            tool_constraints,
        } = prep
        else {
            debug_assert!(false, "pipeline guarantees Chat variant");
            return Err(error::internal_error(
                "wrong_preparation_type",
                "Expected Chat preparation output",
            ));
        };

        // Build chat request
        let request_id = format!("chatcmpl-{}", Uuid::now_v7());

        // For MLX: string stop sequences are pre-tokenized into stop_token_ids because
        // the MLX proto only supports token-ID stops, not string stops.
        // Other backends receive string stops natively through body.stop.
        let mlx_modified: Option<ChatCompletionRequest> = if builder_client.is_mlx() {
            chat_request
                .stop
                .as_ref()
                .filter(|s| !s.is_empty())
                .and_then(|stop| ctx.state.tokenizer.as_deref().map(|tok| (stop, tok)))
                .map(|(stop, tok)| {
                    let extra_ids = stop_strings_to_token_ids(stop, tok);
                    let mut req = (*chat_request).clone();
                    let mut merged = req.stop_token_ids.take().unwrap_or_default();
                    merged.extend(extra_ids);
                    req.stop = None;
                    req.stop_token_ids = if merged.is_empty() { None } else { Some(merged) };
                    req
                })
        } else {
            None
        };
        let chat_request_body: &ChatCompletionRequest =
            mlx_modified.as_ref().map_or(&*chat_request, |r| r);

        // Reject multimodal for backends that don't support it, before assembling
        if processed_messages.multimodal_intermediate.is_some() && builder_client.is_mlx() {
            return Err(error::bad_request(
                "multimodal_not_supported",
                "MLX backend does not support multimodal inputs".to_string(),
            ));
        }

        // Assemble backend-specific multimodal data now that the backend is known
        let multimodal_data = processed_messages
            .multimodal_intermediate
            .map(|intermediate| assemble_multimodal_data(intermediate, builder_client));

        let mut proto_request = builder_client
            .build_chat_request(
                request_id,
                chat_request_body,
                processed_messages.text,
                token_ids,
                multimodal_data,
                tool_constraints,
            )
            .map_err(|e| {
                error!(function = "ChatRequestBuildingStage::execute", error = %e, "Failed to build generate request");
                error::bad_request("invalid_request_parameters", format!("Invalid request parameters: {e}"))
            })?;

        if self.inject_pd_metadata {
            if let Some(workers) = ctx.state.workers.as_ref() {
                helpers::maybe_inject_pd_metadata(&mut proto_request, workers);
            }
        }

        ctx.state.proto_request = Some(ProtoRequest::Generate(proto_request));
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "ChatRequestBuilding"
    }
}
