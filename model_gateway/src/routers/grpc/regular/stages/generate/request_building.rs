//! Generate request building stage: Build proto GenerateRequest for generate requests

use async_trait::async_trait;
use axum::response::Response;
use openai_protocol::generate::GenerateRequest;
use tracing::error;
use uuid::Uuid;

use crate::routers::{
    error,
    grpc::{
        common::stages::{helpers, PipelineStage},
        context::{ClientSelection, RequestContext},
        proto_wrapper::ProtoRequest,
        utils::stop_strings_to_token_ids,
    },
};

/// Generate request building stage
///
/// Extracts generate-specific request building logic from the old unified RequestBuildingStage.
pub(crate) struct GenerateRequestBuildingStage {
    inject_pd_metadata: bool,
}

impl GenerateRequestBuildingStage {
    pub fn new(inject_pd_metadata: bool) -> Self {
        Self { inject_pd_metadata }
    }
}

#[async_trait]
impl PipelineStage for GenerateRequestBuildingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let prep = ctx.state.preparation.as_ref().ok_or_else(|| {
            error!(
                function = "GenerateRequestBuildingStage::execute",
                "Preparation not completed"
            );
            error::internal_error("preparation_not_completed", "Preparation not completed")
        })?;

        let clients = ctx.state.clients.as_ref().ok_or_else(|| {
            error!(
                function = "GenerateRequestBuildingStage::execute",
                "Client acquisition not completed"
            );
            error::internal_error(
                "client_acquisition_not_completed",
                "Client acquisition not completed",
            )
        })?;

        let generate_request = ctx.generate_request_arc();

        // Get client for building request (use prefill client if PD mode)
        let builder_client = match clients {
            ClientSelection::Single { client } => client,
            ClientSelection::Dual { prefill, .. } => prefill,
        };

        // Build generate request
        let request_id = generate_request
            .rid
            .clone()
            .unwrap_or_else(|| format!("gen-{}", Uuid::now_v7()));

        // For MLX: pre-tokenize string stop sequences in sampling_params into stop_token_ids.
        let mlx_modified: Option<GenerateRequest> = if builder_client.is_mlx() {
            generate_request
                .sampling_params
                .as_ref()
                .and_then(|sp| sp.stop.as_ref().filter(|s| !s.is_empty()).map(|s| (s, sp)))
                .and_then(|(stop, _sp)| {
                    ctx.state.tokenizer.as_deref().map(|tok| (stop, tok))
                })
                .map(|(stop, tok)| {
                    let extra_ids = stop_strings_to_token_ids(stop, tok);
                    let mut req = (*generate_request).clone();
                    if let Some(sp) = req.sampling_params.as_mut() {
                        let mut merged = sp.stop_token_ids.take().unwrap_or_default();
                        merged.extend(extra_ids);
                        sp.stop = None;
                        sp.stop_token_ids = if merged.is_empty() { None } else { Some(merged) };
                    }
                    req
                })
        } else {
            None
        };
        let generate_request_body: &GenerateRequest =
            mlx_modified.as_ref().map_or(&*generate_request, |r| r);

        // Build proto request using centralized dispatch
        let mut proto_request = builder_client
            .build_generate_request(
                request_id,
                generate_request_body,
                prep.routing_text().map(String::from),
                prep.token_ids().to_vec(),
            )
            .map_err(|e| {
                error!(function = "GenerateRequestBuildingStage::execute", error = %e, "Failed to build generate request");
                error::bad_request("build_request_failed", e)
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
        "GenerateRequestBuilding"
    }
}
