//! REST endpoints for the dashboard.

use std::sync::Arc;

use axum::{
    extract::{Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};

use inference::Predictor;
use market_domain::Snapshot;

use crate::state::AppState;

pub async fn healthz() -> &'static str {
    "ok"
}

#[derive(Serialize)]
pub struct ChampionResponse {
    /// Identifier of the predictor currently serving live decisions.
    /// Either an ONNX `model_id` or `"fallback-neutral"`.
    pub model_id: String,
    pub n_features: usize,
    pub kind: String,
}

/// `GET /api/strategy/champion` — what predictor is currently live?
pub async fn champion(State(state): State<Arc<AppState>>) -> Json<ChampionResponse> {
    let (model_id, n_features, kind) = match &state.inference {
        Some(reg) => {
            let h = reg.current();
            let id = h.id().to_string();
            let kind = if id == inference::fallback::FALLBACK_ID {
                "fallback".to_string()
            } else {
                "onnx".to_string()
            };
            (id, h.n_features(), kind)
        }
        None => (
            "uninitialised".to_string(),
            0,
            "none".to_string(),
        ),
    };
    Json(ChampionResponse {
        model_id,
        n_features,
        kind,
    })
}

#[derive(Deserialize)]
pub struct OptimizerTrialsQuery {
    #[serde(default)]
    pub study: Option<String>,
    #[serde(default = "default_trial_limit")]
    pub limit: usize,
}

fn default_trial_limit() -> usize {
    50
}

/// `GET /api/optimizer/trials?study=&limit=` — recent trader-optimiser trials.
pub async fn optimizer_trials(
    State(state): State<Arc<AppState>>,
    Query(q): Query<OptimizerTrialsQuery>,
) -> Result<Json<Vec<persistence::OptimizerTrialRow>>, StatusCode> {
    let db = state.db.as_ref().ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let rows = db
        .recent_optimizer_trials(q.study.as_deref(), q.limit)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(rows))
}

/// `GET /api/lockbox/result` — latest sealed lockbox row, if any.
pub async fn lockbox_result(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Option<persistence::LockboxRow>>, StatusCode> {
    let db = state.db.as_ref().ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let row = db
        .latest_lockbox_result()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(row))
}

#[derive(Deserialize)]
pub struct InstrumentQuery {
    pub instrument: String,
    #[serde(default = "default_history_limit")]
    pub limit: usize,
}

fn default_history_limit() -> usize {
    1000
}

/// `GET /api/model/deployment-gate?instrument=` — the latest deployment-gate
/// outcome for the given instrument. Includes the would-be champion's
/// metrics, whether it passed, the blocked reasons (if any), and the
/// thresholds in effect. Drives the dashboard's `DeploymentGateCard`.
pub async fn model_deployment_gate(
    State(state): State<Arc<AppState>>,
    Query(q): Query<InstrumentQuery>,
) -> Result<Json<Option<persistence::ModelDeploymentGateRow>>, StatusCode> {
    let db = state.db.as_ref().ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let row = db
        .latest_deployment_gate(&q.instrument)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(row))
}

/// `GET /api/model/candidates?instrument=` — all model-zoo candidates
/// from the most recent training run for `instrument`. Sorted by OOS
/// log loss (winner first). Drives the dashboard's "model zoo
/// comparison" table so the operator can see which family won and
/// what the runners-up scored.
pub async fn model_candidates(
    State(state): State<Arc<AppState>>,
    Query(q): Query<InstrumentQuery>,
) -> Result<Json<Vec<persistence::ModelCandidateRow>>, StatusCode> {
    let db = state.db.as_ref().ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let rows = db
        .latest_model_candidates(&q.instrument)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(rows))
}

/// `GET /api/model/metrics?instrument=` — current champion classifier metrics.
pub async fn model_metrics(
    State(state): State<Arc<AppState>>,
    Query(q): Query<InstrumentQuery>,
) -> Result<Json<Option<persistence::ModelMetricsRow>>, StatusCode> {
    let db = state.db.as_ref().ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let row = db
        .latest_model_metrics(&q.instrument)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(row))
}

/// `GET /api/trader/metrics` — most recent trader-parameter row.
pub async fn trader_metrics(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Option<persistence::TraderMetricsRow>>, StatusCode> {
    let db = state.db.as_ref().ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let row = db
        .latest_trader_metrics()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(row))
}

/// `GET /api/labels/recent?instrument=&limit=` — ideal entry points
/// the label optimiser picked on the trailing 1000 bars. Used by the
/// dashboard to overlay buy/sell markers on the price chart.
pub async fn labels_recent(
    State(state): State<Arc<AppState>>,
    Query(q): Query<InstrumentQuery>,
) -> Result<Json<Vec<persistence::LabelRow>>, StatusCode> {
    let db = state.db.as_ref().ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let rows = db
        .recent_labels(&q.instrument, q.limit)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(rows))
}

#[derive(Deserialize)]
pub struct TradeLedgerQuery {
    /// Optional. When omitted, returns recent trades across all instruments.
    pub instrument: Option<String>,
    #[serde(default = "default_history_limit")]
    pub limit: usize,
}

/// `GET /api/trade/ledger?instrument=&limit=` — recent closed
/// round-trip trades from the live trader. Used by the dashboard's
/// "live trade tape" panel.
pub async fn trade_ledger(
    State(state): State<Arc<AppState>>,
    Query(q): Query<TradeLedgerQuery>,
) -> Result<Json<Vec<persistence::TradeLedgerRow>>, StatusCode> {
    let db = state.db.as_ref().ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let rows = db
        .recent_trade_ledger(q.instrument.as_deref(), q.limit)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(rows))
}

#[derive(Deserialize)]
pub struct PipelineRunsQuery {
    #[serde(default = "default_pipeline_limit")]
    pub limit: usize,
}

fn default_pipeline_limit() -> usize {
    50
}

/// `GET /api/pipeline/runs?limit=` — recent Python research pipeline
/// invocations. Each `python -m research <cmd>` call writes one start
/// row + one finish row (in-place update via run_id).
pub async fn pipeline_runs(
    State(state): State<Arc<AppState>>,
    Query(q): Query<PipelineRunsQuery>,
) -> Result<Json<Vec<persistence::PipelineRunRow>>, StatusCode> {
    let db = state.db.as_ref().ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let rows = db
        .recent_pipeline_runs(q.limit)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(rows))
}

#[derive(Deserialize)]
pub struct TradeContextQuery {
    pub run_id: String,
    pub ts_in_ms: i64,
}

#[derive(Deserialize)]
pub struct ResearchArchiveQuery {
    /// Which time-series table's archive to read. Allowed:
    /// `bars_10s`, `champion_signals`, `signals`, `oof_predictions`.
    pub table: String,
    pub instrument: String,
    /// UTC date in `YYYY-MM-DD` form. The archive layout is one
    /// parquet per day per instrument.
    pub date: String,
}

/// `GET /api/research/archive?table=&instrument=&date=` — return the
/// archived parquet for one (table, instrument, date) tuple. Sourced
/// from the pre-sweep archive written by the retention task when
/// `RESEARCH_ARCHIVE_DIR` is configured.
pub async fn research_archive(
    State(_state): State<Arc<AppState>>,
    Query(q): Query<ResearchArchiveQuery>,
) -> Result<axum::response::Response<axum::body::Body>, (StatusCode, String)> {
    let allowed = ["bars_10s", "champion_signals", "signals", "oof_predictions"];
    if !allowed.contains(&q.table.as_str()) {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("table must be one of {allowed:?}"),
        ));
    }
    if q.date.len() != 10
        || !q.date[..4].chars().all(|c| c.is_ascii_digit())
        || &q.date[4..5] != "-"
        || !q.date[5..7].chars().all(|c| c.is_ascii_digit())
        || &q.date[7..8] != "-"
        || !q.date[8..10].chars().all(|c| c.is_ascii_digit())
    {
        return Err((StatusCode::BAD_REQUEST, "date must be YYYY-MM-DD".into()));
    }
    if !q
        .instrument
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '/'))
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "instrument must be alphanumeric/underscore/slash".into(),
        ));
    }

    let safe_instrument = q.instrument.replace('/', "_");
    let root = std::env::var("RESEARCH_ARCHIVE_DIR")
        .ok()
        .filter(|s| !s.is_empty())
        .ok_or((
            StatusCode::SERVICE_UNAVAILABLE,
            "RESEARCH_ARCHIVE_DIR not set".into(),
        ))?;
    let path = std::path::PathBuf::from(&root)
        .join(&q.table)
        .join(&safe_instrument)
        .join(format!("{}.parquet", q.date));
    if !path.exists() {
        return Err((StatusCode::NOT_FOUND, format!("no archive at {path:?}")));
    }
    let body = std::fs::read(&path)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("read: {e}")))?;
    Ok(axum::response::Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/octet-stream")
        .header(
            "content-disposition",
            format!(
                "attachment; filename=\"{}-{}-{}.parquet\"",
                q.table, safe_instrument, q.date
            ),
        )
        .header("cache-control", "public, max-age=3600")
        .body(axum::body::Body::from(body))
        .expect("valid response"))
}

#[derive(Deserialize)]
pub struct ResearchTradesQuery {
    /// Optional instrument filter. When omitted, exports every
    /// instrument.
    #[serde(default)]
    pub instrument: Option<String>,
    /// Inclusive lower bound on `trade_ledger.ts_out_ms`. Default 0.
    #[serde(default)]
    pub start_ms: Option<i64>,
    /// Inclusive upper bound on `trade_ledger.ts_out_ms`. Default
    /// year-3000-ish.
    #[serde(default)]
    pub end_ms: Option<i64>,
    /// `jsonl` (default) or `parquet`. Parquet is ~10× smaller +
    /// query-friendly; JSONL is grep-friendly + agent-friendly.
    #[serde(default)]
    pub format: Option<String>,
    /// Cap on rows. Default 10 000; server-capped by
    /// MAX_HISTORY_LIMIT.
    #[serde(default)]
    pub limit: Option<usize>,
}

/// `GET /api/research/trades?instrument=&start_ms=&end_ms=&format=`
/// — bulk export of trade_ledger × model_metrics for retroactive
/// study by an agent. Streams either JSONL or Parquet.
///
/// Implementation: DuckDB writes the rows to a temp file via native
/// `COPY ... TO`, the handler streams that file back to the client
/// with the appropriate Content-Type, then deletes it. This keeps the
/// in-memory footprint flat regardless of result size.
pub async fn research_trades(
    State(state): State<Arc<AppState>>,
    Query(q): Query<ResearchTradesQuery>,
) -> Result<axum::response::Response<axum::body::Body>, (StatusCode, String)> {
    let db = state
        .db
        .as_ref()
        .ok_or((StatusCode::SERVICE_UNAVAILABLE, "no DB".into()))?;
    let format = q.format.as_deref().unwrap_or("jsonl");
    if !matches!(format, "jsonl" | "parquet") {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("format must be jsonl|parquet, got {format:?}"),
        ));
    }
    let start_ms = q.start_ms.unwrap_or(0);
    // 32503680000000 ms ≈ year 3000.
    let end_ms = q.end_ms.unwrap_or(32_503_680_000_000);
    let limit = q.limit.unwrap_or(10_000);

    // Write to a unique temp file so concurrent requests don't collide.
    let stem = chrono::Utc::now().timestamp_micros();
    let ext = if format == "parquet" { "parquet" } else { "jsonl" };
    let tmp = std::env::temp_dir().join(format!("rtk-research-trades-{stem}.{ext}"));
    db.export_research_trades_to_file(
        &tmp,
        format,
        q.instrument.as_deref(),
        start_ms,
        end_ms,
        limit,
    )
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let body = std::fs::read(&tmp).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("read tmp export: {e}"),
        )
    })?;
    let _ = std::fs::remove_file(&tmp);

    let content_type = if format == "parquet" {
        "application/octet-stream"
    } else {
        "application/x-ndjson"
    };
    let filename = format!("trades-{stem}.{ext}");
    Ok(axum::response::Response::builder()
        .status(StatusCode::OK)
        .header("content-type", content_type)
        .header(
            "content-disposition",
            format!("attachment; filename=\"{filename}\""),
        )
        .header("cache-control", "no-store")
        .body(axum::body::Body::from(body))
        .expect("valid response"))
}

/// `GET /api/trade/context?run_id=&ts_in_ms=` — full forensic
/// payload for a single trade. Returns the JSON snapshot file
/// (Phase D3 forensics writer output) when one exists; falls back
/// to the trade_ledger row + a marker so the dashboard can still
/// render basic context for legacy trades whose snapshot was never
/// written.
pub async fn trade_context(
    State(state): State<Arc<AppState>>,
    Query(q): Query<TradeContextQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let db = state
        .db
        .as_ref()
        .ok_or((StatusCode::SERVICE_UNAVAILABLE, "no DB".into()))?;
    let row = db
        .trade_ledger_by_key(&q.run_id, q.ts_in_ms)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let row = row.ok_or((StatusCode::NOT_FOUND, "trade not found".into()))?;

    // If the per-trade snapshot file exists, return it verbatim — that's
    // the highest-fidelity context (includes pre/in/post-trade bar
    // windows). Otherwise return the bare trade_ledger row + a stub
    // payload so the dashboard renders gracefully.
    if let Some(p) = &row.snapshot_path {
        match std::fs::read_to_string(p) {
            Ok(text) => match serde_json::from_str::<serde_json::Value>(&text) {
                Ok(mut v) => {
                    if let Some(obj) = v.as_object_mut() {
                        obj.insert("source".into(), serde_json::json!("snapshot_file"));
                        obj.insert("snapshot_path".into(), serde_json::json!(p));
                    }
                    return Ok(Json(v));
                }
                Err(e) => {
                    tracing::warn!(error = %e, path = %p, "trade snapshot is not valid JSON");
                }
            },
            Err(e) => {
                tracing::warn!(error = %e, path = %p, "trade snapshot file missing or unreadable");
            }
        }
    }

    // Fallback: enrich the trade_ledger row inline.
    Ok(Json(serde_json::json!({
        "source": "trade_ledger_only",
        "run_id": row.run_id,
        "instrument": row.instrument,
        "side": row.side,
        "qty": row.qty,
        "entry_ts_ms": row.ts_in_ms,
        "entry_price": row.entry_px,
        "exit_ts_ms": row.ts_out_ms,
        "exit_price": row.exit_px,
        "realized_r": row.realized_r,
        "exit_reason": row.exit_reason,
        "model_id": row.model_id,
        "params_id": row.params_id,
        "entry_p_long": row.entry_p_long,
        "entry_p_short": row.entry_p_short,
        "entry_calibrated": row.entry_calibrated,
        "entry_spread_bp": row.entry_spread_bp,
        "entry_atr_14": row.entry_atr_14,
        "exit_p_long": row.exit_p_long,
        "exit_p_short": row.exit_p_short,
        "decision_chain": row.decision_chain,
        "snapshot_path": row.snapshot_path,
        "pre_trade_bars": [],
        "in_trade_bars": [],
        "post_trade_bars": [],
    })))
}

/// `GET /api/champion/signals?instrument=&limit=` — recent live
/// predictions from the deployed champion (or fallback).
pub async fn champion_signals(
    State(state): State<Arc<AppState>>,
    Query(q): Query<InstrumentQuery>,
) -> Result<Json<Vec<persistence::ChampionSignalRow>>, StatusCode> {
    let db = state.db.as_ref().ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let rows = db
        .recent_champion_signals(&q.instrument, q.limit)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(rows))
}

#[derive(Serialize)]
pub struct StateResponse {
    pub account_id: String,
    pub environment: String,
    pub instruments: Vec<String>,
    pub snapshot: Snapshot,
}

pub async fn current_state(State(state): State<Arc<AppState>>) -> Json<StateResponse> {
    Json(StateResponse {
        account_id: state.account_id.clone(),
        environment: state.cfg.environment.clone(),
        instruments: state.cfg.instruments.clone(),
        snapshot: state.snapshot(),
    })
}

#[derive(Serialize)]
pub struct InstrumentsResponse {
    pub instruments: Vec<String>,
}

pub async fn instruments(State(state): State<Arc<AppState>>) -> Json<InstrumentsResponse> {
    Json(InstrumentsResponse {
        instruments: state.cfg.instruments.clone(),
    })
}

// --- /api/history ----------------------------------------------------

#[derive(Deserialize)]
pub struct HistoryQuery {
    /// One of: `price`, `signal`, `fill`. Default: `price`.
    #[serde(default = "default_kind")]
    pub kind: String,
    /// Required for `price` and `signal`. Optional for `fill` (when
    /// omitted, returns the most recent fills across all instruments).
    pub instrument: Option<String>,
    /// Number of points to return (most recent). Default 1000, capped
    /// server-side at 10 000.
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_kind() -> String {
    "price".into()
}
fn default_limit() -> usize {
    1000
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HistoryResponse {
    Price {
        instrument: String,
        points: Vec<persistence::PriceHistoryPoint>,
    },
    Signal {
        instrument: String,
        points: Vec<persistence::SignalHistoryPoint>,
    },
    Fill {
        instrument: Option<String>,
        points: Vec<persistence::FillHistoryPoint>,
    },
}

/// `GET /api/history?kind=price&instrument=BTC/USD&limit=1000`
///
/// Returns the most-recent N points (in ascending time order) so the
/// dashboard can prefill its plots on connect. Falls back to an empty
/// array (not an error) when persistence is disabled — keeps the
/// dashboard's render path simple.
pub async fn history(
    State(state): State<Arc<AppState>>,
    Query(q): Query<HistoryQuery>,
) -> Result<Json<HistoryResponse>, (StatusCode, String)> {
    let limit = q.limit.min(persistence::MAX_HISTORY_LIMIT);

    let db = match &state.db {
        Some(db) => db,
        None => {
            // No persistence — return an empty result of the requested
            // shape so the client renders a blank chart cleanly instead
            // of erroring.
            return Ok(Json(empty_history(&q.kind, q.instrument.clone())));
        }
    };

    match q.kind.as_str() {
        "price" => {
            let inst = q.instrument.as_deref().ok_or((
                StatusCode::BAD_REQUEST,
                "kind=price requires instrument".to_string(),
            ))?;
            let points = db.recent_price_ticks(inst, limit).map_err(internal)?;
            Ok(Json(HistoryResponse::Price {
                instrument: inst.to_string(),
                points,
            }))
        }
        "signal" => {
            let inst = q.instrument.as_deref().ok_or((
                StatusCode::BAD_REQUEST,
                "kind=signal requires instrument".to_string(),
            ))?;
            let points = db.recent_signals(inst, limit).map_err(internal)?;
            Ok(Json(HistoryResponse::Signal {
                instrument: inst.to_string(),
                points,
            }))
        }
        "fill" => {
            let inst = q.instrument.as_deref();
            let points = db.recent_fills(inst, limit).map_err(internal)?;
            Ok(Json(HistoryResponse::Fill {
                instrument: q.instrument.clone(),
                points,
            }))
        }
        other => Err((
            StatusCode::BAD_REQUEST,
            format!("unknown kind: {other} (expected price | signal | fill)"),
        )),
    }
}

fn empty_history(kind: &str, instrument: Option<String>) -> HistoryResponse {
    match kind {
        "signal" => HistoryResponse::Signal {
            instrument: instrument.unwrap_or_default(),
            points: vec![],
        },
        "fill" => HistoryResponse::Fill {
            instrument,
            points: vec![],
        },
        _ => HistoryResponse::Price {
            instrument: instrument.unwrap_or_default(),
            points: vec![],
        },
    }
}

fn internal(e: anyhow::Error) -> (StatusCode, String) {
    tracing::warn!("history endpoint error: {e:#}");
    (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}
