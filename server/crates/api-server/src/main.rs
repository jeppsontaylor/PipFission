//! Live OANDA v20 streaming server entry point.
//!
//! Streams pricing + transactions from OANDA's practice environment, polls
//! the account summary, runs an internal "estimated balance" reconciler,
//! and fans the whole thing out over a WebSocket so the React dashboard
//! can render it in real time.

use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use axum::{routing::get, Router};
use tokio::sync::mpsc;
use tower_http::cors::CorsLayer;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use market_domain::Config;
use oanda_adapter::streams::{account_poll, pricing, transactions};
use oanda_adapter::{replay, synthetic, Client};

use alpaca_adapter::AlpacaConfig;

use api_server::{alpaca_runner, estimator, http as http_api, runners, ws, AppState};

#[tokio::main]
async fn main() -> Result<()> {
    // Best-effort .env loading (search current dir then parent — handy when
    // the binary lives at server/target/... and .env lives at the repo root).
    let _ = dotenvy::dotenv();
    let _ = dotenvy::from_path("../.env");
    let _ = dotenvy::from_path("../../.env");

    tracing_subscriber::registry()
        .with(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,api_server=debug,oanda_adapter=debug")),
        )
        .with(fmt::layer().with_target(false).with_line_number(false))
        .init();

    let mut cfg = Config::from_env().context("loading config from env")?;
    // Merge Alpaca symbols into the instrument list so the dashboard sees
    // them in the Hello payload + ML status / signals tables.
    if let Ok(ac) = AlpacaConfig::from_env() {
        for sym in &ac.symbols {
            if !cfg.instruments.contains(sym) {
                cfg.instruments.push(sym.clone());
            }
        }
    }
    tracing::info!(
        instruments = ?cfg.instruments,
        env = %cfg.environment,
        bind = %cfg.bind_addr,
        "starting api-server"
    );

    // Resolve the account ID (auto-discover if not provided).
    let oanda_client = Client::new(cfg.clone()).context("building OANDA client")?;
    let account_id = match cfg.account_id.clone() {
        Some(id) => id,
        None => {
            tracing::info!("no OANDA_ACCOUNT_ID set, auto-discovering...");
            oanda_client
                .discover_account_id()
                .await
                .context("auto-discovering account ID")?
        }
    };
    tracing::info!(%account_id, "using OANDA account");

    // Build AppState. If ALLOW_OANDA_ROUTING=true we also wire an OandaRouter
    // so the mode toggle can flip to OandaPractice without a restart.
    let mut app_state = AppState::new(cfg.clone(), account_id.clone());
    if std::env::var("ALLOW_OANDA_ROUTING").as_deref() == Ok("true") {
        let oanda_router = std::sync::Arc::new(oanda_adapter::OandaRouter::new(
            oanda_client.clone(),
            account_id.clone(),
            app_state.price_oracle.clone(),
        ));
        let dyn_router: std::sync::Arc<dyn portfolio::OrderRouter> = oanda_router;
        app_state = app_state.with_oanda_router(dyn_router);
        tracing::warn!("ALLOW_OANDA_ROUTING=true; OandaRouter wired (mode flip enabled)");
    }

    // Alpaca: present iff ALPACA_KEY + ALPACA_SECRET are in env. We also
    // build an AlpacaRouter (Live mode) when present — there is no
    // separate "ALLOW_ALPACA_ROUTING" because the same `Live` mode toggle
    // controls both venues.
    let alpaca_cfg_result = AlpacaConfig::from_env();
    let mut alpaca_router_held: Option<std::sync::Arc<alpaca_adapter::AlpacaRouter>> = None;
    if let Ok(ref ac) = alpaca_cfg_result {
        let alpaca_router = std::sync::Arc::new(
            alpaca_adapter::AlpacaRouter::new(ac.clone(), app_state.price_oracle.clone())
                .context("building AlpacaRouter")?,
        );
        let dyn_router: std::sync::Arc<dyn portfolio::OrderRouter> = alpaca_router.clone();
        app_state = app_state.with_alpaca_router(dyn_router);
        alpaca_router_held = Some(alpaca_router);
        tracing::warn!(
            symbols = ?ac.symbols,
            "ALPACA enabled; data WS + paper-trading router wired"
        );
    }

    // === Persistence: DuckDB-backed store, HARD 10k row per-instrument cap ===
    // Default path is `./data/oanda.duckdb`; override with DATABASE_PATH.
    // Empty string disables persistence (useful for tests / ephemeral runs).
    let db_path = std::env::var("DATABASE_PATH")
        .unwrap_or_else(|_| "./data/oanda.duckdb".to_string());
    let mut db_opt: Option<persistence::Db> = None;
    if !db_path.is_empty() {
        match persistence::Db::open(&db_path) {
            Ok(db) => {
                let counts = db.row_counts().unwrap_or_default();
                tracing::info!(
                    path = %db.path().display(),
                    ?counts,
                    "persistence: opened DuckDB store"
                );
                db_opt = Some(db);
            }
            Err(e) => {
                tracing::error!("persistence: failed to open db at {db_path}: {e:#}; continuing without persistence");
            }
        }
    } else {
        tracing::warn!("persistence: DATABASE_PATH empty; running without persistence");
    }

    if let Some(ref db) = db_opt {
        app_state = app_state.with_db(db.clone());
    }

    // M10 — inference registry. Created here so it can be attached to
    // AppState before the Arc wrap; the file-watcher hot-swap is
    // spawned later (it only needs the registry, not the state).
    let inference_registry = std::sync::Arc::new(
        inference::PredictorRegistry::new(market_domain::FEATURE_DIM),
    );
    app_state = app_state.with_inference(inference_registry.clone());

    // Auto-retrain handle. Constructed regardless of `enabled` flag so
    // the dashboard can show "off + N tracked instruments" before the
    // operator opts in.
    let auto_retrain_cfg = api_server::auto_retrain::AutoRetrainConfig::from_env();
    let auto_retrain = std::sync::Arc::new(api_server::auto_retrain::AutoRetrain::new(
        auto_retrain_cfg.clone(),
    ));
    app_state = app_state.with_auto_retrain(auto_retrain.clone());

    let state = Arc::new(app_state);

    // Spawn the auto-retrain bar listener. Always running so counters
    // populate for telemetry; only fires subprocesses when `enabled`.
    api_server::auto_retrain::spawn(state.clone(), auto_retrain.clone());
    tracing::info!(
        enabled = auto_retrain_cfg.enabled,
        threshold = auto_retrain_cfg.bars_threshold,
        instruments = ?auto_retrain_cfg.instruments,
        "auto-retrain task wired"
    );

    // === Spawn pure-IO stream tasks (oanda-adapter) and forwarders (api-server) ===

    let (pricing_tx, pricing_rx) = mpsc::unbounded_channel();
    // Pricing source priority: OANDA_REPLAY (real recorded ticks from the
    // Python collector) > SYNTHETIC_TICKS (random walks, last resort) > live
    // OANDA stream. Replay is the recommended demo mode when forex is closed.
    let replay_dir = std::env::var("OANDA_REPLAY")
        .ok()
        .filter(|s| !s.is_empty())
        .map(std::path::PathBuf::from);
    let synthetic_enabled = std::env::var("SYNTHETIC_TICKS").as_deref() == Ok("true");
    if let Some(dir) = replay_dir {
        let speed: f64 = std::env::var("OANDA_REPLAY_SPEED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(100.0);
        tracing::warn!(
            speed,
            dir = %dir.display(),
            "OANDA_REPLAY set; replaying real recorded OANDA pricing data"
        );
        replay::spawn(dir, pricing_tx, speed);
    } else if synthetic_enabled {
        let hz: f64 = std::env::var("SYNTHETIC_HZ")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(100.0);
        tracing::warn!(
            hz,
            "SYNTHETIC_TICKS=true; using random-walk price source (DEMO/TEST ONLY)"
        );
        synthetic::spawn(cfg.instruments.clone(), pricing_tx, hz);
    } else {
        tracing::info!("using live OANDA pricing stream");
        pricing::spawn(
            oanda_client.clone(),
            account_id.clone(),
            cfg.instruments.clone(),
            pricing_tx,
        );
    }
    runners::pricing_forwarder(state.clone(), pricing_rx);

    let (tx_tx, tx_rx) = mpsc::unbounded_channel();
    transactions::spawn(oanda_client.clone(), account_id.clone(), tx_tx);
    runners::transaction_forwarder(state.clone(), tx_rx);

    let (acct_tx, acct_rx) = mpsc::unbounded_channel();
    account_poll::spawn(
        oanda_client.clone(),
        account_id.clone(),
        cfg.account_poll_ms,
        acct_tx,
    );
    runners::account_forwarder(state.clone(), acct_rx);

    estimator::spawn(state.clone());

    // Alpaca data WS + account poller (when configured).
    if let Ok(ac) = alpaca_cfg_result {
        let (alpaca_tx, alpaca_rx) = mpsc::unbounded_channel();
        alpaca_adapter::stream::spawn(ac.clone(), alpaca_tx);
        alpaca_runner::spawn_forwarder(state.clone(), alpaca_rx);
        if let Some(router) = alpaca_router_held {
            alpaca_runner::spawn_account_poller(state.clone(), router, ac.account_poll_ms);
        }
    }

    // M3: feature engine — subscribes to Price, emits Features.
    feature_engine::spawn(state.bus.clone());

    // Bar aggregator — subscribes to Price, emits closed 10s OHLCV bars.
    // Persisted by the writer below; consumed by the new ML pipeline.
    bar_aggregator::spawn(state.bus.clone());

    // M10 — inference hot-swap watcher. The registry was constructed
    // earlier and is already attached to AppState; this watcher does
    // the file-system polling for new champion ONNX files.
    let repo_root = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let (mut hot_swap_rx, _hot_swap_handle) = inference::spawn_hot_swap_watcher(
        repo_root,
        inference_registry.clone(),
    );
    {
        // Forward HotSwapEvent → broadcast bus so the dashboard WS gets
        // ChampionChanged + ChampionLoadFailed banners.
        let bus = state.bus.clone();
        let registry = inference_registry.clone();
        tokio::spawn(async move {
            while let Ok(ev) = hot_swap_rx.recv().await {
                match ev {
                    inference::HotSwapEvent::ChampionChanged { model_id } => {
                        tracing::info!(model_id, "inference: champion changed");
                        if let Err(e) = write_champion_change_jsonl(
                            &model_id,
                            registry.expected_n_features() as i32,
                            "onnx",
                        ) {
                            tracing::warn!(error = %e, "champion_changes.jsonl append failed");
                        }
                        let _ = bus.send(market_domain::Event::ChampionChanged(
                            market_domain::ChampionChanged {
                                model_id,
                                n_features: registry.expected_n_features(),
                                kind: "onnx".to_string(),
                            },
                        ));
                    }
                    inference::HotSwapEvent::ChampionLoadFailed { reason } => {
                        tracing::warn!(reason, "inference: champion load failed");
                        let _ = bus.send(market_domain::Event::ChampionLoadFailed(
                            market_domain::ChampionLoadFailed { reason },
                        ));
                    }
                    inference::HotSwapEvent::Fallback => {
                        tracing::warn!("inference: fell back to neutral predictor");
                        let _ = bus.send(market_domain::Event::ChampionChanged(
                            market_domain::ChampionChanged {
                                model_id: "fallback-neutral".to_string(),
                                n_features: registry.expected_n_features(),
                                kind: "fallback".to_string(),
                            },
                        ));
                    }
                }
            }
        });
    }
    tracing::info!(
        n_features = market_domain::FEATURE_DIM,
        "inference: registry up, hot-swap watcher running"
    );

    // M11.B — live-inference runner: bar-features → registry → ChampionSignal.
    // Subscribes to Event::Bar10s; emits ChampionSignal on each closed bar
    // once the rolling buffer has enough history.
    live_inference::spawn(state.bus.clone(), inference_registry.clone());
    tracing::info!("live-inference: runner spawned");

    // Strategy rewire — live-trader runner. Default OFF; opt in with
    // LIVE_TRADER_ENABLED=true once the champion's ChampionSignal
    // stream looks reasonable in the dashboard. Emits
    // Event::TraderDecision per bar and writes trade_ledger rows on
    // close events. Does NOT route real orders — paper-only until the
    // user wires the ledger to the active OrderRouter.
    let live_trader_enabled = std::env::var("LIVE_TRADER_ENABLED")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false);
    if live_trader_enabled {
        if let Some(ref db) = db_opt {
            let cfg = live_trader::LiveTraderConfig {
                instruments: state.cfg.instruments.clone(),
                run_id: format!(
                    "live_{}",
                    chrono::Utc::now().format("%Y%m%dT%H%M%SZ")
                ),
                reload_interval_secs: std::env::var("LIVE_TRADER_RELOAD_SECS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(600),
            };
            live_trader::spawn(state.bus.clone(), db.clone(), cfg.clone());
            tracing::info!(run_id = %cfg.run_id, "live-trader: runner spawned");

            // Refresh per-ticker summary.json/LATEST.md and the
            // top-level PERFORMANCE.md/portfolio_summary.json from any
            // existing trades.jsonl so the artifacts reflect today's
            // state even if no trade fires immediately. Best-effort —
            // failures are logged at debug; the next live trade will
            // overwrite anyway.
            if let Err(e) = live_trader::refresh_all_ticker_summaries() {
                tracing::debug!(error = %e, "live-trader: ticker rollup refresh failed at boot");
            }
            if let Err(e) = live_trader::refresh_portfolio_summary() {
                tracing::debug!(error = %e, "live-trader: portfolio rollup refresh failed at boot");
            }

            // Order routing is a SECOND opt-in: live-trader emits
            // TraderDecision events regardless, but we only route them
            // through the active OrderRouter when the operator sets
            // CHAMPION_ROUTING_ENABLED=true. Until then the system is
            // research-only — TraderDecision events are visible in the
            // dashboard but no orders flow.
            if api_server::champion_router::routing_enabled_in_env() {
                let cr_cfg = api_server::champion_router::ChampionRouterConfig::from_env();
                api_server::champion_router::spawn(state.clone(), cr_cfg.clone());
                tracing::warn!(
                    run_id = %cr_cfg.run_id,
                    units_per_trade = cr_cfg.units_per_trade,
                    "champion-router: ENABLED — TraderDecision events will route real orders"
                );
            } else {
                tracing::info!(
                    "champion-router: disabled (set CHAMPION_ROUTING_ENABLED=true to opt in)"
                );
            }
        } else {
            tracing::warn!("live-trader: enabled but no DB; skipping");
        }
    } else {
        tracing::info!("live-trader: disabled (set LIVE_TRADER_ENABLED=true to opt in)");
    }

    // === Persistence: spawn writer + rolloff (DB was opened earlier
    // and attached to AppState before Arc-wrapping). ===
    if let Some(ref db) = db_opt {
        let policy = persistence::RetentionPolicy::from_env();
        persistence::spawn_writer(db.clone(), state.bus.clone());
        persistence::spawn_rolloff(db.clone(), policy.clone());
        tracing::info!(?policy, "persistence: writer + rolloff spawned");
    }

    // M4: strategy — subscribes to Features (and Price for mid lookups),
    // trains every TRAIN_AFTER feats, retrains every RETRAIN_EVERY,
    // emits Signal + Fitness. When the DB has prior feature vectors,
    // hydrate the strategy buffers so it can train immediately on
    // restart instead of re-collecting 17 minutes of warmup.
    let strategy_prefill = match &db_opt {
        Some(db) => {
            match db.hydrate_recent_strategy_state(persistence::HYDRATE_FEATURES_PER_INSTRUMENT) {
                Ok(m) => {
                    let total: usize = m.values().map(|v| v.len()).sum();
                    if total > 0 {
                        tracing::info!(
                            total_pairs = total,
                            instruments = m.len(),
                            "strategy: hydrated buffers from db"
                        );
                    }
                    m
                }
                Err(e) => {
                    tracing::warn!("strategy: hydration failed: {e:#}");
                    Default::default()
                }
            }
        }
        None => Default::default(),
    };
    strategy::spawn_with_prefill(state.bus.clone(), strategy_prefill);

    // M5: intent runner — subscribes to Signal, fires the active router,
    // applies fills to the internal paper book, broadcasts PaperFill + PaperBook.
    api_server::commands::spawn_intent_runner(state.clone());
    api_server::commands::spawn_paper_book_ticker(state.clone());

    // === HTTP / WS server ===

    let app = Router::new()
        .route("/healthz", get(http_api::healthz))
        .route("/api/state", get(http_api::current_state))
        .route("/api/instruments", get(http_api::instruments))
        .route("/api/history", get(http_api::history))
        .route("/api/strategy/champion", get(http_api::champion))
        .route("/api/optimizer/trials", get(http_api::optimizer_trials))
        .route("/api/lockbox/result", get(http_api::lockbox_result))
        .route("/api/model/metrics", get(http_api::model_metrics))
        .route("/api/model/candidates", get(http_api::model_candidates))
        .route("/api/model/deployment-gate", get(http_api::model_deployment_gate))
        .route("/api/trader/metrics", get(http_api::trader_metrics))
        .route("/api/labels/recent", get(http_api::labels_recent))
        .route("/api/champion/signals", get(http_api::champion_signals))
        .route("/api/trade/ledger", get(http_api::trade_ledger))
        .route("/api/trade/context", get(http_api::trade_context))
        .route("/api/research/trades", get(http_api::research_trades))
        .route("/api/pipeline/runs", get(http_api::pipeline_runs))
        .route("/api/pipeline/run", axum::routing::post(api_server::pipeline_trigger::pipeline_run))
        .route("/api/pipeline/status", get(api_server::pipeline_trigger::pipeline_status))
        .route("/api/pipeline/auto-retrain", get(api_server::pipeline_trigger::auto_retrain_status))
        .route("/api/pipeline/log", get(api_server::pipeline_trigger::pipeline_log))
        .route("/api/pipeline/last-completed", get(api_server::pipeline_trigger::last_completed_pipeline))
        .route("/ws", get(ws::ws_handler))
        .layer(CorsLayer::very_permissive())
        .with_state(state.clone());

    let listener = tokio::net::TcpListener::bind(&cfg.bind_addr)
        .await
        .with_context(|| format!("binding {}", cfg.bind_addr))?;
    tracing::info!("HTTP/WS listening on http://{}", cfg.bind_addr);

    axum::serve(listener, app)
        .await
        .map_err(|e| anyhow!("axum serve error: {e}"))?;
    Ok(())
}

/// Append a champion-change row to
/// `trade_logs/<v>/champion_changes.jsonl` for the agent-readable
/// preview. Reuses `live_trader::jsonl_log` to keep the 2k-line roll
/// rule consistent across crates.
fn write_champion_change_jsonl(
    new_model_id: &str,
    n_features: i32,
    kind: &str,
) -> anyhow::Result<()> {
    use serde_json::json;
    use std::path::PathBuf;
    let root: PathBuf = std::env::var("TRADE_LOGS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./trade_logs"));
    let version = env!("CARGO_PKG_VERSION");
    let record = json!({
        "v": format!("v{version}"),
        "ts_ms": chrono::Utc::now().timestamp_millis(),
        "new_model_id": new_model_id,
        "kind": kind,
        "n_features": n_features,
    });
    live_trader::jsonl_log::append(&root, version, "champion_changes.jsonl", &record)
}
