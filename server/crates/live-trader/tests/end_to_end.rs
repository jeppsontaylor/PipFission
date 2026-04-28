//! Integration test: feed Bar10s + ChampionSignal events for 200 bars
//! and assert:
//!   * Stream of TraderDecision events appears on the bus.
//!   * Skip events dominate when the model has no edge (uniform 0.5
//!     probabilities); a few open events appear when probs are
//!     skewed; close events appear after stop/take-profit/max-hold.
//!   * trade_ledger gains rows on close events.
//!
//! Uses a temp DuckDB seeded with one trader_metrics row so the runner
//! has a TraderParams to consume.

use std::path::PathBuf;
use std::time::Duration;

use chrono::TimeZone;
use duckdb::params;
use market_domain::{Bar10sNamed, ChampionSignal, Event};
use persistence::Db;
use tokio::sync::broadcast;
use trader::TraderParams;

fn tmp_db_path(label: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let pid = std::process::id();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    p.push(format!("live_trader_test_{label}_{pid}_{ts}.duckdb"));
    p
}

fn seed_trader_metrics(db: &Db, params_id: &str, model_id: &str) {
    // Force the trader to actually open positions on the synthetic
    // upward drift used below: tight thresholds, short hold, generous
    // stops, immediate cooldown clear.
    let params = TraderParams {
        long_threshold: 0.55,
        short_threshold: 0.55,
        take_threshold: 0.50,
        min_conf_margin: 0.05,
        stop_loss_atr: 5.0,
        take_profit_atr: 0.05, // small so the rising series triggers TP fast
        trailing_stop_atr: 0.0,
        min_hold_bars: 1,
        max_hold_bars: 30,
        cooldown_bars: 1,
        max_position_frac: 0.10,
        daily_loss_limit_bp: 5000.0,
        max_dd_pause_bp: 50_000.0,
        spread_max_bp: 50.0,
        stale_data_ms: 1_000_000,
    };
    let json = serde_json::to_string(&params).unwrap();
    db.with_conn(|conn| {
        conn.execute(
            "INSERT INTO trader_metrics
             (params_id, model_id, ts_ms, in_sample_sharpe, in_sample_sortino,
              fine_tune_sharpe, fine_tune_sortino, max_dd_bp, turnover_per_day,
              hit_rate, profit_factor, n_trades, params_json)
             VALUES (?, ?, ?, 0, 0, 0, 0, 0, 0, 0, 0, 0, ?)",
            params![params_id, model_id, 1_700_000_000_000_i64, json],
        )
        .unwrap();
    });
}

fn make_bar(instrument: &str, ts_ms: i64, close: f64) -> Bar10sNamed {
    Bar10sNamed {
        instrument: instrument.to_string(),
        ts_ms,
        open: close,
        high: close + 1e-4,
        low: close - 1e-4,
        close,
        n_ticks: 1,
        spread_bp_avg: 0.5,
    }
}

fn make_signal(instrument: &str, ts_ms: i64, p_long: f64) -> ChampionSignal {
    let p_short = (1.0 - p_long).clamp(0.0, 1.0);
    let p_take = p_long.max(p_short);
    let time = chrono::Utc
        .timestamp_millis_opt(ts_ms)
        .single()
        .unwrap();
    ChampionSignal {
        instrument: instrument.to_string(),
        time,
        p_long,
        p_short,
        p_take,
        calibrated: p_take,
        model_id: "test_model".to_string(),
        kind: "onnx".to_string(),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn live_trader_emits_decisions_and_records_trades() {
    let path = tmp_db_path("e2e");
    let _ = std::fs::remove_file(&path);
    let db = Db::open(&path).expect("open db");
    seed_trader_metrics(&db, "params_e2e", "model_e2e");

    let (bus, _initial) = broadcast::channel::<Event>(512);
    let mut sink = bus.subscribe();

    let cfg = live_trader::LiveTraderConfig {
        instruments: vec!["TEST_PAIR".to_string()],
        run_id: "test_run".to_string(),
        reload_interval_secs: 3600,
    };
    let _join = live_trader::spawn(bus.clone(), db.clone(), cfg);

    // Pump 60 bars first (warmup), then pump 100 bars with strongly
    // positive p_long so the trader opens a long and rides it. Each
    // (bar, signal) pair is sent strictly in order so the runner
    // observes the bar before the matching signal.
    let inst = "TEST_PAIR";
    for i in 0..160_i64 {
        let close = 1.0 + (i as f64) * 0.0005;
        let _ = bus.send(Event::Bar10s(make_bar(inst, i * 10_000, close)));
        let p_long = if i < 60 { 0.5 } else { 0.85 };
        let _ = bus.send(Event::ChampionSignal(make_signal(inst, i * 10_000, p_long)));
        // Yield so the runner can process before the next pair.
        tokio::task::yield_now().await;
    }

    // Drain decision events for up to 4 seconds; require we've seen
    // at least one open and one close.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(4);
    let mut saw_open = false;
    let mut saw_close = false;
    let mut saw_skip = false;
    while tokio::time::Instant::now() < deadline {
        let recv = tokio::time::timeout(Duration::from_millis(50), sink.recv()).await;
        match recv {
            Ok(Ok(Event::TraderDecision(d))) => {
                match d.action.as_str() {
                    "open_long" | "open_short" => saw_open = true,
                    "close" => saw_close = true,
                    "skip" => saw_skip = true,
                    _ => {}
                }
                if saw_open && saw_close && saw_skip {
                    break;
                }
            }
            Ok(Ok(_)) => continue,
            Ok(Err(_)) => break,
            Err(_) => continue, // poll timeout
        }
    }
    assert!(saw_skip, "expected skip events");
    assert!(saw_open, "expected at least one open event");
    assert!(saw_close, "expected at least one close event");

    let n_ledger: i64 = db.with_conn(|conn| {
        conn.query_row(
            "SELECT COUNT(*) FROM trade_ledger WHERE run_id = ?",
            params!["test_run"],
            |r| r.get(0),
        )
        .unwrap()
    });
    assert!(
        n_ledger >= 1,
        "expected >= 1 trade_ledger row, got {n_ledger}"
    );

    let _ = std::fs::remove_file(&path);
}
