//! Replay real recorded OANDA pricing data from JSONL files.
//!
//! Reads `pricing_1s-*.jsonl` files written by the Python `oanda_collector.py`
//! pipeline at the repo root. Each line is one tick:
//!
//! ```json
//! {"kind":"pricing_1s","ts":"2026-02-19T20:54:53Z","instrument":"EUR_USD",
//!  "source_time":"2026-02-19T20:54:52Z","bid":"1.17678","ask":"1.17695",
//!  "closeoutBid":"1.17669","closeoutAsk":"1.17705","status":"tradeable"}
//! ```
//!
//! Replays into the same `PricingMessage` channel as the live OANDA stream
//! so the rest of the pipeline (forwarders, feature engine, strategy)
//! cannot tell the difference.
//!
//! **This is real OANDA data**, just from a previous trading session.
//! Used when forex markets are closed (weekends) or for offline development.
//!
//! Enable via:
//! ```bash
//! OANDA_REPLAY=oanda_data/20260219_205451 OANDA_REPLAY_SPEED=100 cargo run -p api-server
//! ```

use std::path::{Path, PathBuf};
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde_json::Value;
use tokio::fs::File;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc::UnboundedSender;
use tokio::time::sleep;

use market_domain::PriceTick;

use crate::streams::pricing::PricingMessage;

/// Spawn a replay task. Reads `pricing_1s-*.jsonl` files in `dir` in
/// timestamp order, emits one `PricingMessage::Price` per row at
/// `speed`× real time. Loops forever.
pub fn spawn(
    dir: PathBuf,
    sink: UnboundedSender<PricingMessage>,
    speed: f64,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move { run(dir, sink, speed).await })
}

async fn run(dir: PathBuf, sink: UnboundedSender<PricingMessage>, speed: f64) {
    // Sane bounds: speed ∈ (0, 5000). Above 5000 the wall-clock pacing
    // becomes negligible — emits as fast as the broadcast bus drains.
    let speed = speed.clamp(0.001, 5000.0);
    let files = match collect_pricing_files(&dir).await {
        Ok(f) if !f.is_empty() => f,
        Ok(_) => {
            tracing::error!(
                "OANDA_REPLAY={dir:?}: no pricing_1s-*.jsonl files found",
                dir = dir
            );
            return;
        }
        Err(e) => {
            tracing::error!("OANDA_REPLAY={dir:?}: {e}", dir = dir);
            return;
        }
    };

    let _ = sink.send(PricingMessage::Connected);
    tracing::warn!(
        files = files.len(),
        speed,
        dir = %dir.display(),
        "REPLAY mode: streaming real recorded OANDA data (DEMO/OFFLINE)"
    );

    let mut total_emitted: u64 = 0;
    'outer: loop {
        // Monotonic source-time used for pacing (collector `ts` field —
        // zigzag-free, unlike per-instrument `source_time`).
        let mut last_pacing_ts: Option<DateTime<Utc>> = None;
        for path in &files {
            tracing::info!(path = %path.display(), "replay: opening file");
            let f = match File::open(&path).await {
                Ok(f) => f,
                Err(e) => {
                    tracing::warn!(path = %path.display(), "replay: open failed: {e}");
                    continue;
                }
            };
            let mut reader = BufReader::new(f).lines();
            while let Some(line) = reader.next_line().await.transpose() {
                if sink.is_closed() {
                    tracing::info!("replay: sink closed; exiting");
                    return;
                }
                let line = match line {
                    Ok(l) => l,
                    Err(e) => {
                        tracing::warn!("replay: read error: {e}");
                        continue;
                    }
                };
                if line.is_empty() {
                    continue;
                }
                let v: Value = match serde_json::from_str(&line) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let kind = v.get("kind").and_then(|x| x.as_str()).unwrap_or("");
                if kind != "pricing_1s" {
                    continue;
                }
                let tick = match parse_recorded_tick(&v) {
                    Some(t) => t,
                    None => continue,
                };
                // For pacing we use the collector `ts` field (monotonic
                // wall-clock at write time), not the per-instrument
                // `source_time` (which zigzags between EUR_USD and USD_JPY
                // rows). Falls back to `source_time` if `ts` missing.
                let pacing_ts: DateTime<Utc> = v
                    .get("ts")
                    .and_then(|x| x.as_str())
                    .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                    .map(|d| d.with_timezone(&Utc))
                    .unwrap_or(tick.time);

                if let Some(prev) = last_pacing_ts {
                    let dt_ms = (pacing_ts - prev).num_milliseconds().max(0) as f64;
                    let scaled_ms = (dt_ms / speed).clamp(0.0, 5_000.0);
                    if scaled_ms >= 0.2 {
                        sleep(Duration::from_micros((scaled_ms * 1_000.0) as u64)).await;
                    }
                }
                last_pacing_ts = Some(pacing_ts);
                if let Err(_) = sink.send(PricingMessage::Price(tick)) {
                    tracing::info!("replay: receiver dropped");
                    return;
                }
                total_emitted = total_emitted.saturating_add(1);
                if total_emitted % 5_000 == 0 {
                    tracing::info!(
                        emitted = total_emitted,
                        "replay: progress"
                    );
                }
            }
        }
        tracing::info!(
            emitted = total_emitted,
            "replay: end of dataset reached; looping"
        );
        if sink.is_closed() {
            break 'outer;
        }
    }
}

async fn collect_pricing_files(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut rd = tokio::fs::read_dir(dir).await?;
    while let Some(entry) = rd.next_entry().await? {
        let p = entry.path();
        if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
            if name.starts_with("pricing_1s-") && name.ends_with(".jsonl") {
                out.push(p);
            }
        }
    }
    out.sort();
    Ok(out)
}

fn parse_recorded_tick(v: &Value) -> Option<PriceTick> {
    let instrument = v.get("instrument")?.as_str()?.to_string();
    let bid: f64 = v.get("bid")?.as_str()?.parse().ok()?;
    let ask: f64 = v.get("ask")?.as_str()?.parse().ok()?;
    let closeout_bid = v
        .get("closeoutBid")
        .and_then(|x| x.as_str())
        .and_then(|s| s.parse().ok());
    let closeout_ask = v
        .get("closeoutAsk")
        .and_then(|x| x.as_str())
        .and_then(|s| s.parse().ok());
    let status = v
        .get("status")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string());
    // Prefer source_time (OANDA's wall clock at quote time); fall back to
    // the collector's `ts` field.
    let time_str = v
        .get("source_time")
        .and_then(|x| x.as_str())
        .or_else(|| v.get("ts").and_then(|x| x.as_str()))?;
    let time: DateTime<Utc> =
        DateTime::parse_from_rfc3339(time_str).ok()?.with_timezone(&Utc);

    let mid = (bid + ask) / 2.0;
    let spread = ask - bid;
    Some(PriceTick {
        instrument,
        time,
        bid,
        ask,
        mid,
        spread,
        closeout_bid,
        closeout_ask,
        status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_recorded_handles_real_row() {
        let raw = serde_json::json!({
            "kind": "pricing_1s",
            "ts": "2026-02-19T20:54:53.001054+00:00",
            "instrument": "EUR_USD",
            "source_time": "2026-02-19T20:54:52.533470375Z",
            "bid": "1.17678",
            "ask": "1.17695",
            "closeoutBid": "1.17669",
            "closeoutAsk": "1.17705",
            "status": "tradeable"
        });
        let t = parse_recorded_tick(&raw).expect("should parse");
        assert_eq!(t.instrument, "EUR_USD");
        assert!((t.bid - 1.17678).abs() < 1e-9);
        assert!((t.ask - 1.17695).abs() < 1e-9);
        assert!((t.spread - 0.00017).abs() < 1e-9);
        assert_eq!(t.status.as_deref(), Some("tradeable"));
    }

    #[test]
    fn parse_recorded_skips_unparseable_bid() {
        let raw = serde_json::json!({
            "kind": "pricing_1s",
            "instrument": "EUR_USD",
            "source_time": "2026-02-19T20:54:52Z",
            "bid": "x",
            "ask": "1.17695"
        });
        assert!(parse_recorded_tick(&raw).is_none());
    }
}
