//! Alpaca crypto WebSocket data stream.
//!
//! Protocol:
//! 1. Open WS to `wss://stream.data.alpaca.markets/v1beta3/crypto/us`
//! 2. Send: `{"action":"auth","key":"...","secret":"..."}`
//! 3. Receive: `[{"T":"success","msg":"authenticated"}]`
//! 4. Send: `{"action":"subscribe","trades":[...],"quotes":[...],"orderbooks":[...]}`
//! 5. Receive arrays of typed messages forever:
//!    - `T:"q"` quote: `{T,S,t,bp,bs,ap,as,bx,ax}`
//!    - `T:"t"` trade: `{T,S,t,p,s,i,x}`
//!    - `T:"b"` minute bar: `{T,S,t,o,h,l,c,v}`
//!    - `T:"o"` orderbook snapshot: `{T,S,t,b:[[p,s],...],a:[[p,s],...]}`
//!
//! We translate quotes into [`PriceTick`]s (mid = (bp+ap)/2, spread = ap-bp)
//! so the existing pipeline downstream works unchanged. Orderbook snapshots
//! are emitted as [`AlpacaMessage::OrderBook`] for the feature engine to
//! consume.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use futures_util::{SinkExt, StreamExt};
use parking_lot::Mutex;
use serde_json::{json, Value};
use tokio::sync::mpsc::UnboundedSender;
use tokio::time::sleep;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, tungstenite::client::IntoClientRequest};

use market_domain::{OrderBookSide, OrderBookSnapshot, PriceTick};

use crate::config::AlpacaConfig;

/// Per-instrument merged order book. Alpaca sends delta updates rather
/// than full snapshots, so we maintain state and emit a merged snapshot
/// on every update. Sizes use `f64` keys via integer-quantization so we
/// can use BTreeMap for ordered iteration without `Ord` headaches.
#[derive(Debug, Default)]
pub(crate) struct BookState {
    /// price (* 1_000_000 as i64, descending) → size
    bids: BTreeMap<i64, f64>,
    /// price (* 1_000_000 as i64, ascending) → size
    asks: BTreeMap<i64, f64>,
}

const PX_SCALE: f64 = 1_000_000.0;

fn px_key(p: f64) -> i64 {
    (p * PX_SCALE).round() as i64
}

fn key_to_px(k: i64) -> f64 {
    (k as f64) / PX_SCALE
}

impl BookState {
    fn apply(&mut self, side_is_bid: bool, levels: &[(f64, f64)]) {
        let map = if side_is_bid { &mut self.bids } else { &mut self.asks };
        for (p, s) in levels {
            let k = px_key(*p);
            if *s <= 0.0 {
                map.remove(&k);
            } else {
                map.insert(k, *s);
            }
        }
    }

    fn snapshot(&self, instrument: String, time: DateTime<Utc>) -> OrderBookSnapshot {
        // Bids sorted highest-first.
        let bids: Vec<(f64, f64)> = self
            .bids
            .iter()
            .rev()
            .take(20)
            .map(|(k, s)| (key_to_px(*k), *s))
            .collect();
        // Asks sorted lowest-first.
        let asks: Vec<(f64, f64)> = self
            .asks
            .iter()
            .take(20)
            .map(|(k, s)| (key_to_px(*k), *s))
            .collect();
        OrderBookSnapshot {
            instrument,
            time,
            bids: OrderBookSide { levels: bids },
            asks: OrderBookSide { levels: asks },
        }
    }
}

// ---- Public message + types ----

#[derive(Clone, Debug)]
pub enum AlpacaMessage {
    /// Latest top-of-book quote, expressed as a `PriceTick`. We use quotes
    /// rather than trades for the "current price" so we always have a valid
    /// bid/ask spread.
    Price(PriceTick),
    /// 20-level depth snapshot (bid + ask).
    OrderBook(OrderBookSnapshot),
    /// Last trade — useful for diagnostics; not (yet) used by features.
    Trade {
        instrument: String,
        time: DateTime<Utc>,
        price: f64,
        size: f64,
    },
    /// Connection has been authenticated and subscribed.
    Connected,
    /// WS closed / failed; auto-reconnect backoff in progress.
    Disconnected(String),
    Reconnecting {
        attempt: u64,
    },
}

// OrderBookSide / OrderBookSnapshot live in `market-domain` and are
// re-exported by this crate for ergonomics.

// ---- Spawn ----

pub fn spawn(
    cfg: AlpacaConfig,
    sink: UnboundedSender<AlpacaMessage>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move { run(cfg, sink).await })
}

async fn run(cfg: AlpacaConfig, sink: UnboundedSender<AlpacaMessage>) {
    let mut backoff_ms: u64 = 250;
    let max_backoff_ms: u64 = 30_000;
    let mut attempt: u64 = 0;
    // Persistent per-instrument order book across reconnects. We deliberately
    // KEEP state on reconnect — Alpaca will resend deltas; missed levels are
    // backfilled by the next update for that price (or pruned naturally).
    let books: Arc<Mutex<std::collections::HashMap<String, BookState>>> =
        Arc::new(Mutex::new(std::collections::HashMap::new()));
    loop {
        if sink.is_closed() {
            tracing::info!("alpaca: sink closed; exiting");
            return;
        }
        match connect_once(&cfg, &sink, &books).await {
            Ok(()) => {
                tracing::info!("alpaca: stream ended cleanly; reconnecting");
                backoff_ms = 250;
            }
            Err(e) => {
                tracing::warn!("alpaca: connect failed: {e:#}");
                let _ = sink.send(AlpacaMessage::Disconnected(format!("{e:#}")));
            }
        }
        attempt = attempt.saturating_add(1);
        let _ = sink.send(AlpacaMessage::Reconnecting { attempt });
        sleep(Duration::from_millis(backoff_ms)).await;
        backoff_ms = (backoff_ms.saturating_mul(2)).min(max_backoff_ms);
    }
}

async fn connect_once(
    cfg: &AlpacaConfig,
    sink: &UnboundedSender<AlpacaMessage>,
    books: &Arc<Mutex<std::collections::HashMap<String, BookState>>>,
) -> anyhow::Result<()> {
    tracing::info!(url = %cfg.data_ws_url, symbols = ?cfg.symbols, "alpaca: opening WS");
    let req = cfg.data_ws_url.as_str().into_client_request()?;
    let (mut ws, _resp) = connect_async(req).await?;

    // Step 1: auth
    let auth = json!({"action":"auth","key":cfg.key,"secret":cfg.secret});
    ws.send(Message::Text(auth.to_string())).await?;

    // Step 2: read until we see {T:"success",msg:"authenticated"}.
    let mut authenticated = false;
    while let Some(msg) = ws.next().await {
        let msg = msg?;
        match msg {
            Message::Text(t) => {
                let v: Value = serde_json::from_str(&t)?;
                if let Some(arr) = v.as_array() {
                    for item in arr {
                        let t_kind = item.get("T").and_then(|x| x.as_str()).unwrap_or("");
                        let msg_str = item.get("msg").and_then(|x| x.as_str()).unwrap_or("");
                        if t_kind == "success" && msg_str == "authenticated" {
                            authenticated = true;
                            break;
                        }
                        if t_kind == "error" {
                            return Err(anyhow::anyhow!("alpaca auth error: {item}"));
                        }
                    }
                }
                if authenticated {
                    break;
                }
            }
            Message::Ping(p) => {
                ws.send(Message::Pong(p)).await?;
            }
            Message::Close(_) => return Err(anyhow::anyhow!("alpaca closed during auth")),
            _ => {}
        }
    }
    if !authenticated {
        return Err(anyhow::anyhow!("alpaca: no auth ack received"));
    }

    // Step 3: subscribe
    let sub = json!({
        "action":"subscribe",
        "trades": cfg.symbols,
        "quotes": cfg.symbols,
        "orderbooks": cfg.symbols,
    });
    ws.send(Message::Text(sub.to_string())).await?;
    let _ = sink.send(AlpacaMessage::Connected);
    tracing::info!("alpaca: authenticated + subscribed");

    // Step 4: forward forever.
    while let Some(msg) = ws.next().await {
        if sink.is_closed() {
            return Ok(());
        }
        let msg = match msg {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!("alpaca: ws error: {e:#}");
                return Err(e.into());
            }
        };
        match msg {
            Message::Text(t) => handle_text(&t, sink, books),
            Message::Binary(b) => {
                // Alpaca occasionally sends binary frames with the same
                // JSON payload — try to decode.
                if let Ok(s) = std::str::from_utf8(&b) {
                    handle_text(s, sink, books);
                }
            }
            Message::Ping(p) => {
                ws.send(Message::Pong(p)).await?;
            }
            Message::Close(_) => return Ok(()),
            _ => {}
        }
    }
    Ok(())
}

fn handle_text(
    text: &str,
    sink: &UnboundedSender<AlpacaMessage>,
    books: &Arc<Mutex<std::collections::HashMap<String, BookState>>>,
) {
    let v: Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(e) => {
            tracing::debug!("alpaca: bad JSON: {e}; raw={text:.200}");
            return;
        }
    };
    let arr = match v.as_array() {
        Some(a) => a,
        None => return,
    };
    for item in arr {
        let kind = item.get("T").and_then(|x| x.as_str()).unwrap_or("");
        match kind {
            "q" => {
                if let Some(t) = parse_quote(item) {
                    let _ = sink.send(AlpacaMessage::Price(t));
                }
            }
            "t" => {
                if let Some(t) = parse_trade(item) {
                    let _ = sink.send(t);
                }
            }
            "o" => {
                if let Some((instrument, time, bid_levels, ask_levels)) =
                    parse_orderbook_delta(item)
                {
                    let snap = {
                        let mut map = books.lock();
                        let book = map.entry(instrument.clone()).or_default();
                        book.apply(true, &bid_levels);
                        book.apply(false, &ask_levels);
                        book.snapshot(instrument, time)
                    };
                    // Skip emission until both sides have at least one level
                    // (avoids confusing the feature engine with half-empty books).
                    if !snap.bids.levels.is_empty() && !snap.asks.levels.is_empty() {
                        let _ = sink.send(AlpacaMessage::OrderBook(snap));
                    }
                }
            }
            "b" => { /* minute bar — not used yet */ }
            "subscription" | "success" => { /* control */ }
            "error" => {
                tracing::warn!("alpaca: error frame: {item}");
            }
            _ => {
                tracing::trace!(%kind, "alpaca: unknown frame");
            }
        }
    }
}

fn parse_quote(v: &Value) -> Option<PriceTick> {
    let instrument = v.get("S")?.as_str()?.to_string();
    let time_str = v.get("t")?.as_str()?;
    let time = DateTime::parse_from_rfc3339(time_str)
        .ok()?
        .with_timezone(&Utc);
    let bp = v.get("bp")?.as_f64()?;
    let ap = v.get("ap")?.as_f64()?;
    if bp <= 0.0 || ap <= 0.0 {
        return None;
    }
    let mid = (bp + ap) / 2.0;
    let spread = (ap - bp).max(0.0);
    Some(PriceTick {
        instrument,
        time,
        bid: bp,
        ask: ap,
        mid,
        spread,
        closeout_bid: None,
        closeout_ask: None,
        status: Some("tradeable".into()),
    })
}

fn parse_trade(v: &Value) -> Option<AlpacaMessage> {
    let instrument = v.get("S")?.as_str()?.to_string();
    let time_str = v.get("t")?.as_str()?;
    let time = DateTime::parse_from_rfc3339(time_str)
        .ok()?
        .with_timezone(&Utc);
    let price = v.get("p")?.as_f64()?;
    let size = v.get("s").and_then(|x| x.as_f64()).unwrap_or(0.0);
    Some(AlpacaMessage::Trade { instrument, time, price, size })
}

/// Parse a `T:"o"` frame into (instrument, time, bid_levels, ask_levels).
/// Returns `None` only if the frame is structurally malformed; an empty
/// bid or ask array is fine — those are deltas applied to BookState.
fn parse_orderbook_delta(v: &Value) -> Option<(String, DateTime<Utc>, Vec<(f64, f64)>, Vec<(f64, f64)>)> {
    let instrument = v.get("S")?.as_str()?.to_string();
    let time_str = v.get("t")?.as_str()?;
    let time = DateTime::parse_from_rfc3339(time_str)
        .ok()?
        .with_timezone(&Utc);
    let bid_levels = v.get("b").and_then(parse_levels).unwrap_or_default();
    let ask_levels = v.get("a").and_then(parse_levels).unwrap_or_default();
    Some((instrument, time, bid_levels, ask_levels))
}

fn parse_levels(v: &Value) -> Option<Vec<(f64, f64)>> {
    let arr = v.as_array()?;
    let mut out = Vec::with_capacity(arr.len());
    for level in arr {
        // Alpaca format: {p:price, s:size} per level (object, not array)
        // — verify against live data; some endpoints use [[p,s]] arrays.
        if let Some(obj) = level.as_object() {
            let p = obj.get("p").and_then(|x| x.as_f64())?;
            let s = obj.get("s").and_then(|x| x.as_f64())?;
            out.push((p, s));
        } else if let Some(pair) = level.as_array() {
            if pair.len() >= 2 {
                let p = pair[0].as_f64()?;
                let s = pair[1].as_f64()?;
                out.push((p, s));
            }
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_quote_ok() {
        let raw = serde_json::json!({
            "T":"q","S":"BTC/USD",
            "t":"2026-04-25T09:00:00.123456789Z",
            "bp":50000.5,"bs":0.4,"ap":50001.5,"as":0.7,
            "bx":"CBSE","ax":"CBSE"
        });
        let t = parse_quote(&raw).expect("should parse");
        assert_eq!(t.instrument, "BTC/USD");
        assert!((t.mid - 50001.0).abs() < 1e-9);
        assert!((t.spread - 1.0).abs() < 1e-9);
    }

    #[test]
    fn parse_orderbook_delta_object_levels() {
        let raw = serde_json::json!({
            "T":"o","S":"BTC/USD",
            "t":"2026-04-25T09:00:00Z",
            "b":[{"p":50000,"s":0.5},{"p":49999,"s":1.0}],
            "a":[{"p":50001,"s":0.4},{"p":50002,"s":0.8}]
        });
        let (inst, _, bids, asks) = parse_orderbook_delta(&raw).expect("should parse");
        assert_eq!(inst, "BTC/USD");
        assert_eq!(bids.len(), 2);
        assert_eq!(asks.len(), 2);
        assert_eq!(bids[0], (50000.0, 0.5));
    }

    #[test]
    fn book_state_applies_deltas_and_truncates() {
        let mut bs = BookState::default();
        // First delta: insert two bids and two asks.
        bs.apply(true, &[(50000.0, 0.5), (49999.0, 1.0)]);
        bs.apply(false, &[(50001.0, 0.4), (50002.0, 0.8)]);
        // Second delta: update size at 50000.0; remove 49999.0 (size=0).
        bs.apply(true, &[(50000.0, 1.5), (49999.0, 0.0)]);
        // Third delta: insert a new best bid at 50000.5.
        bs.apply(true, &[(50000.5, 2.0)]);
        let snap = bs.snapshot("BTC/USD".into(), Utc::now());
        assert_eq!(snap.bids.levels[0], (50000.5, 2.0), "best bid should be 50000.5");
        assert_eq!(snap.bids.levels[1], (50000.0, 1.5), "second-best bid 50000.0@1.5");
        // 49999.0 should be gone (size 0 deletes it).
        assert_eq!(snap.bids.levels.len(), 2);
        assert_eq!(snap.asks.levels[0], (50001.0, 0.4));
    }
}
