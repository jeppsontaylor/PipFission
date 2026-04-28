//! Writer task. Subscribes to the broadcast bus, batches inserts, and
//! flushes them to DuckDB in transactions of up to 200 jobs or 200ms,
//! whichever comes first. Same shape as the previous SQLite writer; the
//! only differences are the DuckDB API surface and the BIGINT-only time
//! columns.

use std::time::Duration;

use anyhow::Result;
use duckdb::params;
use tokio::sync::broadcast;
use tokio::sync::mpsc::{self, UnboundedReceiver};

use market_domain::{
    AccountSnapshot, Bar10sNamed, ChampionSignal, Event, ModelFitness, PaperFillEvent, PriceTick,
    StrategySignal,
};

use crate::connection::Db;

/// Spawn a task that subscribes to the bus and persists every relevant
/// event. Returns immediately. The task ends when the bus closes.
pub fn spawn_writer(db: Db, bus: broadcast::Sender<Event>) -> tokio::task::JoinHandle<()> {
    let (tx, rx) = mpsc::unbounded_channel::<WriteJob>();
    let writer_db = db.clone();
    tokio::task::spawn_blocking(move || {
        if let Err(e) = writer_loop(writer_db, rx) {
            tracing::error!("persistence writer fatal: {e:#}");
        }
    });
    tokio::spawn(async move {
        let mut rx = bus.subscribe();
        loop {
            match rx.recv().await {
                Ok(event) => {
                    if let Some(job) = WriteJob::from_event(&event) {
                        if tx.send(job).is_err() {
                            tracing::warn!(
                                "persistence: writer dropped — stopping subscriber"
                            );
                            return;
                        }
                    }
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!(
                        "persistence: subscriber lagged {n} events (some not persisted)"
                    );
                }
                Err(broadcast::error::RecvError::Closed) => {
                    return;
                }
            }
        }
    })
}

#[derive(Debug)]
enum WriteJob {
    Price(Box<PriceTick>),
    Bar(Box<Bar10sNamed>),
    Signal(Box<StrategySignal>),
    Champion(Box<ChampionSignal>),
    Fill(Box<PaperFillEvent>),
    Fitness(Box<ModelFitness>),
    Account(Box<AccountSnapshot>),
}

impl WriteJob {
    fn from_event(e: &Event) -> Option<Self> {
        Some(match e {
            Event::Price(p) => WriteJob::Price(Box::new(p.clone())),
            Event::Bar10s(b) => WriteJob::Bar(Box::new(b.clone())),
            Event::Signal(s) => WriteJob::Signal(Box::new(s.clone())),
            Event::ChampionSignal(c) => WriteJob::Champion(Box::new(c.clone())),
            Event::PaperFill(f) => WriteJob::Fill(Box::new(f.clone())),
            Event::Fitness(f) => WriteJob::Fitness(Box::new(f.clone())),
            Event::Account(a) => WriteJob::Account(Box::new(a.clone())),
            _ => return None,
        })
    }
}

/// Synchronous writer running on a blocking thread. Batches inserts in
/// transactions of up to 200 jobs or 200ms — whichever comes first — to
/// keep transaction overhead amortised.
fn writer_loop(db: Db, mut rx: UnboundedReceiver<WriteJob>) -> Result<()> {
    use std::time::Instant;
    const BATCH_LIMIT: usize = 200;
    const MAX_WAIT: Duration = Duration::from_millis(200);

    let mut buf: Vec<WriteJob> = Vec::with_capacity(BATCH_LIMIT);
    let mut last_flush = Instant::now();

    loop {
        let job = match rx.try_recv() {
            Ok(j) => Some(j),
            Err(mpsc::error::TryRecvError::Empty) => {
                if !buf.is_empty() && last_flush.elapsed() >= MAX_WAIT {
                    flush(&db, &mut buf)?;
                    last_flush = Instant::now();
                }
                match rx.blocking_recv() {
                    Some(j) => Some(j),
                    None => break,
                }
            }
            Err(mpsc::error::TryRecvError::Disconnected) => break,
        };
        if let Some(j) = job {
            buf.push(j);
            if buf.len() >= BATCH_LIMIT {
                flush(&db, &mut buf)?;
                last_flush = Instant::now();
            }
        }
    }
    if !buf.is_empty() {
        flush(&db, &mut buf)?;
    }
    Ok(())
}

fn flush(db: &Db, buf: &mut Vec<WriteJob>) -> Result<()> {
    if buf.is_empty() {
        return Ok(());
    }
    let mut conn = db.inner.lock();
    let tx = conn.transaction()?;
    {
        for job in buf.drain(..) {
            match job {
                WriteJob::Price(p) => {
                    tx.execute(
                        "INSERT INTO price_ticks
                         (instrument, ts_ms, bid, ask, mid, spread_bp, status)
                         VALUES (?, ?, ?, ?, ?, ?, ?)",
                        params![
                            p.instrument,
                            p.time.timestamp_millis(),
                            p.bid,
                            p.ask,
                            p.mid,
                            // Spread is reported in price units; convert to basis points
                            // relative to mid. Defensive: avoid div-by-zero on a stale tick.
                            if p.mid > 0.0 {
                                p.spread / p.mid * 10_000.0
                            } else {
                                0.0
                            },
                            p.status,
                        ],
                    )?;
                }
                WriteJob::Bar(b) => {
                    tx.execute(
                        "INSERT INTO bars_10s
                         (instrument, ts_ms, open, high, low, close, n_ticks, spread_bp_avg)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        params![
                            b.instrument,
                            b.ts_ms,
                            b.open,
                            b.high,
                            b.low,
                            b.close,
                            b.n_ticks as i64,
                            b.spread_bp_avg,
                        ],
                    )?;
                }
                WriteJob::Signal(s) => {
                    let dir = match s.direction {
                        market_domain::SignalDirection::Long => "long",
                        market_domain::SignalDirection::Flat => "flat",
                        market_domain::SignalDirection::Short => "short",
                    };
                    tx.execute(
                        "INSERT INTO signals
                         (instrument, ts_ms, direction, confidence,
                          prob_long, prob_flat, prob_short, prob_take,
                          model_id, model_version)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        params![
                            s.instrument,
                            s.time.timestamp_millis(),
                            dir,
                            s.confidence,
                            s.probs[0],
                            s.probs[1],
                            s.probs[2],
                            // prob_take defaults to confidence until the meta-model
                            // is wired through; that's what the existing logreg
                            // exports as its uncalibrated take-trade signal.
                            s.confidence,
                            s.model_id,
                            s.model_version as i64,
                        ],
                    )?;
                }
                WriteJob::Champion(c) => {
                    tx.execute(
                        "INSERT INTO champion_signals
                         (instrument, ts_ms, p_long, p_short, p_take, calibrated, model_id, kind)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        params![
                            c.instrument,
                            c.time.timestamp_millis(),
                            c.p_long,
                            c.p_short,
                            c.p_take,
                            c.calibrated,
                            c.model_id,
                            c.kind,
                        ],
                    )?;
                }
                WriteJob::Fill(f) => {
                    let mode = match f.mode {
                        market_domain::RoutingMode::Internal => "internal",
                        market_domain::RoutingMode::OandaPractice => "oanda_practice",
                    };
                    // INSERT OR IGNORE equivalent in DuckDB is ON CONFLICT DO NOTHING.
                    tx.execute(
                        "INSERT INTO paper_fills
                         (order_id, instrument, ts_ms, units, price, fee, mode)
                         VALUES (?, ?, ?, ?, ?, ?, ?)
                         ON CONFLICT(order_id) DO NOTHING",
                        params![
                            f.order_id,
                            f.instrument,
                            f.time.timestamp_millis(),
                            f.units,
                            f.price,
                            f.fee,
                            mode,
                        ],
                    )?;
                }
                WriteJob::Fitness(f) => {
                    tx.execute(
                        "INSERT INTO fitness
                         (instrument, model_id, model_version, trained_at_ms,
                          train_samples, train_accuracy, train_log_loss, train_sharpe,
                          train_dist_long, train_dist_flat, train_dist_short,
                          oos_samples, oos_accuracy, oos_log_loss, oos_sharpe,
                          oos_dist_long, oos_dist_flat, oos_dist_short,
                          samples_seen, train_window_lo, train_window_hi,
                          oos_window_lo, oos_window_hi)
                         VALUES (?, ?, ?, ?,
                                 ?, ?, ?, ?,
                                 ?, ?, ?,
                                 ?, ?, ?, ?,
                                 ?, ?, ?,
                                 ?, ?, ?,
                                 ?, ?)
                         ON CONFLICT(model_id, model_version) DO NOTHING",
                        params![
                            f.instrument,
                            f.model_id,
                            f.model_version as i64,
                            f.trained_at.timestamp_millis(),
                            f.train.samples as i64,
                            f.train.accuracy,
                            f.train.log_loss,
                            f.train.sharpe,
                            f.train.class_distribution[0] as i64,
                            f.train.class_distribution[1] as i64,
                            f.train.class_distribution[2] as i64,
                            f.oos.samples as i64,
                            f.oos.accuracy,
                            f.oos.log_loss,
                            f.oos.sharpe,
                            f.oos.class_distribution[0] as i64,
                            f.oos.class_distribution[1] as i64,
                            f.oos.class_distribution[2] as i64,
                            f.samples_seen as i64,
                            f.train_window.0 as i64,
                            f.train_window.1 as i64,
                            f.oos_window.0 as i64,
                            f.oos_window.1 as i64,
                        ],
                    )?;
                }
                WriteJob::Account(a) => {
                    tx.execute(
                        "INSERT INTO account_snapshots
                         (ts_ms, nav, balance, unrealized_pl, realized_pl,
                          margin_used, margin_avail, currency)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        params![
                            a.time.timestamp_millis(),
                            a.nav,
                            a.balance,
                            a.unrealized_pl,
                            a.realized_pl,
                            a.margin_used,
                            a.margin_available,
                            a.currency,
                        ],
                    )?;
                }
            }
        }
    }
    tx.commit()?;
    Ok(())
}
