//! AppState: live snapshot the dashboard cares about, plus a broadcast bus
//! that fans out delta events. Wire types live in `market-domain`; this
//! crate is the only place those types meet `tokio::broadcast` and
//! `dashmap`.

use std::collections::VecDeque;
use std::sync::Arc;

use arc_swap::ArcSwap;
use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};
use tokio::sync::broadcast;

use market_domain::{
    AccountSnapshot, AlpacaAccountSnapshot, ConnStatus, EstimateTick, Event, History, PriceTick,
    RoutingMode, Snapshot, TransactionEvent, BROADCAST_CAPACITY, TRANSACTION_LIMIT,
};
use portfolio::{OrderRouter, PaperBook, PaperRouter, PriceOracle};

use crate::auto_retrain::AutoRetrain;
use crate::pipeline_trigger::{PipelineFlight, PipelineFlightSlot};

pub struct AppState {
    pub cfg: market_domain::Config,
    pub account_id: String,

    /// Latest tick per instrument.
    pub latest_prices: DashMap<String, PriceTick>,
    /// Per-instrument mid price history.
    pub price_history: DashMap<String, History>,

    /// Latest account summary.
    pub account: RwLock<Option<AccountSnapshot>>,
    /// NAV history (1 sample per account poll).
    pub nav_history: RwLock<History>,

    /// Latest estimate tick + history.
    pub estimate: RwLock<Option<EstimateTick>>,
    pub estimate_history: RwLock<History>,

    /// Recent transaction events.
    pub transactions: RwLock<VecDeque<TransactionEvent>>,

    /// Stream health.
    pub conn: RwLock<ConnStatus>,

    /// Broadcast channel for live events.
    pub bus: broadcast::Sender<Event>,

    // === Sprint 2 additions ===
    /// Current routing mode. Atomic read; swapped via SetMode.
    pub mode: ArcSwap<RoutingMode>,

    /// Internal paper book (separate from the OANDA-actual reconciler that
    /// already lives in `estimator.rs`). Strategy fills land here.
    pub paper_book: Arc<Mutex<PaperBook>>,

    /// Monotonic version stamp for PaperBook snapshots so the dashboard
    /// can detect missed updates after a Lagged resync.
    pub paper_book_version: std::sync::atomic::AtomicU64,

    /// Latest best bid/ask per instrument, mirrored from `latest_prices`
    /// so the paper router doesn't reach into AppState.
    pub price_oracle: Arc<PriceOracle>,

    /// Active order router. `ArcSwap<dyn OrderRouter>` so the mode toggle
    /// is a lock-free pointer swap between ticks. Default is PaperRouter.
    pub router: ArcSwap<Arc<dyn OrderRouter>>,

    /// PaperRouter pointer (held so we can swap back from OANDA -> Internal).
    pub paper_router: Arc<Arc<dyn OrderRouter>>,

    /// OandaRouter pointer (set by main on startup if the OANDA client
    /// was successfully constructed). When None, OandaPractice mode is
    /// rejected with a "router not initialized" error.
    pub oanda_router: Option<Arc<dyn OrderRouter>>,

    /// AlpacaRouter pointer. Set by main on startup if ALPACA_KEY +
    /// ALPACA_SECRET are present. Used for crypto orders in `Live` mode.
    pub alpaca_router: Option<Arc<dyn OrderRouter>>,

    /// Alpaca paper account snapshot (latest poll result).
    pub alpaca_account: RwLock<Option<AlpacaAccountSnapshot>>,

    /// Optional handle to the persistence DB. `None` when persistence
    /// is disabled. Used by the `/api/history` endpoint to serve the
    /// dashboard's plot prefill on page load.
    pub db: Option<persistence::Db>,

    /// Live ONNX inference registry. Holds the current champion (or
    /// the neutral fallback) and is hot-swapped when the research
    /// pipeline publishes a new model. `None` only during early boot
    /// before main() injects it.
    pub inference: Option<Arc<inference::PredictorRegistry>>,

    /// Single-flight slot for the Python pipeline orchestrator. Empty
    /// when nothing is running; `Some(flight)` while a `python -m
    /// research pipeline run` subprocess is in flight.
    pub pipeline_flight: PipelineFlightSlot,

    /// The last pipeline subprocess to complete. Updated by the
    /// watcher task in `pipeline_trigger` before the flight slot is
    /// released so the dashboard never sees a gap. Used to fetch the
    /// log of a recently-finished run after it's gone from the
    /// in-flight slot.
    pub last_completed_pipeline_flight: RwLock<Option<PipelineFlight>>,

    /// Auto-retrain state (per-instrument bar counters + last-fired
    /// stamps). The auto-retrain task spawned by main holds an Arc
    /// to the same value, so the status route reads the live state
    /// without round-tripping through a channel.
    pub auto_retrain: Option<Arc<AutoRetrain>>,
}

impl AppState {
    pub fn new(cfg: market_domain::Config, account_id: String) -> Self {
        let (bus, _) = broadcast::channel(BROADCAST_CAPACITY);
        let oracle = Arc::new(PriceOracle::new());
        let paper_router: Arc<dyn OrderRouter> = Arc::new(PaperRouter::new(oracle.clone()));
        let paper_router_held = Arc::new(paper_router.clone());
        Self {
            cfg,
            account_id,
            latest_prices: DashMap::new(),
            price_history: DashMap::new(),
            account: RwLock::new(None),
            nav_history: RwLock::new(History::default()),
            estimate: RwLock::new(None),
            estimate_history: RwLock::new(History::default()),
            transactions: RwLock::new(VecDeque::with_capacity(TRANSACTION_LIMIT)),
            conn: RwLock::new(ConnStatus::default()),
            bus,
            mode: ArcSwap::from_pointee(RoutingMode::Internal),
            paper_book: Arc::new(Mutex::new(PaperBook::new())),
            paper_book_version: std::sync::atomic::AtomicU64::new(0),
            price_oracle: oracle,
            router: ArcSwap::from_pointee(paper_router),
            paper_router: paper_router_held,
            oanda_router: None,
            alpaca_router: None,
            alpaca_account: RwLock::new(None),
            db: None,
            inference: None,
            pipeline_flight: PipelineFlightSlot::new(),
            last_completed_pipeline_flight: RwLock::new(None),
            auto_retrain: None,
        }
    }

    /// Inject the inference registry after construction (api-server::main
    /// does this once the registry + hot-swap watcher are spawned).
    pub fn with_inference(mut self, reg: Arc<inference::PredictorRegistry>) -> Self {
        self.inference = Some(reg);
        self
    }

    /// Inject the auto-retrain handle. Done after construction so the
    /// AutoRetrain instance can be built with the AppState's bus
    /// reference and shared with both the spawn task and the HTTP
    /// status route.
    pub fn with_auto_retrain(mut self, ar: Arc<AutoRetrain>) -> Self {
        self.auto_retrain = Some(ar);
        self
    }

    /// Inject the persistence DB after construction (api-server::main does
    /// this once it has opened SQLite). Idempotent.
    pub fn with_db(mut self, db: persistence::Db) -> Self {
        self.db = Some(db);
        self
    }

    /// Inject the OANDA router after construction (api-server::main does
    /// this once it has built a `Client`). Idempotent.
    pub fn with_oanda_router(mut self, router: Arc<dyn OrderRouter>) -> Self {
        self.oanda_router = Some(router);
        self
    }

    /// Inject the Alpaca router after construction.
    pub fn with_alpaca_router(mut self, router: Arc<dyn OrderRouter>) -> Self {
        self.alpaca_router = Some(router);
        self
    }

    /// Replace the Alpaca account snapshot with a fresh one parsed from
    /// the JSON body of `GET /v2/account`. Tolerant of missing fields.
    pub fn update_alpaca_account(&self, body: serde_json::Value) {
        fn fnum(v: &serde_json::Value, k: &str) -> f64 {
            v.get(k)
                .and_then(|x| x.as_str())
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or_else(|| v.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0))
        }
        let snap = AlpacaAccountSnapshot {
            time: chrono::Utc::now(),
            equity: fnum(&body, "equity"),
            cash: fnum(&body, "cash"),
            buying_power: fnum(&body, "buying_power"),
            portfolio_value: fnum(&body, "portfolio_value"),
            status: body
                .get("status")
                .and_then(|x| x.as_str())
                .unwrap_or("UNKNOWN")
                .to_string(),
            currency: body
                .get("currency")
                .and_then(|x| x.as_str())
                .unwrap_or("USD")
                .to_string(),
            long_market_value: fnum(&body, "long_market_value"),
            short_market_value: fnum(&body, "short_market_value"),
        };
        *self.alpaca_account.write() = Some(snap);
    }

    pub fn snapshot(&self) -> Snapshot {
        let mut prices: Vec<PriceTick> = self
            .latest_prices
            .iter()
            .map(|kv| kv.value().clone())
            .collect();
        prices.sort_by(|a, b| a.instrument.cmp(&b.instrument));
        let recent_transactions: Vec<TransactionEvent> =
            self.transactions.read().iter().cloned().collect();
        Snapshot {
            server_time: chrono::Utc::now(),
            prices,
            account: self.account.read().clone(),
            estimate: self.estimate.read().clone(),
            recent_transactions,
            connections: self.conn.read().clone(),
            alpaca: self.alpaca_account.read().clone(),
        }
    }

    pub fn record_price(&self, tick: PriceTick) {
        self.price_history
            .entry(tick.instrument.clone())
            .or_default()
            .push(tick.time, tick.mid);
        self.latest_prices
            .insert(tick.instrument.clone(), tick.clone());
        // Mirror to the price oracle so the router can fill at current bid/ask.
        self.price_oracle.update(tick.clone());
        let _ = self.bus.send(Event::Price(tick));
    }

    pub fn record_account(&self, snap: AccountSnapshot) {
        self.nav_history.write().push(snap.time, snap.nav);
        *self.account.write() = Some(snap.clone());
        let _ = self.bus.send(Event::Account(snap));
    }

    pub fn record_transaction(&self, ev: TransactionEvent) {
        {
            let mut q = self.transactions.write();
            if q.len() >= TRANSACTION_LIMIT {
                q.pop_front();
            }
            q.push_back(ev.clone());
        }
        let _ = self.bus.send(Event::Transaction(ev));
    }

    pub fn record_estimate(&self, est: EstimateTick) {
        self.estimate_history
            .write()
            .push(est.time, est.estimated_balance);
        *self.estimate.write() = Some(est.clone());
        let _ = self.bus.send(Event::Estimate(est));
    }

    pub fn touch_conn<F: FnOnce(&mut ConnStatus)>(self: &Arc<Self>, f: F) {
        let mut c = self.conn.write();
        f(&mut c);
        let snap = c.clone();
        drop(c);
        let _ = self.bus.send(Event::Status(snap));
    }
}
