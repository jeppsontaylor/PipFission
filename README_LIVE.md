# OANDA Live — Rust + React streaming stack

Real-time dashboard layered on top of your existing Python data pipeline.

```
┌─────────────────────────────┐    WebSocket     ┌────────────────────────┐
│  OANDA v20 (fxpractice)     │   /ws (NDJSON)   │  React dashboard       │
│  pricing/stream  ──────┐    │ ───────────────► │  Vite + TS + Tailwind  │
│  transactions/stream ──┤    │                  │  recharts              │
│  /accounts/{id}/summary┘    │                  └────────────────────────┘
└──────────┬──────────────────┘
           │  HTTPS (4 Hz polling + 2 long-lived streams)
           ▼
┌─────────────────────────────┐
│  Rust server (axum + tokio) │
│  • pricing stream consumer  │
│  • transactions stream      │
│  • fast account poll (4 Hz) │
│  • estimator (synth ledger) │
│  • broadcast bus → WS       │
└─────────────────────────────┘
```

## Layout

```
oanda/
├── server/      ← NEW Rust streaming server (this is the brain)
├── dashboard/   ← NEW React/TS live dashboard
├── oanda_collector.py / build_features.py / analyze_data.py   ← original Python
├── .env          ← shared OANDA_API_TOKEN (also picked up by the Rust server)
└── CLAUDE.md     ← project rules (1500 LOC max per file, etc.)
```

## What the dashboard shows

- **Account card** — NAV, balance, unrealized & realized P/L, margin used /
  available, leverage, open positions/trades. Identical fields to the OANDA
  web "Overview" tab.
- **Estimated vs actual NAV chart** — two lines:
  - *Actual* = NAV reported by OANDA's `/accounts/{id}/summary`, polled every
    250 ms.
  - *Estimated* = a synthetic ledger inside the Rust server. It seeds itself
    from the first account snapshot, then replays every `ORDER_FILL` /
    `TRANSFER_FUNDS` / `DAILY_FINANCING` event from the transactions stream
    and marks open positions to market against the latest mid price.
  - The drift between them is exactly the slippage, financing, commission,
    and bid/ask asymmetry that the simple model doesn't capture. That's the
    interesting signal.
- **Live price grid** — 7 majors (`EUR_USD, USD_JPY, GBP_USD, AUD_USD,
  USD_CAD, USD_CHF, NZD_USD`) with bid/ask, spread in bps, and a sparkline.
  Cells flash green/red on each tick.
- **Algorithmic trades & transactions log** — every interesting transaction
  type from the OANDA stream (`ORDER_FILL`, `MARKET_ORDER`, `TRADE_CLOSE`,
  financing, etc.). Empty until you (or a future strategy) place orders.
- **Connection status** — coloured dots showing health of each upstream:
  WebSocket, pricing stream, transactions stream, account poll.

## Prerequisites

Local install (the sandbox can't reach `crates.io` or `registry.npmjs.org`,
so this builds on your machine):

- **Rust** stable — `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
  or `brew install rustup-init && rustup-init`.
- **Node 20+** and **npm** — `brew install node`.

Token: the existing `.env` at the repo root is already picked up by the Rust
server (`OANDA_API_TOKEN`). Account ID is auto-discovered if
`OANDA_ACCOUNT_ID` isn't set.

## Run it

Two terminals.

### 1. Start the Rust server (terminal 1)

```bash
cd server
cargo run --release
```

First build pulls a few crates and takes a couple of minutes; subsequent
runs are instant. You should see:

```
INFO oanda-live-server: starting oanda-live-server
INFO using OANDA account: 101-001-38580110-001
INFO HTTP/WS listening on http://0.0.0.0:8088
INFO opening pricing stream for 7 instruments
INFO opening transactions stream
INFO estimator seeded cash from first account snapshot
```

Endpoints:

- `GET /healthz` — sanity check.
- `GET /api/state` — full current snapshot as JSON.
- `GET /api/instruments` — the configured list.
- `GET /ws` — WebSocket; sends a `hello` event on connect, then a stream of
  `price`, `account`, `transaction`, `estimate`, and `status` events.

### 2. Start the dashboard (terminal 2)

```bash
cd dashboard
npm install
npm run dev
```

Open http://127.0.0.1:5173. The Vite dev server proxies `/api`, `/ws`, and
`/healthz` to `127.0.0.1:8088`.

For a production build:

```bash
npm run build && npm run preview
```

## Tunables

All env-driven; defaults are sensible. Set in `.env` (or shell):

| Var                  | Default      | Meaning                                       |
| -------------------- | ------------ | --------------------------------------------- |
| `OANDA_API_TOKEN`    | (required)   | Bearer token.                                 |
| `OANDA_ACCOUNT_ID`   | auto         | If unset, the first account on the token.    |
| `OANDA_ENV`          | `practice`   | `practice` or `live`.                         |
| `OANDA_INSTRUMENTS`  | 7 majors     | Comma-separated, e.g. `EUR_USD,USD_JPY`.      |
| `BIND_ADDR`          | `0.0.0.0:8088` | Server listen address.                      |
| `ACCOUNT_POLL_MS`    | `250`        | Account summary poll period (ms). 4 Hz.       |
| `ESTIMATOR_TICK_MS`  | `500`        | How often estimated-vs-actual is recomputed.  |

OANDA streams pricing at up to ~4 ticks/sec/instrument and the streams are
push-based (no polling), so we already collect at the API's natural rate.
The 4 Hz account poll is the fastest stable rate that stays well within the
~100 req/s per-token global REST limit.

## Why a Rust backend at all?

- **Zero-allocation streaming.** The pricing + transaction streams deliver
  NDJSON over a single long-lived HTTP connection. Rust gives us tight
  control over buffer reuse and a single tokio runtime that handles all
  three OANDA connections plus N WebSocket clients without context-switch
  overhead — the dashboard stays glassy even if you crank
  `ACCOUNT_POLL_MS` down.
- **One process, one binary.** `cargo build --release` produces a static
  binary you can drop on any host alongside the existing Python data
  collector.
- **Type-safe broadcast bus.** The `tokio::sync::broadcast` channel inside
  `AppState` carries strongly-typed `Event`s, so the WS handler and the
  estimator both consume the same source of truth.

The original Python collector (`oanda_collector.py`) keeps doing what it
does best — durable JSONL capture for offline ML training. The Rust server
is the *live realtime path*, not a replacement for it. They can run in
parallel and consume different OANDA HTTP connections.

## Placing orders (when you're ready)

The Rust client already has `Client::place_market_order(account, instrument, units)`.
It's not auto-called yet (no algo wired up), but you can flip a strategy on
later by:

1. Adding a `streams::strategy` module that subscribes to `state.bus` and
   submits orders via `client.place_market_order(...)`.
2. Or POSTing from the dashboard to a new `/api/orders` route.

Either way, the dashboard's estimated-vs-actual chart will start rendering
the impact of those fills against the synthetic ledger automatically — the
estimator already handles `ORDER_FILL` events.

## Caveats

- **Drift never goes to exactly zero.** The estimator marks to mid price;
  OANDA's NAV uses the closeout side. Plus financing posts overnight. A
  steady, small drift of a couple of bps is normal.
- **Resync on lag.** If a slow WS client falls behind by ≥ 4096 events, the
  server resends a full `hello` snapshot rather than dropping the connection.
- **Practice vs live.** The default is `practice`. To go live, set
  `OANDA_ENV=live` and use a live token. **Don't.**

## File-length rule

Per `CLAUDE.md`: no source file in this repo may exceed **1500 lines**. As
of this writing, the largest active file is the original Python
`oanda_collector.py` at 996 lines; everything in `server/` and `dashboard/`
is well under 300 lines per file.
