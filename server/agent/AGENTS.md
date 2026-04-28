# AGENTS — oanda-live-server workspace

> Short router. Read this first. Each crate has its own README for
> deeper context.

## Mission

Stream live OANDA v20 market + account data, run an online ML
strategy, route trades to either an internal paper book or the OANDA
practice account, and fan everything out over a WebSocket to the
React dashboard.

## Start

```
just fast      # <2 min, run on every change
just medium    # <10 min, run before handoff
just dev       # cargo run -p api-server
```

## Map (owners)

| Crate            | Owns                                                              | Touches I/O? |
|------------------|-------------------------------------------------------------------|--------------|
| `market-domain`  | Pure types: PriceTick, AccountSnapshot, TransactionEvent, Estimate, typed IDs (Instrument, AccountId, OrderId, TransactionId), Config | No |
| `oanda-adapter`  | OANDA REST + streaming + (M6) OandaRouter                         | Yes (reqwest) |
| `portfolio`      | PaperBook, PaperRouter, OANDA-actual reconciler, 3-way Reconciliation | No |
| `feature-engine` | Tick → 18-feature vector, incremental rolling state               | No |
| `strategy`       | Online multinomial logistic regression, walk-forward fitness, signal/intent emission | No |
| `persistence`    | sqlx-sqlite: ticks, transactions, account snapshots, model snapshots, paper fills | Yes (file IO) |
| `api-server`     | axum HTTP + WebSocket, AppState wiring, mode toggle, inbound commands | Yes (network) |

Dependency DAG: `market-domain` is the leaf. `feature-engine`,
`portfolio`, `oanda-adapter`, `persistence` depend on
`market-domain`. `strategy` depends on `market-domain` +
`feature-engine`. `api-server` is the sink.

## Forbidden

- `target/` and other build artifacts — never commit.
- Adding a new dependency without updating `Cargo.toml` workspace
  deps (additive features only).
- Holding a `MutexGuard` across `.await`.
- Live-money routing. The toggle is two-position only:
  `Internal` ⇄ `OandaPractice`. `OANDA_ENV=live` is out of scope.

## Never compress (in proof receipts, summaries, handoff notes)

- Exit codes
- Failing test names (path + test fn name)
- Panic text, span, file:line
- Compiler error codes (E0XXX, lint name)
- Advisory IDs (RUSTSEC-YYYY-XXXX)
- Fuzz / proptest seeds
- Raw log paths and SHA-256 hashes
- Tool versions (cargo, rustc, just, sqlx, OANDA stream URL)
- Order IDs, transaction IDs, account IDs

## Security lane triggers

Run `just security` whenever you touch:

- `oanda-adapter` (auth, network, secrets)
- `crates/api-server/src/commands.rs` (inbound user commands)
- Anything that reads `.env` or constructs an OANDA URL
- Order placement / routing path
- New dependency adds

## Layout

See `agent/owner-map.json` for symbol → owner mapping.
See `agent/test-map.json` for symbol → test files.
See `agent/proof-lanes.toml` for which crates each lane runs.
See `agent/unsafe-ledger.toml` for any `unsafe` justifications.
