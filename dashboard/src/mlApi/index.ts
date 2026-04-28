// Public surface of the dashboard's REST clients.
//
// Importers should always use `"./mlApi"` (or `"../mlApi"`) — the
// per-resource files behind this index are an implementation detail.
//
// One file per backend resource:
//   * champion.ts    — /api/strategy/champion + /api/{model,trader}/metrics + /api/lockbox/result
//   * optimizer.ts   — /api/optimizer/trials
//   * labels.ts      — /api/labels/recent
//   * trades.ts      — /api/trade/ledger
//   * pipeline.ts    — /api/pipeline/{runs,run,status,last-completed,log,auto-retrain}
//   * internal.ts    — shared `BASE` + `fetchJson` (NOT re-exported)

export * from "./champion";
export * from "./optimizer";
export * from "./labels";
export * from "./trades";
export * from "./pipeline";
