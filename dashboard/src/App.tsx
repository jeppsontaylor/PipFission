import { useMemo, useState } from "react";

import { AccountOverview } from "./components/AccountOverview";
import { AlpacaAccountCard } from "./components/AlpacaAccountCard";
import { BalanceChart } from "./components/BalanceChart";
import { ConnectionStatus } from "./components/ConnectionStatus";
import { DataHeaderPanel } from "./components/DataHeaderPanel";
import { LiveChartView } from "./components/LiveChartView";
import { ManualOrderPanel } from "./components/ManualOrderPanel";
import { MLPipelinePanel } from "./components/MLPipelinePanel";
import { ModeToggle } from "./components/ModeToggle";
import { ModelStatusCard } from "./components/ModelStatusCard";
import { ChampionBanner } from "./components/ChampionBanner";
import { DeploymentGateCard } from "./components/DeploymentGateCard";
import { LiveTradeTape } from "./components/LiveTradeTape";
import { ModelZooCard } from "./components/ModelZooCard";
import { OptimizerTrialsTable } from "./components/OptimizerTrialsTable";
import { PipelineRunsCard } from "./components/PipelineRunsCard";
import { TradePerformanceBreakdown } from "./components/TradePerformanceBreakdown";
import { PaperBookPanel } from "./components/PaperBookPanel";
import { PaperFillsLog } from "./components/PaperFillsLog";
import { PriceGrid } from "./components/PriceGrid";
import { RejectionBanner } from "./components/RejectionBanner";
import { SignalsTable } from "./components/SignalsTable";
import { TradeLog } from "./components/TradeLog";
import { VenueFilter, type VenueChoice } from "./components/VenueFilter";
import { useInstrumentList, useLiveStore, useRejectionRate } from "./store";
import { instrumentVenue } from "./types";

export default function App() {
  const { state, setMode, manualOrder } = useLiveStore();
  const allInstruments = useInstrumentList(state);
  const isOanda = state.mode === "oanda_practice";
  const rejectionRate = useRejectionRate(state);

  const [venueFilter, setVenueFilter] = useState<VenueChoice>("all");

  const { forexCount, cryptoCount, filteredInstruments } = useMemo(() => {
    const forex = allInstruments.filter((i) => instrumentVenue(i) === "oanda");
    const crypto = allInstruments.filter((i) => instrumentVenue(i) === "alpaca");
    const filtered =
      venueFilter === "all"
        ? allInstruments
        : venueFilter === "oanda"
          ? forex
          : crypto;
    return {
      forexCount: forex.length,
      cryptoCount: crypto.length,
      filteredInstruments: filtered,
    };
  }, [allInstruments, venueFilter]);

  // Filter signals + paper-fills + transactions by the selected venue too,
  // so the "Crypto only" view is genuinely crypto-focused.
  const filteredSignals = useMemo(() => {
    if (venueFilter === "all") return state.signals;
    const out: typeof state.signals = {};
    for (const inst of filteredInstruments) {
      if (state.signals[inst]) out[inst] = state.signals[inst];
    }
    return out;
  }, [state.signals, filteredInstruments, venueFilter]);

  const filteredPaperFills = useMemo(() => {
    if (venueFilter === "all") return state.paperFills;
    return state.paperFills.filter((f) => instrumentVenue(f.instrument) === venueFilter);
  }, [state.paperFills, venueFilter]);

  return (
    <div className="min-h-screen w-full">
      {isOanda && <div className="h-1 w-full bg-amber-500" aria-hidden="true" />}
      <header className="border-b border-ink-700 bg-ink-900/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-[1800px] mx-auto px-6 py-3 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent-500 to-fuchsia-500" />
            <div>
              <div className="font-semibold tracking-wide">OANDA + Alpaca · Live</div>
              <div className="text-[11px] text-slate-400 font-mono">
                {state.environment ?? "—"} · {state.accountId ?? "—"}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <ModeToggle
              mode={state.mode}
              pendingMode={state.pendingMode}
              lastError={state.lastModeAck?.error ?? null}
              onChange={setMode}
            />
            <ConnectionStatus ws={state.wsStatus} conn={state.connections} />
          </div>
        </div>
      </header>

      <main className="max-w-[1800px] mx-auto px-6 py-6 space-y-6">
        <RejectionBanner
          mode={state.mode}
          count={rejectionRate.count}
          topReason={rejectionRate.topReason}
          topReasonCount={rejectionRate.topReasonCount}
        />

        <ChampionBanner status={state.ml.championStatus} />

        {/* Pipeline header — totals + ETA to first model + scorecard.
            Lives above the chart so it's the first thing the user reads. */}
        <DataHeaderPanel
          instruments={allInstruments}
          priceCounts={state.priceCounts}
          featureCounts={state.featureCounts}
          signalCounts={state.signalCounts}
          fillCounts={state.fillCounts}
          priceTimes={state.priceTimes}
          fitness={state.fitness}
        />

        {/* Big live chart — the primary surface. Streaming price + fills +
            confidence trace per instrument. */}
        <LiveChartView
          instruments={filteredInstruments}
          prices={state.prices}
          priceHistory={state.priceHistory}
          fills={state.paperFills}
          signals={state.signals}
          signalHistory={state.signalHistory}
          fitness={state.fitness}
        />

        <div className="flex flex-wrap items-center justify-between gap-3">
          <VenueFilter
            value={venueFilter}
            onChange={setVenueFilter}
            forexCount={forexCount}
            cryptoCount={cryptoCount}
          />
          <div className="text-xs text-slate-500">
            {filteredInstruments.length} of {allInstruments.length} instruments shown
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <AccountOverview
            account={state.account}
            estimate={state.estimate}
            accountId={state.accountId}
            environment={state.environment}
          />
          <AlpacaAccountCard account={state.alpacaAccount} />
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <PaperBookPanel
            paperBook={state.paperBook}
            reconciliation={state.reconciliation}
            mode={state.mode}
          />
          <ManualOrderPanel
            instruments={allInstruments}
            mode={state.mode}
            onSubmit={manualOrder}
          />
        </div>

        <ModelStatusCard
          featureCounts={state.featureCounts}
          fitness={state.fitness}
          instruments={filteredInstruments}
        />

        {filteredInstruments.length > 0 && (
          <MLPipelinePanel state={state} instruments={filteredInstruments} />
        )}

        <LiveTradeTape />

        <TradePerformanceBreakdown />

        <PipelineRunsCard instruments={allInstruments} />

        <DeploymentGateCard instruments={allInstruments} />

        <ModelZooCard instruments={allInstruments} />

        <OptimizerTrialsTable />

        <BalanceChart history={state.navHistory} />

        <SignalsTable signals={filteredSignals} instruments={filteredInstruments} />

        <PriceGrid
          prices={state.prices}
          history={state.priceHistory}
          instruments={filteredInstruments}
        />

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <PaperFillsLog fills={filteredPaperFills} />
          <TradeLog transactions={state.transactions} />
        </div>

        <footer className="text-center text-xs text-slate-500 pt-4 pb-8">
          Forex via <span className="font-mono">stream-fxpractice.oanda.com</span> · Crypto via{" "}
          <span className="font-mono">stream.data.alpaca.markets</span> · Rust server on{" "}
          <span className="font-mono">:8088</span>
        </footer>
      </main>
    </div>
  );
}
