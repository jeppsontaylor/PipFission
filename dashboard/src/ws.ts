// WebSocket client with auto-reconnect and exponential backoff. Exposes a
// minimal pub/sub for ServerEvents so React components can listen via hooks.

import type { ClientCommand, ServerEvent } from "./types";

type Listener = (e: ServerEvent) => void;
type StatusListener = (s: WsStatus) => void;

export type WsStatus = "connecting" | "open" | "closed";

export class LiveSocket {
  private url: string;
  private ws: WebSocket | null = null;
  private listeners = new Set<Listener>();
  private statusListeners = new Set<StatusListener>();
  private status: WsStatus = "connecting";
  private backoff = 250;
  private maxBackoff = 10_000;
  private stopped = false;
  /** Outbound commands queued while the socket is reconnecting. Drained on open. */
  private pending: ClientCommand[] = [];

  constructor(url: string) {
    this.url = url;
  }

  start(): void {
    this.stopped = false;
    this.connect();
  }

  stop(): void {
    this.stopped = true;
    this.ws?.close();
    this.ws = null;
  }

  on(fn: Listener): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  onStatus(fn: StatusListener): () => void {
    this.statusListeners.add(fn);
    fn(this.status);
    return () => this.statusListeners.delete(fn);
  }

  /**
   * Send a typed ClientCommand to the server. If the socket is not yet
   * open the command is queued and sent on the next connect.
   */
  send(cmd: ClientCommand): void {
    if (this.ws && this.status === "open") {
      try {
        this.ws.send(JSON.stringify(cmd));
        return;
      } catch (err) {
        console.warn("ws: send failed, queueing", err);
      }
    }
    this.pending.push(cmd);
  }

  private drainPending() {
    if (!this.ws || this.status !== "open") return;
    const queue = this.pending;
    this.pending = [];
    for (const cmd of queue) {
      try {
        this.ws.send(JSON.stringify(cmd));
      } catch (err) {
        console.warn("ws: drain send failed, requeuing", err);
        this.pending.push(cmd);
      }
    }
  }

  private setStatus(s: WsStatus) {
    this.status = s;
    for (const fn of this.statusListeners) fn(s);
  }

  private connect() {
    if (this.stopped) return;
    this.setStatus("connecting");
    const ws = new WebSocket(this.url);
    this.ws = ws;

    ws.onopen = () => {
      this.backoff = 250;
      this.setStatus("open");
      this.drainPending();
    };
    ws.onmessage = (msg) => {
      try {
        const ev = JSON.parse(msg.data) as ServerEvent;
        for (const fn of this.listeners) fn(ev);
      } catch (err) {
        // Ignore parse errors but log for debugging.
        console.warn("ws: bad message", err);
      }
    };
    ws.onclose = () => {
      this.setStatus("closed");
      if (!this.stopped) {
        const wait = this.backoff;
        this.backoff = Math.min(this.backoff * 2, this.maxBackoff);
        setTimeout(() => this.connect(), wait);
      }
    };
    ws.onerror = () => {
      // onclose will fire next; let it handle reconnect.
    };
  }
}

/** Returns a WebSocket URL derived from the current page origin. */
export function defaultWsUrl(): string {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}/ws`;
}
