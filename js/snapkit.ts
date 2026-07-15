/**
 * snapkit.js — JavaScript/TypeScript port of snapkit-v2 core.
 *
 * The core primitives — Eisenstein lattice, BeatGrid, spectral analysis,
 * connectome — run in the browser. This is the visual side of harmony.
 *
 * Built for the canvas dashboard. The browser is the deck where you watch
 * the harmony unfold.
 *
 * MIT License — SuperInstance
 */

// ═══════════════════════════════════════════════════════════════
// Eisenstein Lattice
// ═══════════════════════════════════════════════════════════════

export class EisensteinInteger {
  constructor(public a: number, public b: number) {
    if (!Number.isInteger(a) || !Number.isInteger(b)) {
      throw new Error("EisensteinInteger requires integers");
    }
  }

  get normSquared(): number {
    return this.a * this.a - this.a * this.b + this.b * this.b;
  }

  get norm(): number {
    return Math.sqrt(this.normSquared);
  }

  get complex(): { x: number; y: number } {
    return { x: this.a - this.b / 2, y: (this.b * Math.sqrt(3)) / 2 };
  }

  static fromComplex(z: { x: number; y: number }): EisensteinInteger {
    // Approximate round to nearest Eisenstein integer
    const a = z.x - z.y / Math.sqrt(3);
    const b = 2 * z.y / Math.sqrt(3);
    // Try all 3 nearest lattice points and pick closest
    const candidates = [
      [Math.round(a), Math.round(b)],
      [Math.round(a - 0.5), Math.round(b + 0.5)],
      [Math.round(a + 0.5), Math.round(b - 0.5)],
    ];
    let best = candidates[0];
    let bestDist = Infinity;
    for (const [ca, cb] of candidates) {
      const d = (ca - a) ** 2 + (cb - b) ** 2;
      if (d < bestDist) {
        bestDist = d;
        best = [ca, cb];
      }
    }
    return new EisensteinInteger(best[0], best[1]);
  }
}

export function eisensteinRound(z: { x: number; y: number }): EisensteinInteger {
  return EisensteinInteger.fromComplex(z);
}

export function eisensteinSnap(
  z: { x: number; y: number },
  tolerance: number = 0.5
): [EisensteinInteger, number, boolean] {
  const nearest = eisensteinRound(z);
  const dz = { x: z.x - nearest.complex.x, y: z.y - nearest.complex.y };
  const distance = Math.sqrt(dz.x * dz.x + dz.y * dz.y);
  return [nearest, distance, distance <= tolerance];
}

// ═══════════════════════════════════════════════════════════════
// BeatGrid
// ═══════════════════════════════════════════════════════════════

export interface TemporalResult {
  originalTime: number;
  snappedTime: number;
  offset: number;
  isOnBeat: boolean;
  isTMinusZero: boolean;
  beatIndex: number;
  beatPhase: number;
}

export class BeatGrid {
  constructor(
    public period: number = 1.0,
    public phase: number = 0.0,
    public tStart: number = 0.0
  ) {
    if (period <= 0) throw new Error("period must be positive");
  }

  nearestBeat(t: number): [number, number] {
    const adjusted = t - this.tStart - this.phase;
    const index = Math.round(adjusted / this.period);
    const beatTime = this.tStart + this.phase + index * this.period;
    return [beatTime, index];
  }

  snap(t: number, tolerance: number = 0.1): TemporalResult {
    const [beatTime, beatIndex] = this.nearestBeat(t);
    const offset = t - beatTime;
    const isOnBeat = Math.abs(offset) <= tolerance;
    const adjusted = t - this.tStart - this.phase;
    const phase =
      ((((adjusted % this.period) + this.period) % this.period) / this.period);

    return {
      originalTime: t,
      snappedTime: beatTime,
      offset,
      isOnBeat,
      isTMinusZero: false,
      beatIndex,
      beatPhase: phase,
    };
  }
}

// ═══════════════════════════════════════════════════════════════
// Spectral Analysis
// ═══════════════════════════════════════════════════════════════

export function entropy(data: number[], bins: number = 10): number {
  const n = data.length;
  if (n < 2) return 0.0;

  let minVal = data[0], maxVal = data[0];
  for (const x of data) {
    if (x < minVal) minVal = x;
    if (x > maxVal) maxVal = x;
  }
  if (maxVal === minVal) return 0.0;

  const invRange = bins / (maxVal - minVal);
  const counts = new Array(bins).fill(0);
  for (const x of data) {
    let idx = Math.floor((x - minVal) * invRange);
    if (idx >= bins) idx = bins - 1;
    counts[idx]++;
  }

  const invN = 1.0 / n;
  const invLog2 = 1.0 / Math.log(2);
  let h = 0;
  for (const c of counts) {
    if (c > 0) {
      const p = c * invN;
      h -= p * Math.log(p) * invLog2;
    }
  }
  return h;
}

export function hurstExponent(data: number[]): number {
  const n = data.length;
  if (n < 20) return 0.5;

  const invN = 1.0 / n;
  const mean = data.reduce((a, b) => a + b, 0) * invN;
  const centered = data.map(x => x - mean);

  const testSizes: number[] = [];
  let s = 16;
  while (s <= n / 2) {
    testSizes.push(s);
    s = Math.floor(s * 2);
  }

  const sizes: number[] = [];
  const rsValues: number[] = [];

  for (const size of testSizes) {
    if (size < 4 || size > n) continue;
    const numSubseries = Math.floor(n / size);
    if (numSubseries < 1) continue;

    const invSize = 1.0 / size;
    let rsSum = 0;
    let rsCount = 0;

    for (let i = 0; i < numSubseries; i++) {
      const start = i * size;
      const sub = centered.slice(start, start + size);
      const subMean = sub.reduce((a, b) => a + b, 0) * invSize;

      let running = 0, cumMin = 0, cumMax = 0;
      for (const x of sub) {
        running += x - subMean;
        if (running < cumMin) cumMin = running;
        if (running > cumMax) cumMax = running;
      }
      const r = cumMax - cumMin;

      let v = 0;
      for (const x of sub) {
        const d = x - subMean;
        v += d * d;
      }
      v *= invSize;

      if (v > 1e-20) {
        rsSum += r / Math.sqrt(v);
        rsCount++;
      }
    }

    if (rsCount > 0) {
      const avgRs = rsSum / rsCount;
      if (avgRs > 0) {
        sizes.push(size);
        rsValues.push(avgRs);
      }
    }
  }

  if (sizes.length < 2) return 0.5;

  const logN = sizes.map(Math.log);
  const logRs = rsValues.map(Math.log);
  const np = sizes.length;

  let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
  for (let i = 0; i < np; i++) {
    sumX += logN[i];
    sumY += logRs[i];
    sumXY += logN[i] * logRs[i];
    sumX2 += logN[i] * logN[i];
  }

  const denom = np * sumX2 - sumX * sumX;
  if (denom === 0) return 0.5;
  const h = (np * sumXY - sumX * sumY) / denom;
  return Math.max(0, Math.min(1, h));
}

export interface SpectralSummary {
  entropyBits: number;
  hurst: number;
  isStationary: boolean;
}

export function spectralSummary(data: number[]): SpectralSummary {
  const h = entropy(data);
  const hurstVal = hurstExponent(data);
  const isStationary = (0.4 <= hurstVal && hurstVal <= 0.6);
  return { entropyBits: h, hurst: hurstVal, isStationary };
}

// ═══════════════════════════════════════════════════════════════
// FluxTensorMIDI (simplified, browser-friendly)
// ═══════════════════════════════════════════════════════════════

export enum MIDIEventType {
  NOTE_ON = 0x90,
  NOTE_OFF = 0x80,
  CONTROL_CHANGE = 0xB0,
}

export interface MIDIEvent {
  tick: number;
  channel: number;
  type: MIDIEventType;
  value: number;
  velocity: number;
}

export class FluxTensorMIDI {
  private events: MIDIEvent[] = [];

  addEvent(tick: number, channel: number, type: MIDIEventType,
           value: number, velocity: number = 0): MIDIEvent {
    const event: MIDIEvent = { tick, channel, type, value, velocity };
    this.events.push(event);
    return event;
  }

  noteOn(room: string, tick: number, note: number, velocity: number = 100,
         channel: number = 0): MIDIEvent {
    return this.addEvent(tick, channel, MIDIEventType.NOTE_ON, note, velocity);
  }

  noteOff(room: string, tick: number, note: number,
          channel: number = 0): MIDIEvent {
    return this.addEvent(tick, channel, MIDIEventType.NOTE_OFF, note, 0);
  }

  render(): MIDIEvent[] {
    return [...this.events].sort((a, b) => a.tick - b.tick);
  }

  clear(): void {
    this.events = [];
  }

  get eventCount(): number {
    return this.events.length;
  }
}

// ═══════════════════════════════════════════════════════════════
// HarmonyGovernor (simplified, in-browser)
// ═══════════════════════════════════════════════════════════════

export enum FrictionLevel {
  HARMONY = 0,
  STRAIN = 1,
  SURPRISE = 2,
}

export interface FrictionAlarm {
  room: string;
  channel: number;
  level: FrictionLevel;
  phi: number;
  entropy: number;
  hurst: number;
  tick: number;
  timestamp: number;
}

export interface ChannelConfig {
  name: string;
  channel: number;
  deadband: number;
  windowSize: number;
}

export class ChannelState {
  predictions: number[] = [];
  actuals: number[] = [];
  phiHistory: number[] = [];
  lastSummary: SpectralSummary | null = null;
  sustainedCount = 0;

  constructor(public config: ChannelConfig) {}

  record(prediction: number, actual: number): number {
    this.predictions.push(prediction);
    this.actuals.push(actual);
    if (this.predictions.length > this.config.windowSize) {
      this.predictions.shift();
      this.actuals.shift();
    }

    const error = Math.abs(prediction - actual);

    if (this.actuals.length < 4) {
      this.phiHistory.push(error);
      return error;
    }

    const errors = this.predictions.map((p, i) => p - this.actuals[i]);
    const filteredErrors = errors.filter(e => !isNaN(e) && isFinite(e));

    if (filteredErrors.length < 4) {
      this.phiHistory.push(error);
      return error;
    }

    this.lastSummary = spectralSummary(filteredErrors);

    // Normalize entropy
    const maxEntropy = Math.log2(Math.min(10, filteredErrors.length / 2));
    const normE = this.lastSummary.entropyBits / (maxEntropy || 1);

    // Direct error magnitude
    const recentErrors = filteredErrors.slice(-8);
    const meanAbs = recentErrors.reduce((a, b) => a + Math.abs(b), 0) / recentErrors.length;
    const normError = Math.min(meanAbs / 10, 1);

    // Hurst penalty
    let hurstPenalty = 0;
    if (this.lastSummary.hurst < 0.45) {
      hurstPenalty = (0.45 - this.lastSummary.hurst) * 2;
    }

    const phi = 0.6 * normE + 0.1 * hurstPenalty + 0.4 * normError;
    this.phiHistory.push(phi);
    return phi;
  }

  get phi(): number {
    return this.phiHistory.length > 0
      ? this.phiHistory[this.phiHistory.length - 1]
      : 0;
  }

  get isInHarmony(): boolean { return this.phi < this.config.deadband * 0.7; }
  get isStrained(): boolean {
    return this.config.deadband * 0.7 <= this.phi && this.phi < this.config.deadband;
  }
  get isSurprised(): boolean { return this.phi >= this.config.deadband; }
}

export class HarmonyGovernor {
  channels = new Map<string, ChannelState>();
  private alarms: FrictionAlarm[] = [];
  private tickCount = 0;
  private sustainedThreshold: number;

  constructor(sustainedThreshold: number = 3) {
    this.sustainedThreshold = sustainedThreshold;
  }

  registerChannel(name: string, channel: number, deadband: number = 1.5,
                   windowSize: number = 64): ChannelState {
    if (this.channels.has(name)) {
      throw new Error(`channel '${name}' already registered`);
    }
    const state = new ChannelState({ name, channel, deadband, windowSize });
    this.channels.set(name, state);
    return state;
  }

  tick(tick?: number): void {
    this.tickCount = tick !== undefined ? tick : this.tickCount + 1;
  }

  recordObservation(name: string, prediction: number, actual: number): number {
    const state = this.channels.get(name);
    if (!state) throw new Error(`channel '${name}' not registered`);
    const phi = state.record(prediction, actual);

    if (state.isSurprised) {
      state.sustainedCount++;
      if (state.sustainedCount >= this.sustainedThreshold) {
        this.alarms.push({
          room: name,
          channel: state.config.channel,
          level: FrictionLevel.SURPRISE,
          phi: state.phi,
          entropy: state.lastSummary?.entropyBits || 0,
          hurst: state.lastSummary?.hurst || 0.5,
          tick: this.tickCount,
          timestamp: Date.now() / 1000,
        });
      }
    } else {
      state.sustainedCount = Math.max(0, state.sustainedCount - 1);
    }

    return phi;
  }

  get unacknowledgedAlarms(): FrictionAlarm[] {
    return this.alarms.filter(a => a.level === FrictionLevel.SURPRISE);
  }

  clearAlarms(): void { this.alarms = []; }
  get tick_count(): number { return this.tickCount; }

  globalPhi(): number {
    if (this.channels.size === 0) return 0;
    let sum = 0;
    for (const c of this.channels.values()) sum += c.phi;
    return sum / this.channels.size;
  }

  systemState(): object {
    const channels: any = {};
    for (const [name, c] of this.channels) {
      channels[name] = {
        phi: c.phi,
        harmony: c.isInHarmony,
        strained: c.isStrained,
        surprised: c.isSurprised,
        hurst: c.lastSummary?.hurst,
        entropy: c.lastSummary?.entropyBits,
      };
    }
    return {
      tick: this.tickCount,
      globalPhi: this.globalPhi(),
      channels,
      activeAlarms: this.unacknowledgedAlarms.length,
    };
  }
}

// ═══════════════════════════════════════════════════════════════
// Clever Tokens
// ═══════════════════════════════════════════════════════════════

export enum ConstraintType {
  RIGID = 0,
  ELASTIC = 1,
  PERIODIC = 2,
  COUPLED = 3,
  DIRECTIVE = 4,
}

export interface CleverToken {
  identifier: string;
  lattice: EisensteinInteger;
  type: ConstraintType;
  snapRadius: number;
  expectedEntropy: number;
  expectedHurst: number;
  coupledWith?: string;
  directiveTarget?: EisensteinInteger;
}

export class TokenLattice {
  private tokens = new Map<string, CleverToken>();

  register(token: Omit<CleverToken, "lattice"> & { latCoord: [number, number] }): CleverToken {
    const fullToken: CleverToken = {
      ...token,
      lattice: new EisensteinInteger(token.latCoord[0], token.latCoord[1]),
    };
    this.tokens.set(token.identifier, fullToken);
    return fullToken;
  }

  snap(entropy: number, hurst: number): [CleverToken | null, number, boolean] {
    if (this.tokens.size === 0) return [null, Infinity, false];

    let bestToken: CleverToken | null = null;
    let bestDev = Infinity;

    for (const token of this.tokens.values()) {
      const normE_obs = Math.min(entropy / 8, 1);
      const normE_exp = Math.min(token.expectedEntropy / 8, 1);
      const dev = Math.sqrt(
        (normE_obs - normE_exp) ** 2 + (hurst - token.expectedHurst) ** 2
      );
      if (dev < bestDev) {
        bestDev = dev;
        bestToken = token;
      }
    }

    const onLattice = bestToken ? bestDev <= bestToken.snapRadius : false;
    return [bestToken, bestDev, onLattice];
  }

  get tokens_list(): string[] {
    return Array.from(this.tokens.keys());
  }

  get token(tokenId: string): CleverToken {
    return this.tokens.get(tokenId)!;
  }
}