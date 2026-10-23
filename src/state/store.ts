import type { Arithmetic } from "../components/FormulaView";
import type { Config, Options, Patch } from "../core/model/config";
import type { Forward } from "../core/model/forward";
import type { Weights } from "../core/model/weights";
import type { BPEOptions, Tokenization } from "../core/tokenizer/bpe";
import type { SampleResult, Sampling } from "../core/sample";
export interface RunRequest {
  text: string;
  options: Options;
  bpe: BPEOptions;
  sampling: Sampling;
  maxSeq: number;
  ids?: number[];
  patch?: Patch;
  choices?: number[];
  step?: number;
}
export interface RunResult {
  tokenization: Tokenization;
  trace: Forward;
  sample: SampleResult;
  vocab: string[];
  parameterCount: number;
  elapsed: number;
  truncated: number;
  track: string;
  normComparison: { pre: Float32Array; post: Float32Array };
  baseline?: Float32Array;
  delta?: Float32Array;
  affected?: number;
  contextStart: number;
}
export interface SerializedModel {
  format: "token-by-token-v1";
  config: Config;
  weights: Record<string, number[]>;
  vocab: string[];
  options?: Options;
  tokenizer?: { vocab: string[]; merges: [string, string][]; pattern?: "gpt2" };
  source?: string;
}
export interface TrainPoint {
  step: number;
  loss: number;
  gradNorm: number;
  clippedNorm: number;
  elapsed: number;
  embedding: Float32Array;
  total: number;
}
export interface SavedStep {
  request: RunRequest;
  result: RunResult;
  generatedIds: number[];
  sampling: Sampling;
  options: Options;
}
let worker: Worker | undefined,
  nextId = 0;
const pending = new Map<
  number,
  { resolve: (value: unknown) => void; reject: (e: Error) => void }
>();
function getWorker() {
  if (!worker) {
    worker = new Worker(
      new URL("../workers/compute.worker.ts", import.meta.url),
      { type: "module" },
    );
    worker.onmessage = (e) => {
      const p = pending.get(e.data.id);
      if (p) {
        pending.delete(e.data.id);
        if (e.data.error) p.reject(new Error(e.data.error));
        else p.resolve(e.data.result);
      }
    };
    worker.onerror = (e) => {
      for (const p of pending.values())
        p.reject(new Error(e.message || "计算线程意外停止"));
      pending.clear();
      worker?.terminate();
      worker = undefined;
    };
  }
  return worker;
}
export function rpc<T>(type: string, payload: unknown): Promise<T> {
  return new Promise((resolve, reject) => {
    const id = ++nextId;
    pending.set(id, { resolve: resolve as (value: unknown) => void, reject });
    getWorker().postMessage({ id, type, payload });
  });
}
export const run = (r: RunRequest) => rpc<RunResult>("run", r);
export const inspect = (left: Float32Array, right: Float32Array, bias = 0) =>
  rpc<Arithmetic>("inspect", { left, right, bias });
export const replaceWeights = (weights: Weights) =>
  rpc<boolean>("replace", weights);
export const downloadJSON = (value: unknown, filename: string) => {
  const url = URL.createObjectURL(
      new Blob([JSON.stringify(value)], { type: "application/json" }),
    ),
    a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
