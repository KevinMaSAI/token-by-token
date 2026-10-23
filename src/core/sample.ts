import { softmaxDetails, zeros } from "./ops";
import { random } from "./model/weights";
export interface Sampling {
  method: "greedy" | "temperature" | "top-k" | "top-p";
  temperature: number;
  k: number;
  p: number;
  seed: number;
}
export const DEFAULT_SAMPLING: Sampling = {
  method: "temperature",
  temperature: 1,
  k: 5,
  p: 0.9,
  seed: 42,
};
export function topK(p: Float32Array, k: number) {
  const ids = [...p.keys()]
    .sort((a, b) => p[b] - p[a] || a - b)
    .slice(0, Math.max(1, Math.floor(k)));
  return renormalize(p, ids);
}
export function topP(p: Float32Array, threshold: number) {
  const sorted = [...p.keys()].sort((a, b) => p[b] - p[a] || a - b),
    ids: number[] = [];
  let sum = 0;
  for (const id of sorted) {
    ids.push(id);
    sum += p[id];
    if (sum >= threshold) break;
  }
  return renormalize(p, ids);
}
function renormalize(p: Float32Array, ids: number[]) {
  const out = zeros(p.length),
    sum = ids.reduce((s, i) => s + p[i], 0);
  ids.forEach((i) => (out[i] = sum ? p[i] / sum : 1 / ids.length));
  return out;
}
/** Inverse-CDF sampling: first i for which sum_{j<=i} p_j > u, u in [0,1). */
export function sample(logits: Float32Array, s: Sampling, step = 0) {
  const details = softmaxDetails(logits, s.temperature),
    order = [...logits.keys()].sort((a, b) => logits[b] - logits[a] || a - b);
  let filtered = details.probs.slice();
  if (s.method === "top-k") filtered = topK(filtered, s.k);
  if (s.method === "top-p") filtered = topP(filtered, s.p);
  if (s.method === "greedy") {
    filtered.fill(0);
    filtered[order[0]] = 1;
  }
  const cumulative = zeros(logits.length);
  let sum = 0;
  order.forEach((i) => {
    sum += details.probs[i];
    cumulative[i] = sum;
  });
  const draw = random((s.seed + Math.imul(step, 0x9e3779b9)) >>> 0)();
  let running = 0,
    selected = order[0];
  for (let i = 0; i < filtered.length; i++) {
    running += filtered[i];
    if (draw < running) {
      selected = i;
      break;
    }
  }
  return {
    ...details,
    filtered,
    order,
    cumulative,
    draw,
    selected,
    step,
    seed: s.seed,
    method: s.method,
    comparisons: [0.1, 0.5, 1, 2].map((t) => ({
      t,
      probs: softmaxDetails(logits, t).probs,
    })),
  };
}
export type SampleResult = ReturnType<typeof sample>;
