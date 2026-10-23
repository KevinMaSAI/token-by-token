import { MINI, type Config } from "./config";
import { zeros } from "../ops";
export type Weights = Record<string, Float32Array>;
/** Mulberry32 deterministic PRNG. */
export function random(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function weightShapes(c: Config): Record<string, [number, number]> {
  const s: Record<string, [number, number]> = {
    embedding: [c.vocabSize, c.d],
    position: [c.maxSeq, c.d],
    "final.gamma": [1, c.d],
    "final.beta": [1, c.d],
    lm: [c.d, c.vocabSize],
    lmBias: [1, c.vocabSize],
  };
  for (let l = 0; l < c.layers; l++) {
    for (const k of ["q", "k", "v", "o"]) {
      s[`L${l}.${k}`] = [c.d, c.d];
      s[`L${l}.${k}Bias`] = [1, c.d];
    }
    for (const k of ["n1", "n2"]) {
      s[`L${l}.${k}.gamma`] = [1, c.d];
      s[`L${l}.${k}.beta`] = [1, c.d];
    }
    s[`L${l}.w1`] = [c.d, c.ff];
    s[`L${l}.b1`] = [1, c.ff];
    s[`L${l}.gate`] = [c.d, c.ff];
    s[`L${l}.gateBias`] = [1, c.ff];
    s[`L${l}.w2`] = [c.ff, c.d];
    s[`L${l}.b2`] = [1, c.d];
  }
  return s;
}
export function createWeights(c: Config = MINI, seed = 42): Weights {
  const r = random(seed),
    w: Weights = {};
  for (const [key, [a, b]] of Object.entries(weightShapes(c))) {
    w[key] = zeros(a * b);
    if (key.endsWith("gamma")) w[key].fill(1);
    else if (key.endsWith("beta") || /Bias$|\.b[12]$/.test(key)) continue;
    else
      for (let i = 0; i < a * b; i++) {
        const u = r();
        w[key][i] = u < 0.58 ? 0 : u < 0.78 ? -0.125 : u < 0.96 ? 0.125 : 0.25;
      }
  }
  return w;
}
export const cloneWeights = (w: Weights): Weights =>
  Object.fromEntries(Object.entries(w).map(([k, v]) => [k, v.slice()]));
export const parameterCount = (w: Weights) =>
  Object.values(w).reduce((s, v) => s + v.length, 0);
