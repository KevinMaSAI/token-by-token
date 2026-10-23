import {
  add,
  zeros,
  matmulBackward,
  biasGrad,
  normBackward,
  activationDerivative,
  rope,
  softmaxBackward,
} from "../ops";
import { forward, type Forward } from "./forward";
import type { Config, Options } from "./config";
import { type Weights, random } from "./weights";
import type { FFNCache } from "./ffn";
import type { AttentionCache } from "./attention";
import { tokenize, DEFAULT_BPE } from "../tokenizer/bpe";

/** Average cross entropy: L = mean_i(logsumexp(logits_i)-logits_i[target_i]). */
export function crossEntropy(f: Forward, targets: number[]) {
  const v = f.c.vocabSize,
    n = targets.length;
  let loss = 0;
  for (let i = 0; i < n; i++) {
    let max = -Infinity;
    for (let j = 0; j < v; j++) max = Math.max(max, f.logits[i * v + j]);
    let sum = 0;
    for (let j = 0; j < v; j++) sum += Math.exp(f.logits[i * v + j] - max);
    loss += (Math.log(sum) + max - f.logits[i * v + targets[i]]) / n;
  }
  return loss;
}

/** Explicit reverse pass through the concrete Transformer. No graph, tape or autodiff. */
export function backward(f: Forward, w: Weights, targets: number[]): Weights {
  const c = f.c,
    o = f.options,
    n = f.ids.length,
    d = c.d,
    v = c.vocabSize,
    g: Weights = Object.fromEntries(
      Object.entries(w).map(([k, a]) => [k, zeros(a.length)]),
    );
  if (targets.length !== n) throw new Error("监督标签与序列长度不一致");
  const dl = f.probabilities.map(
    (p, i) => (p - (i % v === targets[Math.floor(i / v)] ? 1 : 0)) / n,
  );
  const head = matmulBackward(f.final.output, w.lm, dl, n, d, v);
  g.lm = head.db;
  g.lmBias = biasGrad(dl, v);
  const fn = normBackward(f.final, head.da, w["final.gamma"]);
  g["final.gamma"] = fn.dg;
  g["final.beta"] = fn.db;
  let dx = fn.dx;
  const normBack = (
    cache: Forward["final"] | Forward["layers"][number]["n1"],
    grad: Float32Array,
    p: string,
  ) => {
    const b = normBackward(cache, grad, w[`${p}.gamma`]);
    g[`${p}.gamma`] = b.dg;
    g[`${p}.beta`] = b.db;
    return b.dx;
  };
  const ffBack = (a: FFNCache, dy: Float32Array, l: number) => {
    const p = `L${l}`,
      ff = c.ff,
      b2 = matmulBackward(a.gated, w[`${p}.w2`], dy, n, ff, d);
    g[`${p}.w2`] = b2.db;
    g[`${p}.b2`] = biasGrad(dy, d);
    const dpre = zeros(a.pre.length),
      dgate = zeros(a.gate.length);
    for (let i = 0; i < dpre.length; i++)
      if (i % ff < o.ffWidth) {
        dpre[i] =
          b2.da[i] *
          activationDerivative(a.pre[i], o.activation) *
          (o.activation === "swiglu" ? a.gate[i] : 1);
        if (o.activation === "swiglu") dgate[i] = b2.da[i] * a.activated[i];
      }
    const b1 = matmulBackward(a.input, w[`${p}.w1`], dpre, n, d, ff);
    g[`${p}.w1`] = b1.db;
    g[`${p}.b1`] = biasGrad(dpre, ff);
    if (o.activation === "swiglu") {
      const gate = matmulBackward(a.input, w[`${p}.gate`], dgate, n, d, ff);
      g[`${p}.gate`] = gate.db;
      g[`${p}.gateBias`] = biasGrad(dgate, ff);
      return add(b1.da, gate.da);
    }
    return b1.da;
  };
  const attBack = (a: AttentionCache, dy: Float32Array, l: number) => {
    const p = `L${l}`,
      hd = d / c.heads,
      bo = matmulBackward(a.concat, w[`${p}.o`], dy, n, d, d);
    g[`${p}.o`] = bo.db;
    g[`${p}.oBias`] = biasGrad(dy, d);
    let dq = zeros(n * d),
      dk = zeros(n * d);
    const dv = zeros(n * d),
      factor = 1 / (o.scaled ? Math.sqrt(hd) : 1) / o.attentionTemperature;
    for (let h = 0; h < c.heads; h++) {
      const cache = a.heads[h],
        dp = zeros(n * n);
      // O_i = sum_j P_ij V_j; dP_ij = dot(dO_i,V_j), dV_j = sum_i P_ij dO_i.
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) {
          let sum = 0;
          for (let k = 0; k < hd; k++) {
            sum += bo.da[i * d + h * hd + k] * a.v[j * d + h * hd + k];
            dv[j * d + h * hd + k] +=
              cache.used[i * n + j] * bo.da[i * d + h * hd + k];
          }
          dp[i * n + j] = sum * cache.keep[i * n + j];
        }
      for (let i = 0; i < n; i++) {
        const ds = softmaxBackward(
          cache.probs.subarray(i * n, (i + 1) * n),
          dp.subarray(i * n, (i + 1) * n),
        );
        for (let j = 0; j < n; j++) {
          const s = (o.causal && j > i ? 0 : ds[j]) * factor;
          for (let k = 0; k < hd; k++) {
            dq[i * d + h * hd + k] += s * a.k[j * d + h * hd + k];
            dk[j * d + h * hd + k] += s * a.q[i * d + h * hd + k];
          }
        }
      }
    }
    // A rotation's transpose is its inverse.
    if (o.position === "rope") {
      dq = rope(dq, n, c.heads, hd, true);
      dk = rope(dk, n, c.heads, hd, true);
    }
    let input = zeros(n * d);
    for (const [name, grad] of [
      ["q", dq],
      ["k", dk],
      ["v", dv],
    ] as const) {
      const b = matmulBackward(a.input, w[`${p}.${name}`], grad, n, d, d);
      g[`${p}.${name}`] = b.db;
      g[`${p}.${name}Bias`] = biasGrad(grad, d);
      input = add(input, b.da);
    }
    return input;
  };
  for (let l = c.layers - 1; l >= 0; l--) {
    const a = f.layers[l],
      p = `L${l}`;
    if (o.normOrder === "pre") {
      const dff = ffBack(a.ff, dx, l),
        dm = add(dx, normBack(a.n2, dff, `${p}.n2`));
      const datt = attBack(a.att, dm, l);
      dx = add(dm, normBack(a.n1, datt, `${p}.n1`));
    } else {
      const dr2 = normBack(a.n2, dx, `${p}.n2`),
        dm = add(dr2, ffBack(a.ff, dr2, l));
      const dr1 = normBack(a.n1, dm, `${p}.n1`);
      dx = add(dr1, attBack(a.att, dr1, l));
    }
  }
  for (let i = 0; i < n; i++)
    for (let j = 0; j < d; j++) {
      g.embedding[f.ids[i] * d + j] += dx[i * d + j];
      if (o.position === "learned") g.position[i * d + j] += dx[i * d + j];
    }
  return g;
}
export class Adam {
  private m: Weights = {};
  private v: Weights = {};
  private step = 0;
  constructor(w: Weights) {
    for (const [k, x] of Object.entries(w)) {
      this.m[k] = zeros(x.length);
      this.v[k] = zeros(x.length);
    }
  }
  /** clip g by global norm; m=.9m+.1g; v=.999v+.001g^2; w -= lr*mhat/(sqrt(vhat)+eps). */
  update(w: Weights, g: Weights, lr: number, clip = 1) {
    let norm2 = 0;
    for (const a of Object.values(g)) for (const x of a) norm2 += x * x;
    const gradNorm = Math.sqrt(norm2);
    if (!Number.isFinite(gradNorm))
      throw new Error("梯度非有限，训练已停止，请降低学习率");
    const factor = Math.min(1, clip / (gradNorm || 1));
    this.step++;
    const b1 = 1 - 0.9 ** this.step,
      b2 = 1 - 0.999 ** this.step;
    for (const [key, a] of Object.entries(w))
      for (let i = 0; i < a.length; i++) {
        const x = g[key][i] * factor;
        this.m[key][i] = 0.9 * this.m[key][i] + 0.1 * x;
        this.v[key][i] = 0.999 * this.v[key][i] + 0.001 * x * x;
        a[i] -=
          (lr * (this.m[key][i] / b1)) /
          (Math.sqrt(this.v[key][i] / b2) + 1e-8);
      }
    return { gradNorm, clippedNorm: gradNorm * factor };
  }
}
export const CORPUS = [
  "你好 我爱学习.",
  "你爱学习 我爱学习.",
  "人工智能 学习中文.",
  "今天天气好.".replace("今", ""),
  "我爱中文 你好.",
  "他爱学习 你好.",
  "hello the cat is good.",
  "the cat is a cat.",
  "hello the dog is good.",
  "the dog is a dog.",
  "0 1 2 3 4 5 6 7 8 9.",
  "0123456789 0123456789.",
  "123 234 345 456.",
  "你好 世界 hello.",
];
export function trainingData(maxSeq = 24) {
  return CORPUS.map((text) => {
    const ids = [1, ...tokenize(text, DEFAULT_BPE).ids, 2];
    return {
      input: ids.slice(0, Math.min(maxSeq, ids.length - 1)),
      target: ids.slice(1, Math.min(maxSeq, ids.length - 1) + 1),
      text,
    };
  });
}
export function trainStep(
  w: Weights,
  c: Config,
  o: Options,
  adam: Adam,
  data: { input: number[]; target: number[] },
  lr: number,
  clip: number,
) {
  const f = forward(data.input, c, w, o, undefined, false),
    loss = crossEntropy(f, data.target),
    grads = backward(f, w, data.target),
    metrics = adam.update(w, grads, lr, clip);
  return { loss, ...metrics };
}
export function trainerPicker(seed = 42) {
  const r = random(seed);
  return (n: number) => Math.floor(r() * n);
}
