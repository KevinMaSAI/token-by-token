import {
  biasAdd,
  matmul,
  rope,
  softmaxDetails,
  zeros,
  type softmax,
} from "../ops";
import type { Config, Options } from "./config";
import { random, type Weights } from "./weights";
export type Tap = (
  key: string,
  x: Float32Array,
  rows: number,
  cols: number,
  label?: string,
) => Float32Array;
export interface HeadCache {
  scores: Float32Array;
  scaled: Float32Array;
  shifted: Float32Array;
  exp: Float32Array;
  sums: Float32Array;
  max: Float32Array;
  probs: ReturnType<typeof softmax>;
  used: Float32Array;
  keep: Float32Array;
  output: Float32Array;
}
export interface AttentionCache {
  input: Float32Array;
  qRaw: Float32Array;
  kRaw: Float32Array;
  v: Float32Array;
  q: Float32Array;
  k: Float32Array;
  heads: HeadCache[];
  concat: Float32Array;
  projected: Float32Array;
  output: Float32Array;
}
export function attention(
  x: Float32Array,
  seq: number,
  c: Config,
  w: Weights,
  l: number,
  o: Options,
  tap: Tap,
): AttentionCache {
  const p = `L${l}`,
    d = c.d,
    hd = d / c.heads;
  const qRaw = tap(
    `${p}.qRaw`,
    biasAdd(matmul(x, w[`${p}.q`], seq, d, d), w[`${p}.qBias`]),
    seq,
    d,
    "查询投影 Q",
  );
  const kRaw = tap(
    `${p}.kRaw`,
    biasAdd(matmul(x, w[`${p}.k`], seq, d, d), w[`${p}.kBias`]),
    seq,
    d,
    "键投影 K",
  );
  const v = tap(
    `${p}.vRaw`,
    biasAdd(matmul(x, w[`${p}.v`], seq, d, d), w[`${p}.vBias`]),
    seq,
    d,
    "值投影 V",
  );
  const q = tap(
    `${p}.qRot`,
    o.position === "rope" ? rope(qRaw, seq, c.heads, hd) : qRaw.slice(),
    seq,
    d,
    "位置后的 Q",
  );
  const k = tap(
    `${p}.kRot`,
    o.position === "rope" ? rope(kRaw, seq, c.heads, hd) : kRaw.slice(),
    seq,
    d,
    "位置后的 K",
  );
  const heads: HeadCache[] = [],
    concat = zeros(seq * d),
    rng = random(718 + l);
  for (let h = 0; h < c.heads; h++) {
    const hp = `${p}.H${h}`,
      raw = zeros(seq * seq);
    for (let i = 0; i < seq; i++)
      for (let j = 0; j < seq; j++) {
        let s = 0;
        for (let a = 0; a < hd; a++)
          s += q[i * d + h * hd + a] * k[j * d + h * hd + a];
        raw[i * seq + j] = s;
      }
    const scores = tap(`${hp}.scores`, raw, seq, seq, "Q·K 原始相关分数");
    const scaled = tap(
      `${hp}.scaled`,
      scores.map(
        (v) => v / (o.scaled ? Math.sqrt(hd) : 1) / o.attentionTemperature,
      ),
      seq,
      seq,
      "缩放后的分数",
    );
    const probs = zeros(seq * seq),
      shifted = zeros(seq * seq),
      exp = zeros(seq * seq),
      sums = zeros(seq),
      max = zeros(seq);
    const masked = tap(
      `${hp}.masked`,
      scaled.map((v, index) =>
        o.causal && index % seq > Math.floor(index / seq) ? -Infinity : v,
      ),
      seq,
      seq,
      "因果遮罩后的分数",
    );
    for (let i = 0; i < seq; i++) {
      const sm = softmaxDetails(masked.subarray(i * seq, (i + 1) * seq));
      max[i] = sm.max;
    }
    tap(`${hp}.max`, max, seq, 1, "每行最大值");
    for (let i = 0; i < seq; i++)
      for (let j = 0; j < seq; j++)
        shifted[i * seq + j] = masked[i * seq + j] - max[i];
    tap(`${hp}.shifted`, shifted, seq, seq, "减去最大值");
    for (let i = 0; i < exp.length; i++) exp[i] = Math.exp(shifted[i]);
    tap(`${hp}.exp`, exp, seq, seq, "指数");
    for (let i = 0; i < seq; i++) {
      let sum = 0;
      for (let j = 0; j < seq; j++) sum += exp[i * seq + j];
      sums[i] = sum;
    }
    tap(`${hp}.sums`, sums, seq, 1, "指数和");
    for (let i = 0; i < seq; i++)
      for (let j = 0; j < seq; j++)
        probs[i * seq + j] = sums[i] ? exp[i * seq + j] / sums[i] : 0;
    tap(`${hp}.probs`, probs, seq, seq, "注意力权重");
    const keep = tap(
      `${hp}.keep`,
      probs.map(() =>
        o.dropout ? (rng() >= o.dropout ? 1 / (1 - o.dropout) : 0) : 1,
      ),
      seq,
      seq,
      "演示丢弃掩码",
    );
    const used = tap(
      `${hp}.used`,
      probs.map((v, i) => v * keep[i]),
      seq,
      seq,
      "演示丢弃后的权重",
    );
    const output = zeros(seq * hd);
    for (let i = 0; i < seq; i++)
      for (let a = 0; a < hd; a++) {
        let s = 0;
        for (let j = 0; j < seq; j++)
          s += used[i * seq + j] * v[j * d + h * hd + a];
        output[i * hd + a] = s;
      }
    tap(`${hp}.out`, output, seq, hd, "加权求和");
    for (let i = 0; i < seq; i++)
      concat.set(output.subarray(i * hd, (i + 1) * hd), i * d + h * hd);
    heads.push({
      scores,
      scaled,
      shifted,
      exp,
      sums,
      max,
      probs,
      used,
      keep,
      output,
    });
  }
  tap(`${p}.concat`, concat, seq, d, "多头拼接");
  const projected = tap(
    `${p}.projected`,
    matmul(concat, w[`${p}.o`], seq, d, d),
    seq,
    d,
    "输出矩阵乘法",
  );
  const output = tap(
    `${p}.attention`,
    biasAdd(projected, w[`${p}.oBias`]),
    seq,
    d,
    "注意力输出",
  );
  return { input: x, qRaw, kRaw, v, q, k, heads, concat, projected, output };
}
