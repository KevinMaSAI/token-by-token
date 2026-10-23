import { activate, biasAdd, matmul } from "../ops";
import type { Config, Options } from "./config";
import type { Weights } from "./weights";
import type { Tap } from "./attention";
export interface FFNCache {
  input: Float32Array;
  linear: Float32Array;
  pre: Float32Array;
  gate: Float32Array;
  activated: Float32Array;
  gated: Float32Array;
  projected: Float32Array;
  output: Float32Array;
}
export function ffn(
  x: Float32Array,
  seq: number,
  c: Config,
  w: Weights,
  l: number,
  o: Options,
  tap: Tap,
): FFNCache {
  const p = `L${l}`,
    d = c.d,
    f = c.ff;
  const linear = tap(
    `${p}.ffLinear`,
    matmul(x, w[`${p}.w1`], seq, d, f),
    seq,
    f,
    "FFN 第一次矩阵乘法",
  );
  const pre = tap(
    `${p}.ffPre`,
    biasAdd(linear, w[`${p}.b1`]),
    seq,
    f,
    "激活前",
  );
  const gate = tap(
    `${p}.ffGate`,
    biasAdd(matmul(x, w[`${p}.gate`], seq, d, f), w[`${p}.gateBias`]),
    seq,
    f,
    "SwiGLU 门",
  );
  const activated = tap(
    `${p}.ffActivated`,
    pre.map((v) => activate(v, o.activation)),
    seq,
    f,
    "激活后",
  );
  const gated = tap(
    `${p}.ffGated`,
    activated.map((v, i) =>
      i % f >= o.ffWidth ? 0 : o.activation === "swiglu" ? v * gate[i] : v,
    ),
    seq,
    f,
    "有效隐藏维度",
  );
  const projected = tap(
    `${p}.ffProjected`,
    matmul(gated, w[`${p}.w2`], seq, f, d),
    seq,
    d,
    "FFN 第二次矩阵乘法",
  );
  const output = tap(
    `${p}.ffOutput`,
    biasAdd(projected, w[`${p}.b2`]),
    seq,
    d,
    "FFN 输出",
  );
  return { input: x, linear, pre, gate, activated, gated, projected, output };
}
