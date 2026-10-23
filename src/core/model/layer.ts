import { add } from "../ops";
import { normalization } from "./normalization";
import type { Config, Options } from "./config";
import type { Weights } from "./weights";
import { attention, type AttentionCache, type Tap } from "./attention";
import { ffn, type FFNCache } from "./ffn";
export interface LayerCache {
  input: Float32Array;
  n1: ReturnType<typeof normalization>;
  att: AttentionCache;
  residual1: Float32Array;
  middle: Float32Array;
  n2: ReturnType<typeof normalization>;
  ff: FFNCache;
  residual2: Float32Array;
  output: Float32Array;
  delta: Float32Array;
}
export function layer(
  x: Float32Array,
  seq: number,
  c: Config,
  w: Weights,
  l: number,
  o: Options,
  tap: Tap,
): LayerCache {
  const p = `L${l}`,
    d = c.d;
  tap(`${p}.input`, x, seq, d, "层输入");
  const n = (a: Float32Array, k: string) =>
    normalization(
      a,
      w[`${p}.${k}.gamma`],
      w[`${p}.${k}.beta`],
      o.eps,
      o.norm,
      tap,
      `${p}.${k}`,
    );
  let n1: ReturnType<typeof normalization>,
    n2: ReturnType<typeof normalization>,
    att: AttentionCache,
    ff: FFNCache,
    middle: Float32Array,
    output: Float32Array,
    residual1: Float32Array,
    residual2: Float32Array;
  if (o.normOrder === "pre") {
    n1 = n(x, "n1");
    att = attention(n1.output, seq, c, w, l, o, tap);
    residual1 = tap(
      `${p}.residual1`,
      add(x, att.output),
      seq,
      d,
      "注意力残差相加",
    );
    middle = residual1;
    n2 = n(middle, "n2");
    ff = ffn(n2.output, seq, c, w, l, o, tap);
    residual2 = tap(
      `${p}.residual2`,
      add(middle, ff.output),
      seq,
      d,
      "FFN 残差相加",
    );
    output = residual2;
  } else {
    att = attention(x, seq, c, w, l, o, tap);
    residual1 = tap(
      `${p}.residual1`,
      add(x, att.output),
      seq,
      d,
      "注意力残差相加",
    );
    n1 = n(residual1, "n1");
    middle = n1.output;
    ff = ffn(middle, seq, c, w, l, o, tap);
    residual2 = tap(
      `${p}.residual2`,
      add(middle, ff.output),
      seq,
      d,
      "FFN 残差相加",
    );
    n2 = n(residual2, "n2");
    output = n2.output;
  }
  // Copy aliases before editable taps so one intervention cannot retroactively change its source.
  output = tap(`${p}.output`, output.slice(), seq, d, "层输出");
  const delta = tap(
    `${p}.delta`,
    output.map((v, i) => v - x[i]),
    seq,
    d,
    "本层对残差流的贡献",
  );
  return {
    input: x,
    n1,
    att,
    residual1,
    middle,
    n2,
    ff,
    residual2,
    output,
    delta,
  };
}
