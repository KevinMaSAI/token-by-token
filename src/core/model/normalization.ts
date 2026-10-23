import { normalize, zeros, type NormKind } from "../ops";
import type { Tap } from "./attention";
/** Editable normalization checkpoints are consumed by every downstream calculation. */
export function normalization(
  x: Float32Array,
  gamma: Float32Array,
  beta: Float32Array,
  eps: number,
  kind: NormKind,
  tap: Tap,
  prefix: string,
) {
  const c = normalize(x, gamma, beta, eps, kind),
    w = gamma.length,
    rows = x.length / w;
  tap(`${prefix}.mean`, c.mean, rows, 1, "均值");
  const centered = tap(
    `${prefix}.centered`,
    x.map((v, i) => v - c.mean[Math.floor(i / w)]),
    rows,
    w,
    "减去均值",
  );
  const squares = tap(
    `${prefix}.squares`,
    centered.map((v) => v * v),
    rows,
    w,
    "偏差平方",
  );
  for (let r = 0; r < rows; r++) {
    let sum = 0;
    for (let j = 0; j < w; j++) sum += squares[r * w + j];
    c.variance[r] = sum / w;
  }
  tap(`${prefix}.variance`, c.variance, rows, 1, "方差或均方");
  for (let r = 0; r < rows; r++)
    c.inv[r] = 1 / Math.sqrt(Math.max(0, c.variance[r]) + eps);
  tap(`${prefix}.inv`, c.inv, rows, 1, "标准差的倒数");
  c.normalized = tap(
    `${prefix}.normalized`,
    centered.map((v, i) => v * c.inv[Math.floor(i / w)]),
    rows,
    w,
    "标准化数值",
  );
  c.output = tap(
    `${prefix}.output`,
    c.normalized.map(
      (v, i) => v * gamma[i % w] + (kind === "layer" ? beta[i % w] : 0),
    ),
    rows,
    w,
    "归一化输出",
  );
  return {
    ...c,
    centered,
    squares,
    sums: Float32Array.from({ length: rows }, (_, r) =>
      x.slice(r * w, (r + 1) * w).reduce((a, b) => a + b, 0),
    ),
    squareSums: Float32Array.from({ length: rows }, (_, r) =>
      squares.slice(r * w, (r + 1) * w).reduce((a, b) => a + b, 0),
    ),
    eps,
  };
}
export const noTap: Tap = (_k, x) => x;
export const zeroBias = (n: number) => zeros(n);
