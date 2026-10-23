import { add, sinusoidal, zeros } from "../ops";
import type { Config, Options } from "./config";
import type { Weights } from "./weights";
export function embed(ids: number[], c: Config, w: Weights) {
  const x = zeros(ids.length * c.d);
  ids.forEach((id, p) => {
    if (!Number.isInteger(id) || id < 0 || id >= c.vocabSize)
      throw new Error("词表索引越界");
    x.set(w.embedding.subarray(id * c.d, (id + 1) * c.d), p * c.d);
  });
  return x;
}
export function positionInput(
  x: Float32Array,
  seq: number,
  c: Config,
  w: Weights,
  o: Options,
) {
  const pos =
    o.position === "sinusoidal"
      ? sinusoidal(seq, c.d)
      : o.position === "learned"
        ? w.position.slice(0, seq * c.d)
        : zeros(x.length);
  return { positional: pos, output: add(x, pos) };
}
