import { biasAdd, matmul, softmax, cosine, norm } from "../ops";
import { normalization } from "./normalization";
import type { Config, Options, Patch, TraceTensor } from "./config";
import { weightShapes, cloneWeights, type Weights } from "./weights";
import { embed, positionInput } from "./embedding";
import { layer, type LayerCache } from "./layer";
import type { Tap } from "./attention";
export function forward(
  ids: number[],
  c: Config,
  original: Weights,
  o: Options,
  patch?: Patch,
  collect = true,
) {
  if (!ids.length || ids.length > c.maxSeq)
    throw new Error("输入序列长度不在模型支持范围内");
  const w = patch?.key.startsWith("W:") ? cloneWeights(original) : original;
  if (patch?.key.startsWith("W:")) {
    const v = w[patch.key.slice(2)];
    if (
      !v ||
      patch.index < 0 ||
      patch.index >= v.length ||
      !Number.isFinite(patch.value)
    )
      throw new Error("无效的权重干预");
    v[patch.index] =
      patch.mode === "add" ? v[patch.index] + patch.value : patch.value;
  }
  const tensors: TraceTensor[] = [];
  if (collect)
    for (const [key, shape] of Object.entries(weightShapes(c)))
      tensors.push({
        key: `W:${key}`,
        label: key,
        shape,
        data: w[key],
        kind: "weight",
      });
  const tap: Tap = (key, x, rows, cols, label = key) => {
    if (patch?.key === key) {
      if (
        patch.index < 0 ||
        patch.index >= x.length ||
        !Number.isFinite(patch.value)
      )
        throw new Error("无效的张量干预");
      x[patch.index] =
        patch.mode === "add" ? x[patch.index] + patch.value : patch.value;
    }
    if (collect)
      tensors.push({
        key,
        label,
        shape: [rows, cols],
        data: x,
        kind: "activation",
      });
    return x;
  };
  const seq = ids.length,
    d = c.d;
  const embedding = tap("embedding", embed(ids, c, w), seq, d, "嵌入查表");
  const pos = positionInput(embedding, seq, c, w, o);
  tap("positional", pos.positional, seq, d, "位置向量");
  // Recompute addition after the positional intervention.
  const initial = tap(
    "initial",
    embedding.map((v, i) => v + pos.positional[i]),
    seq,
    d,
    "初始残差流",
  );
  let x = initial;
  const layers: LayerCache[] = [];
  for (let l = 0; l < c.layers; l++) {
    const cache = layer(x, seq, c, w, l, o, tap);
    layers.push(cache);
    x = cache.output.slice();
  }
  const final = normalization(
    x,
    w["final.gamma"],
    w["final.beta"],
    o.eps,
    o.norm,
    tap,
    "final",
  );
  const logits = tap(
    "logits",
    biasAdd(matmul(final.output, w.lm, seq, d, c.vocabSize), w.lmBias),
    seq,
    c.vocabSize,
    "词表分数",
  );
  const probabilities = tap(
    "probabilities",
    Float32Array.from({ length: seq * c.vocabSize }, () => 0),
    seq,
    c.vocabSize,
    "全部位置预测概率",
  );
  for (let i = 0; i < seq; i++)
    probabilities.set(
      softmax(logits.subarray(i * c.vocabSize, (i + 1) * c.vocabSize)),
      i * c.vocabSize,
    );
  if (patch?.key === "probabilities")
    probabilities[patch.index] =
      patch.mode === "add"
        ? probabilities[patch.index] + patch.value
        : patch.value;
  const similarities =
    collect && c.vocabSize <= 512
      ? Array.from({ length: c.vocabSize }, (_, i) =>
          Float32Array.from({ length: c.vocabSize }, (_, j) =>
            cosine(
              w.embedding.subarray(i * d, (i + 1) * d),
              w.embedding.subarray(j * d, (j + 1) * d),
            ),
          ),
        )
      : [];
  const headSimilarities = layers.map((l) =>
    Float32Array.from({ length: c.heads * c.heads }, (_, i) =>
      cosine(
        l.att.heads[Math.floor(i / c.heads)].probs,
        l.att.heads[i % c.heads].probs,
      ),
    ),
  );
  return {
    ids,
    c,
    options: o,
    embedding,
    positional: pos.positional,
    initial,
    layers,
    final,
    logits,
    probabilities,
    tensors,
    similarities,
    headSimilarities,
    activationNorms: layers.map((l) => ({
      input: norm(l.input),
      attention: norm(l.att.output),
      ffn: norm(l.ff.output),
      output: norm(l.output),
      delta: norm(l.delta),
    })),
  };
}
export type Forward = ReturnType<typeof forward>;
