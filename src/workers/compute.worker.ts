/// <reference lib="webworker" />
import { MINI, DEFAULT_OPTIONS, type Config } from "../core/model/config";
import {
  createWeights,
  parameterCount,
  type Weights,
} from "../core/model/weights";
import { forward } from "../core/model/forward";
import { tokenize } from "../core/tokenizer/bpe";
import { VOCAB, pairKey } from "../core/tokenizer/merges";
import { decodeBytes } from "../core/tokenizer/byteEncoder";
import { sample } from "../core/sample";
import type { RunRequest, RunResult, SerializedModel } from "../state/store";
import { validateModel } from "../core/model/load";
import { cosine } from "../core/ops";
let config: Config = { ...MINI },
  weights: Weights = createWeights(),
  vocab = [...VOCAB],
  track = "mini",
  tokenizer: SerializedModel["tokenizer"];
function run(r: RunRequest): RunResult {
  const start = performance.now(),
    external = tokenizer
      ? {
          vocab: tokenizer.vocab,
          ranks: new Map(
            tokenizer.merges.map(([a, b], i) => [pairKey(a, b), i]),
          ),
          pattern: tokenizer.pattern,
        }
      : undefined;
  const tk = tokenize(r.text, r.bpe, r.choices, external),
    all = r.ids ?? (tk.ids.length ? tk.ids : [config.bosId ?? 1]);
  const limit = Math.max(1, Math.min(config.maxSeq, r.maxSeq)),
    ids = all.slice(-limit),
    truncated = Math.max(0, all.length - limit);
  const options = {
    ...r.options,
    ffWidth: Math.min(config.ff, r.options.ffWidth),
  };
  const trace = forward(ids, config, weights, options, r.patch);
  const baseline = r.patch
    ? forward(ids, config, weights, options, undefined, false)
    : undefined;
  const last = trace.logits.slice(-config.vocabSize),
    selection = sample(last, r.sampling, r.step ?? 0);
  const baseSample = baseline
    ? sample(baseline.logits.slice(-config.vocabSize), r.sampling, r.step ?? 0)
    : undefined;
  const delta = baseSample
    ? selection.probs.map((v, i) => v - baseSample.probs[i])
    : undefined;
  let affected = 0;
  if (baseline)
    for (let p = 0; p < ids.length; p++)
      if (
        trace.probabilities
          .subarray(p * config.vocabSize, (p + 1) * config.vocabSize)
          .some(
            (v, j) =>
              Math.abs(v - baseline.probabilities[p * config.vocabSize + j]) >
              1e-6,
          )
      )
        affected++;
  const other = forward(
    ids,
    config,
    weights,
    { ...options, normOrder: options.normOrder === "pre" ? "post" : "pre" },
    undefined,
    false,
  );
  return {
    tokenization: tk,
    trace,
    sample: selection,
    vocab,
    parameterCount: parameterCount(weights),
    elapsed: performance.now() - start,
    truncated,
    track,
    baseline: baseSample?.probs,
    delta,
    affected,
    contextStart: truncated,
    normComparison: {
      pre:
        options.normOrder === "pre"
          ? trace.probabilities.slice(-config.vocabSize)
          : other.probabilities.slice(-config.vocabSize),
      post:
        options.normOrder === "post"
          ? trace.probabilities.slice(-config.vocabSize)
          : other.probabilities.slice(-config.vocabSize),
    },
  };
}
self.onmessage = async (e: MessageEvent) => {
  const { id, type, payload } = e.data;
  try {
    let result: unknown;
    if (type === "run") result = run(payload);
    else if (type === "inspect") {
      const a = payload.left as Float32Array,
        b = payload.right as Float32Array;
      if (a.length !== b.length) throw new Error("乘加向量长度不一致");
      let sum = 0;
      const products = a.map((v, i) => {
        const p = v * b[i];
        sum += p;
        return p;
      });
      result = {
        left: a,
        right: b,
        products,
        sum,
        bias: payload.bias,
        result: Math.fround(Math.fround(sum) + payload.bias),
      };
    } else if (type === "similarity") {
      const row = Math.max(
          0,
          Math.min(config.vocabSize - 1, Number(payload.row)),
        ),
        d = config.d,
        a = weights.embedding.subarray(row * d, (row + 1) * d);
      const scores = Float32Array.from({ length: config.vocabSize }, (_, i) =>
        cosine(a, weights.embedding.subarray(i * d, (i + 1) * d)),
      );
      result = {
        scores,
        order: [...scores.keys()]
          .filter((i) => i !== row)
          .sort((a, b) => scores[b] - scores[a] || a - b),
      };
    } else if (type === "reset") {
      config = { ...MINI };
      weights = createWeights();
      vocab = [...VOCAB];
      tokenizer = undefined;
      track = "mini";
      result = true;
    } else if (type === "edit") {
      const { key, index, value } = payload;
      if (
        !weights[key] ||
        !Number.isInteger(index) ||
        index < 0 ||
        index >= weights[key].length ||
        !Number.isFinite(value)
      )
        throw new Error("参数修改无效");
      weights[key][index] = value;
      result = true;
    } else if (type === "snapshot") result = { config, weights, vocab, track };
    else if (type === "replace") {
      weights = payload;
      config = { ...MINI };
      vocab = [...VOCAB];
      tokenizer = undefined;
      track = "trained";
      result = true;
    } else if (type === "export")
      result = {
        format: "token-by-token-v1",
        config,
        weights: Object.fromEntries(
          Object.entries(weights).map(([k, v]) => [k, Array.from(v)]),
        ),
        vocab,
        tokenizer,
        source: track,
        options: payload ?? DEFAULT_OPTIONS,
      };
    else if (type === "import") {
      const model = validateModel(payload);
      config = model.config;
      weights = model.weights;
      tokenizer = model.tokenizer;
      vocab = tokenizer
        ? tokenizer.vocab.map((s: string) => {
            try {
              return decodeBytes(s);
            } catch {
              return s;
            }
          })
        : model.vocab;
      track = model.source === "trained" ? "trained" : "real";
      result = { config, options: model.options };
    } else if (type === "loadReal") {
      const controller = new AbortController(),
        timer = setTimeout(() => controller.abort(), 15000);
      try {
        const base = payload.base as string;
        const [cr, wr] = await Promise.all([
          fetch(`${base}config.json`, { signal: controller.signal }),
          fetch(`${base}weights.json`, { signal: controller.signal }),
        ]);
        if (
          !cr.ok ||
          !wr.ok ||
          !cr.headers.get("content-type")?.includes("json") ||
          !wr.headers.get("content-type")?.includes("json")
        )
          throw new Error(
            "没有找到预训练文件。请将 config.json 和 weights.json 放入 public/models/，或导入导出的 JSON。微型模型和浏览器训练仍可完整使用。",
          );
        const meta = await cr.json(),
          values = await wr.json();
        const model = validateModel({ ...meta, weights: values });
        config = model.config;
        weights = model.weights;
        tokenizer = model.tokenizer;
        vocab = tokenizer
          ? tokenizer.vocab.map((s: string) => {
              try {
                return decodeBytes(s);
              } catch {
                return s;
              }
            })
          : model.vocab;
        track = "real";
        result = { config, options: model.options };
      } finally {
        clearTimeout(timer);
      }
    } else throw new Error("未知计算请求");
    self.postMessage({ id, result });
  } catch (error) {
    self.postMessage({
      id,
      error:
        error instanceof Error && error.name === "AbortError"
          ? "预训练模型读取超时（15 秒），已保留当前模型。"
          : error instanceof Error
            ? error.message
            : String(error),
    });
  }
};
