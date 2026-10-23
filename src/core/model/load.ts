import { weightShapes, type Weights } from "./weights";
import type { SerializedModel } from "../../state/store";
export function validateModel(raw: unknown) {
  if (!raw || typeof raw !== "object")
    throw new Error("模型文件不是 JSON 对象");
  const m = raw as SerializedModel,
    c = m.config;
  if (m.format !== "token-by-token-v1" || !c)
    throw new Error("不支持此模型格式，需要 token-by-token-v1");
  for (const k of [
    "vocabSize",
    "layers",
    "heads",
    "d",
    "ff",
    "maxSeq",
  ] as const)
    if (!Number.isInteger(c[k]) || c[k] < 1)
      throw new Error(`模型维度 ${k} 无效`);
  // Inspection bounds avoid accidentally freezing the browser with a multi-billion-parameter export.
  if (
    c.layers > 12 ||
    c.d > 128 ||
    c.ff > 512 ||
    c.vocabSize > 65536 ||
    c.maxSeq > 128 ||
    c.d % c.heads !== 0
  )
    throw new Error(
      "模型超过解剖台的内存上限（维度≤128、12层、上下文≤128），或维度不能整除头数",
    );
  if (
    !Array.isArray(m.vocab) ||
    m.vocab.length !== c.vocabSize ||
    m.vocab.some((v) => typeof v !== "string")
  )
    throw new Error("词表长度与配置不一致");
  for (const id of [c.bosId, c.eosId])
    if (
      id !== undefined &&
      (!Number.isInteger(id) || id < 0 || id >= c.vocabSize)
    )
      throw new Error("起始或结束 token ID 无效");
  if (m.options) {
    const o = m.options;
    if (
      !["rope", "sinusoidal", "learned", "none"].includes(o.position) ||
      !["layer", "rms"].includes(o.norm) ||
      !["pre", "post"].includes(o.normOrder) ||
      !["relu", "gelu", "silu", "swiglu"].includes(o.activation) ||
      !(o.eps > 0 && o.eps <= 1) ||
      !(
        o.attentionTemperature > 0 && Number.isFinite(o.attentionTemperature)
      ) ||
      !(o.dropout >= 0 && o.dropout < 1) ||
      !Number.isInteger(o.ffWidth) ||
      o.ffWidth < 1 ||
      o.ffWidth > c.ff ||
      typeof o.causal !== "boolean" ||
      typeof o.scaled !== "boolean"
    )
      throw new Error("模型运算选项无效");
  }
  const w: Weights = {};
  let count = 0;
  for (const [key, [a, b]] of Object.entries(weightShapes(c))) {
    const v = m.weights?.[key];
    count += a * b;
    if (count > 16_000_000) throw new Error("参数数量超过安全上限");
    if (
      !Array.isArray(v) ||
      v.length !== a * b ||
      v.some((x) => typeof x !== "number" || !Number.isFinite(x))
    )
      throw new Error(`权重 ${key} 缺失、维度错误或含非有限值`);
    w[key] = Float32Array.from(v);
    if (!w[key].every(Number.isFinite))
      throw new Error(`权重 ${key} 超出 Float32 范围`);
  }
  if (
    m.tokenizer &&
    (m.tokenizer.vocab.length !== c.vocabSize ||
      m.tokenizer.merges.some(
        (p) =>
          !Array.isArray(p) ||
          p.length !== 2 ||
          p.some((s) => typeof s !== "string"),
      ))
  )
    throw new Error("BPE 分词器数据无效");
  return { ...m, weights: w };
}
