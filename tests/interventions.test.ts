import { it, expect } from "vitest";
import { MINI, DEFAULT_OPTIONS } from "../src/core/model/config";
import { createWeights } from "../src/core/model/weights";
import { forward } from "../src/core/model/forward";
import { tokenize, DEFAULT_BPE } from "../src/core/tokenizer/bpe";
import { validateModel } from "../src/core/model/load";
import { VOCAB } from "../src/core/tokenizer/merges";
it("中间检查点确实参与后续运算，而非只改显示", () => {
  const w = createWeights(),
    ids = [45, 46, 3, 62],
    a = forward(ids, MINI, w, DEFAULT_OPTIONS);
  for (const [key, index] of [
    ["L0.H0.shifted", 12],
    ["L0.H0.exp", 12],
    ["L0.H0.sums", 3],
    ["L0.n1.mean", 3],
    ["L0.n1.normalized", 97],
    ["L0.ffPre", 193],
    ["L0.ffGated", 193],
    ["L0.concat", 97],
    ["L0.output", 97],
    ["final.normalized", 97],
  ] as [string, number][]) {
    const b = forward(ids, MINI, w, DEFAULT_OPTIONS, {
      key,
      index,
      value: 0.3,
      mode: "add",
    });
    expect(b.logits, `${key} 应传播至最终 logits`).not.toEqual(a.logits);
  }
});
it("没有位置编码且无因果遮罩时，token 置换只置换对应输出行", () => {
  const w = createWeights(),
    o = { ...DEFAULT_OPTIONS, position: "none" as const, causal: false },
    a = forward([45, 46, 62], MINI, w, o),
    b = forward([62, 45, 46], MINI, w, o);
  for (let j = 0; j < 64; j++)
    expect(a.logits[j]).toBeCloseTo(b.logits[64 + j], 5);
});
it("可调边界会改变真实合并，三种表有不同路径", () => {
  expect(tokenize("12").tokens.length).toBe(2);
  expect(
    tokenize("12", { ...DEFAULT_BPE, splitDigits: false }).tokens.length,
  ).toBe(1);
  expect(tokenize("hello!").tokens.length).toBe(2);
  expect(
    tokenize("hello!", {
      ...DEFAULT_BPE,
      splitPunctuation: false,
      crossPunctuation: true,
    }).tokens.length,
  ).toBe(1);
  expect(
    tokenize("the is hello", { ...DEFAULT_BPE, set: "character" }).tokens
      .length,
  ).toBeGreaterThan(
    tokenize("the is hello", { ...DEFAULT_BPE, set: "chinese" }).tokens.length,
  );
});
it("模型导入逐项校验，Float32 往返保持计算一致", () => {
  const w = createWeights(),
    json = {
      format: "token-by-token-v1",
      config: MINI,
      weights: Object.fromEntries(
        Object.entries(w).map(([k, v]) => [k, [...v]]),
      ),
      vocab: VOCAB,
      options: DEFAULT_OPTIONS,
    },
    loaded = validateModel(JSON.parse(JSON.stringify(json)));
  expect(forward([1, 2], MINI, loaded.weights, DEFAULT_OPTIONS).logits).toEqual(
    forward([1, 2], MINI, w, DEFAULT_OPTIONS).logits,
  );
  expect(() =>
    validateModel({ ...json, config: { ...MINI, d: 999 } }),
  ).toThrow();
  expect(() =>
    validateModel({ ...json, weights: { ...json.weights, embedding: [NaN] } }),
  ).toThrow();
});
