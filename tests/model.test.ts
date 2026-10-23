import { it, expect } from "vitest";
import { MINI, DEFAULT_OPTIONS } from "../src/core/model/config";
import { createWeights, parameterCount } from "../src/core/model/weights";
import { forward } from "../src/core/model/forward";
it("M1 微型模型真实前向、严格因果遮罩、参数少于五万", () => {
  const w = createWeights();
  expect(parameterCount(w)).toBeLessThan(50000);
  const f = forward([1, 2, 3, 4], MINI, w, DEFAULT_OPTIONS);
  expect(f.logits.length).toBe(4 * 64);
  for (const l of f.layers)
    for (const h of l.att.heads)
      for (let i = 0; i < 4; i++) {
        let sum = 0;
        for (let j = 0; j < 4; j++) {
          if (j > i) expect(h.probs[i * 4 + j]).toBe(0);
          sum += h.probs[i * 4 + j];
        }
        expect(sum).toBeCloseTo(1, 6);
      }
  expect(f.tensors.every((t) => t.data instanceof Float32Array)).toBe(true);
  expect(f.logits.every(Number.isFinite)).toBe(true);
});
it("同一种子生成相同的可读稀疏权重", () => {
  expect(createWeights().embedding).toEqual(createWeights().embedding);
  expect([...new Set(createWeights().embedding)]).toEqual(
    expect.arrayContaining([0, -0.125, 0.125, 0.25]),
  );
});
it("单个中间数干预真实传播且原始权重不被污染", () => {
  const w = createWeights();
  const a = forward([1, 2, 3], MINI, w, DEFAULT_OPTIONS);
  const b = forward([1, 2, 3], MINI, w, DEFAULT_OPTIONS, {
    key: "L0.qRaw",
    index: 65,
    value: 2,
    mode: "add",
  });
  expect(a.logits).not.toEqual(b.logits);
  expect(a.layers[0].att.qRaw[65] + 2).toBeCloseTo(b.layers[0].att.qRaw[65], 6);
  expect(forward([1, 2, 3], MINI, w, DEFAULT_OPTIONS).logits).toEqual(a.logits);
});
