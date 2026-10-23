import { it, expect } from "vitest";
import {
  MINI,
  DEFAULT_OPTIONS,
  type Options,
  type Config,
} from "../src/core/model/config";
import { createWeights } from "../src/core/model/weights";
import { forward } from "../src/core/model/forward";
import {
  backward,
  crossEntropy,
  Adam,
  trainStep,
  trainingData,
} from "../src/core/model/train";
const cfg: Config = {
  ...MINI,
  layers: 2,
  d: 4,
  heads: 2,
  ff: 6,
  vocabSize: 8,
  maxSeq: 4,
};
it("M9 全网络手写反向传播：有限差分绝对误差 < 1e-5", () => {
  for (const normOrder of ["pre", "post"] as const)
    for (const activation of ["gelu", "swiglu"] as const) {
      const w = createWeights(cfg, 93),
        o: Options = { ...DEFAULT_OPTIONS, normOrder, activation, ffWidth: 6 },
        ids = [1, 2, 3],
        targets = [2, 3, 4],
        f = forward(ids, cfg, w, o, undefined, false),
        g = backward(f, w, targets);
      const checks: [string, number][] = [
        ["embedding", 5],
        ["lm", 9],
        ["final.gamma", 1],
        ["final.beta", 1],
        ["L0.q", 3],
        ["L0.k", 7],
        ["L0.v", 6],
        ["L0.o", 2],
        ["L0.n1.gamma", 0],
        ["L0.n2.beta", 1],
        ["L0.w1", 9],
        ["L0.w2", 8],
        ["L0.gate", 5],
        ["L0.b1", 2],
        ["L1.q", 6],
        ["L1.w2", 3],
      ];
      for (const [key, index] of checks) {
        const value = w[key][index],
          h = 0.01;
        const evalAt = (v: number) => {
          w[key][index] = v;
          return crossEntropy(
            forward(ids, cfg, w, o, undefined, false),
            targets,
          );
        };
        const central = (step: number) => {
          const plus = evalAt(value + step),
            xp = w[key][index],
            minus = evalAt(value - step),
            xm = w[key][index];
          return (plus - minus) / (xp - xm);
        };
        const d1 = central(h),
          d2 = central(h / 2),
          numeric = (4 * d2 - d1) / 3;
        w[key][index] = value;
        expect(
          Math.abs(numeric - g[key][index]),
          `${normOrder}/${activation} ${key}[${index}]: analytic=${g[key][index]}, numeric=${numeric}`,
        ).toBeLessThan(1e-5);
      }
    }
});
it("全部 2 层参数训练：真实损失下降与梯度裁剪", () => {
  const w = createWeights(),
    adam = new Adam(w),
    data = trainingData()[0],
    before = crossEntropy(
      forward(data.input, MINI, w, DEFAULT_OPTIONS, undefined, false),
      data.target,
    );
  let last = before;
  for (let i = 0; i < 80; i++) {
    const m = trainStep(w, MINI, DEFAULT_OPTIONS, adam, data, 0.005, 1);
    last = m.loss;
    expect(m.clippedNorm).toBeLessThanOrEqual(1.000001);
  }
  expect(last).toBeLessThan(before * 0.35);
  expect(w["L0.q"]).not.toEqual(createWeights()["L0.q"]);
});
