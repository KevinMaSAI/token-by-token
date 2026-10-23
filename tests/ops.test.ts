import { describe, it, expect } from "vitest";
import { Tensor } from "../src/core/tensor";
import {
  matmul,
  softmax,
  layernorm,
  rope,
  norm,
  dot,
  softmaxDetails,
} from "../src/core/ops";
describe("M0 数值基础", () => {
  it("Float32 张量、索引、切片与广播", () => {
    const t = new Tensor([2, 2], [1, 2, 3, 4]);
    expect(t.data).toBeInstanceOf(Float32Array);
    expect(t.get(1, 0)).toBe(3);
    expect([...t.row(1)]).toEqual([3, 4]);
    expect([...new Tensor([2], [1, 2]).broadcastRows(2).data]).toEqual([
      1, 2, 1, 2,
    ]);
  });
  it("矩阵乘法与独立三循环误差 < 1e-6", () => {
    const a = Float32Array.from({ length: 21 }, (_, i) => Math.sin(i)),
      b = Float32Array.from({ length: 35 }, (_, i) => Math.cos(i));
    const c = matmul(a, b, 3, 7, 5);
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 5; j++) {
        let s = 0;
        for (let k = 0; k < 7; k++) s += a[i * 7 + k] * b[k * 5 + j];
        expect(Math.abs(c[i * 5 + j] - s)).toBeLessThan(1e-6);
      }
  });
  it("稳定 softmax 不溢出且归一化", () => {
    const p = softmax(new Float32Array([1000, 1001, -Infinity]));
    expect(Math.abs(p.reduce((a, b) => a + b, 0) - 1)).toBeLessThan(1e-6);
    expect(Math.abs(p[0] - 1 / (1 + Math.E))).toBeLessThan(1e-6);
    expect(p[2]).toBe(0);
    expect(softmaxDetails(new Float32Array([-Infinity])).probs[0]).toBe(0);
  });
  it("LayerNorm 均值零、方差一", () => {
    const z = layernorm(new Float32Array([-3, 1, 4, 8]), 1e-8);
    expect(Math.abs(z.reduce((a, b) => a + b, 0) / 4)).toBeLessThan(1e-6);
    expect(Math.abs(dot(z, z) / 4 - 1)).toBeLessThan(1e-6);
  });
  it("RoPE 保范数、相对位置呈余弦关系", () => {
    const r = rope(new Float32Array([1, 0, 1, 0, 1, 0]), 3, 1, 2);
    expect(norm(r)).toBeCloseTo(Math.sqrt(3), 6);
    expect(dot(r.slice(0, 2), r.slice(4, 6))).toBeCloseTo(Math.cos(2), 6);
    expect(dot(r.slice(0, 2), r.slice(2, 4))).toBeCloseTo(
      dot(r.slice(2, 4), r.slice(4, 6)),
      6,
    );
  });
});
