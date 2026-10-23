import { it, expect } from "vitest";
import {
  byteToChar,
  charToByte,
  encodeBytes,
  decodeBytes,
} from "../src/core/tokenizer/byteEncoder";
import { tokenize, DEFAULT_BPE } from "../src/core/tokenizer/bpe";
import { VOCAB } from "../src/core/tokenizer/merges";
import { sample, topP, DEFAULT_SAMPLING } from "../src/core/sample";
it("M2 恰好 64 词表，256 字节映射可逆", () => {
  expect(VOCAB.length).toBe(64);
  expect(new Set(VOCAB).size).toBe(64);
  for (let b = 0; b < 256; b++)
    expect(charToByte.get(byteToChar.get(b)!)).toBe(b);
  expect(decodeBytes(encodeBytes("你好 hello🙂"))).toBe("你好 hello🙂");
});
it("中英混合分词期望及完整合并轨迹可复现", () => {
  const a = tokenize("你好 hello");
  expect(a.tokens.map((t) => t.text)).toEqual(["你", "好", " ", "hello"]);
  expect(a.ids).toHaveLength(4);
  expect(a).toEqual(tokenize("你好 hello"));
  expect(a.rounds.length).toBe(8);
  expect(a.tokens.every((t) => !t.unknown)).toBe(true);
  expect(
    tokenize("你好 hello", { ...DEFAULT_BPE, set: "character" }).tokens.length,
  ).toBe(8);
});
it("手动选合并，未知字符有明确 UNK，空输入安全", () => {
  const a = tokenize("hello");
  const eligible = a.rounds[0].pairs.filter(
    (p) => p.allowed && p.rank !== null,
  );
  if (eligible.length > 1)
    expect(
      tokenize("hello", DEFAULT_BPE, [eligible.at(-1)!.index]).rounds[0].manual,
    ).toBe(true);
  expect(tokenize("🙂").tokens[0].unknown).toBe(true);
  expect(tokenize("").ids).toHaveLength(0);
});
it("M7 采样可复现、top-p 重新归一化", () => {
  const x = new Float32Array([0.4, 0.3, 0.2, 0.1]);
  const p = topP(x, 0.65);
  expect(p[0]).toBeCloseTo(4 / 7, 6);
  expect(p[1]).toBeCloseTo(3 / 7, 6);
  expect([...p.slice(2)]).toEqual([0, 0]);
  expect(Math.abs(p.reduce((a, b) => a + b, 0) - 1)).toBeLessThan(1e-6);
  expect(sample(x, DEFAULT_SAMPLING, 4)).toEqual(
    sample(x, DEFAULT_SAMPLING, 4),
  );
});
