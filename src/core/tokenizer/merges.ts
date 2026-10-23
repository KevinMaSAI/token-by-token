import { encodeBytes } from "./byteEncoder";
export const VOCAB = [
  "<unk>",
  "<bos>",
  "<eos>",
  " ",
  "\n",
  ".",
  ",",
  "!",
  "?",
  "a",
  "b",
  "c",
  "d",
  "e",
  "f",
  "g",
  "h",
  "i",
  "j",
  "k",
  "l",
  "m",
  "n",
  "o",
  "p",
  "q",
  "r",
  "s",
  "t",
  "u",
  "v",
  "w",
  "x",
  "y",
  "z",
  "0",
  "1",
  "2",
  "3",
  "4",
  "5",
  "6",
  "7",
  "8",
  "9",
  "我",
  "你",
  "他",
  "爱",
  "学",
  "习",
  "人",
  "工",
  "智",
  "能",
  "天",
  "气",
  "好",
  "中",
  "文",
  "the",
  "is",
  "hello",
  "世界",
];
export type MergeSet = "english" | "chinese" | "character";
export const pairKey = (a: string, b: string) => a + "\u0000" + b;
export function mergeTable(set: MergeSet, chars: string[] = []) {
  const table = new Map<string, number>();
  const add = (text: string) => {
    const units = [...encodeBytes(text)];
    let left = units[0];
    for (let i = 1; i < units.length; i++) {
      const k = pairKey(left, units[i]);
      if (!table.has(k)) table.set(k, table.size);
      left += units[i];
    }
  };
  if (set === "english") ["the", "is", "hello"].forEach(add);
  // A 256-byte alphabet is independent from the 64-entry model vocabulary.
  [
    ...new Set([
      ...VOCAB.slice(45, 60),
      "世",
      "界",
      ...chars.filter((c) => new TextEncoder().encode(c).length > 1),
    ]),
  ].forEach(add);
  if (set === "chinese") ["the", "is"].forEach(add);
  if (set === "english") add("世界");
  // Boundary rules can be explored even when their merged result maps to UNK.
  if (set !== "character") ["the ", "hello!", "12"].forEach(add);
  return table;
}
