import { byteInfo, byteToChar, decodeBytes } from "./byteEncoder";
import { mergeTable, pairKey, VOCAB, type MergeSet } from "./merges";
export interface BPEOptions {
  set: MergeSet;
  crossPunctuation: boolean;
  splitDigits: boolean;
  splitPunctuation: boolean;
  splitSpaces: boolean;
}
export const DEFAULT_BPE: BPEOptions = {
  set: "english",
  crossPunctuation: false,
  splitDigits: true,
  splitPunctuation: true,
  splitSpaces: true,
};
export interface Piece {
  raw: string;
  start: number;
  end: number;
  charStart: number;
  charEnd: number;
  group: number;
}
export interface Pair {
  index: number;
  left: string;
  right: string;
  rank: number | null;
  allowed: boolean;
}
export interface MergeRound {
  before: Piece[];
  pairs: Pair[];
  selected: number;
  after: Piece[];
  manual: boolean;
}
export interface Token extends Piece {
  id: number;
  text: string;
  byteCount: number;
  unknown: boolean;
}
export interface Tokenization {
  text: string;
  bytes: ReturnType<typeof byteInfo>;
  initial: Piece[];
  rounds: MergeRound[];
  tokens: Token[];
  ids: number[];
  charCount: number;
  ratio: number;
  histogram: Record<number, number>;
}
export function tokenize(
  text: string,
  o: BPEOptions = DEFAULT_BPE,
  choices: number[] = [],
  external?: { vocab: string[]; ranks: Map<string, number>; pattern?: "gpt2" },
): Tokenization {
  const chars = byteInfo(text),
    table = external?.ranks ?? mergeTable(o.set, [...text]),
    pieces: Piece[] = [];
  let group = 0,
    lastSpecial = false;
  // GPT-2's pre-tokenization regex is required for the optional pretrained model.
  const boundaries = new Set<number>();
  if (external?.pattern === "gpt2") {
    let bytes = 0;
    for (const part of text.matchAll(
      /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu,
    )) {
      boundaries.add(bytes);
      bytes += new TextEncoder().encode(part[0]).length;
    }
  }
  for (const c of chars) {
    const punctuation = /[\p{P}\p{S}]/u.test(c.char),
      special =
        !external &&
        ((o.splitSpaces && /\s/u.test(c.char)) ||
          (o.splitDigits && /\p{N}/u.test(c.char)) ||
          ((o.splitPunctuation || !o.crossPunctuation) && punctuation));
    if (special || lastSpecial || boundaries.has(c.start)) group++;
    c.bytes.forEach((b, j) =>
      pieces.push({
        raw: byteToChar.get(b)!,
        start: c.start + j,
        end: c.start + j + 1,
        charStart: c.index,
        charEnd: c.index + 1,
        group,
      }),
    );
    if (special) group++;
    lastSpecial = !!special;
  }
  const initial = pieces.map((p) => ({ ...p })),
    rounds: MergeRound[] = [];
  while (pieces.length > 1) {
    const pairs: Pair[] = pieces
      .slice(0, -1)
      .map((p, i) => ({
        index: i,
        left: p.raw,
        right: pieces[i + 1].raw,
        rank: table.get(pairKey(p.raw, pieces[i + 1].raw)) ?? null,
        allowed: p.group === pieces[i + 1].group,
      }));
    const eligible = pairs
      .filter((p) => p.allowed && p.rank !== null)
      .sort((a, b) => a.rank! - b.rank! || a.index - b.index);
    if (!eligible.length) break;
    const chosen = choices[rounds.length],
      manual = chosen !== undefined && eligible.some((p) => p.index === chosen),
      index = manual ? chosen : eligible[0].index,
      before = pieces.map((p) => ({ ...p }));
    const left = pieces[index],
      right = pieces[index + 1];
    pieces.splice(index, 2, {
      ...left,
      raw: left.raw + right.raw,
      end: right.end,
      charEnd: right.charEnd,
    });
    rounds.push({
      before,
      pairs,
      selected: index,
      after: pieces.map((p) => ({ ...p })),
      manual,
    });
  }
  const vocab = external?.vocab ?? VOCAB,
    lookup = new Map(
      vocab.map((s, i) => [external ? s : i >= 3 ? encodeSurface(s) : s, i]),
    );
  function encodeSurface(s: string) {
    return [...new TextEncoder().encode(s)]
      .map((b) => byteToChar.get(b)!)
      .join("");
  }
  const tokens = pieces.map((p) => {
    const known = lookup.get(p.raw);
    return {
      ...p,
      id: known ?? 0,
      text: decodeBytes(p.raw),
      byteCount: p.end - p.start,
      unknown: known === undefined,
    };
  });
  const histogram: Record<number, number> = {};
  tokens.forEach(
    (t) => (histogram[t.byteCount] = (histogram[t.byteCount] ?? 0) + 1),
  );
  return {
    text,
    bytes: chars,
    initial,
    rounds,
    tokens,
    ids: tokens.map((t) => t.id),
    charCount: chars.length,
    ratio: tokens.length / (chars.length || 1),
    histogram,
  };
}
