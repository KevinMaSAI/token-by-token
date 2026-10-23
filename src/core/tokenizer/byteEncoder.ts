/** GPT-2 bytes_to_unicode: reserve visible byte codepoints, map remaining bytes to 256+n. */
export const byteToChar = new Map<number, string>();
export const charToByte = new Map<string, number>();
const visible = [
  ...Array.from({ length: 94 }, (_, i) => i + 33),
  ...Array.from({ length: 12 }, (_, i) => i + 161),
  ...Array.from({ length: 82 }, (_, i) => i + 174),
];
let extra = 0;
for (let b = 0; b < 256; b++) {
  const ch = String.fromCodePoint(visible.includes(b) ? b : 256 + extra++);
  byteToChar.set(b, ch);
  charToByte.set(ch, b);
}
export const encodeBytes = (text: string) =>
  [...new TextEncoder().encode(text)].map((b) => byteToChar.get(b)!).join("");
export function decodeBytes(s: string) {
  return new TextDecoder().decode(
    Uint8Array.from(
      [...s].map((c) => {
        const b = charToByte.get(c);
        if (b === undefined) throw new Error("未知字节映射");
        return b;
      }),
    ),
  );
}
export function byteInfo(text: string) {
  let offset = 0;
  return [...text].map((char, index) => {
    const bytes = [...new TextEncoder().encode(char)],
      start = offset;
    offset += bytes.length;
    return {
      char,
      index,
      start,
      end: offset,
      bytes,
      mapped: bytes.map((b) => byteToChar.get(b)!),
    };
  });
}
