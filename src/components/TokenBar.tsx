import { fmt, tokenLabel } from "../core/format";
import type { Tokenization } from "../core/tokenizer/bpe";
export function TokenBar({
  data,
  selected,
  onSelect,
}: {
  data: Tokenization;
  selected: number;
  onSelect: (i: number) => void;
}) {
  return (
    <div data-token-list>
      {data.tokens.map((t, i) => (
        <button
          key={i}
          className={`tok ${t.byteCount > 1 ? "multi" : ""} ${selected === i ? "pick" : ""}`}
          onClick={() => onSelect(i)}
          title={`字符区间 [${t.charStart}, ${t.charEnd})，字节区间 [${t.start}, ${t.end})${t.unknown ? "；词表外，使用未知 ID 0" : ""}`}
          style={{ flexGrow: Math.min(t.byteCount, 6) }}
        >
          {tokenLabel(t.text)}
          <span className="id">
            #{fmt(t.id, "integer")} · {fmt(t.byteCount, "integer")} 字节
          </span>
        </button>
      ))}
    </div>
  );
}
