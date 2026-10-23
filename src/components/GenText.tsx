import { tokenLabel } from "../core/format";
export function GenText({
  prompt,
  ids,
  vocab,
}: {
  prompt: string;
  ids: number[];
  vocab: string[];
}) {
  return (
    <div className="gen" aria-label="已生成文本">
      <span className="g1">{prompt}</span>
      {ids.map((id, i) => (
        <span
          key={i}
          className={i === ids.length - 1 ? "g2" : "g3"}
          title={`第 ${i + 1} 步 · ID ${id}`}
        >
          {id < 3 ? `〔${tokenLabel(vocab[id])}〕` : vocab[id]}
        </span>
      ))}
      {!ids.length && <span className="g2">▏</span>}
    </div>
  );
}
