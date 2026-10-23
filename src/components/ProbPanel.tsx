import { fmt, tokenLabel } from "../core/format";
import type { SampleResult } from "../core/sample";
export function ProbPanel({
  sample,
  vocab,
  count = 5,
}: {
  sample: SampleResult;
  vocab: string[];
  count?: number;
}) {
  return (
    <>
      {sample.order.slice(0, count).map((id, i) => (
        <div className={`pbar ${i === 0 ? "top" : ""}`} key={id}>
          <span title={`${vocab[id]} · ID ${id}`}>{tokenLabel(vocab[id])}</span>
          <div>
            <i style={{ width: `${sample.probs[id] * 100}%` }} />
          </div>
          <output>{fmt(sample.probs[id], "prob")}</output>
        </div>
      ))}
      <div data-between data-space>
        <small>当前抽样结果</small>
        <strong data-amber>
          {tokenLabel(vocab[sample.selected])} · #
          {fmt(sample.selected, "integer")}
        </strong>
      </div>
      <div className="fml">
        <span className="lbl">种子</span> {fmt(sample.seed, "integer")}
        <br />
        <span className="lbl">实际随机数 u</span>{" "}
        <span className="res">{fmt(sample.draw, "matrix")}</span>
      </div>
    </>
  );
}
