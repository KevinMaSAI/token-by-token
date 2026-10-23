import { Fragment } from "react";
import type { SavedStep } from "../state/store";
import { tokenLabel } from "../core/format";
export function Timeline({
  steps,
  selected,
  onSelect,
}: {
  steps: SavedStep[];
  selected: number;
  onSelect: (i: number) => void;
}) {
  return (
    <div className="tl" aria-label="生成时间线">
      {steps.length ? (
        steps.map((s, i) => (
          <Fragment key={i}>
            {i > 0 && <span className="tl-l" />}
            <button
              className={`tl-n ${selected === i ? "cur" : "done"}`}
              title={`第 ${i + 1} 步：${tokenLabel(s.result.vocab[s.result.sample.selected])}；点击恢复完整状态`}
              aria-label={`回放第 ${i + 1} 步`}
              onClick={() => onSelect(i)}
            >
              {i + 1}
            </button>
          </Fragment>
        ))
      ) : (
        <small>生成一步，就留下一份可回放的状态。</small>
      )}
    </div>
  );
}
