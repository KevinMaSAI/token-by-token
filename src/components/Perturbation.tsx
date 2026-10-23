import { useState } from "react";
import type { RunResult } from "../state/store";
import type { Patch } from "../core/model/config";
import { fmt, tokenLabel } from "../core/format";
import { Control } from "./Shared";
export function Perturbation({
  result,
  patch,
  onPatch,
}: {
  result: RunResult;
  patch?: Patch;
  onPatch: (p?: Patch) => void;
}) {
  const [key, setKey] = useState("embedding"),
    [row, setRow] = useState(0),
    [col, setCol] = useState(0),
    [magnitude, setMagnitude] = useState(0.3);
  const tensors = result.trace.tensors.filter(
      (t) => !t.key.endsWith(".delta") && t.key !== "probabilities",
    ),
    tensor = tensors.find((t) => t.key === key) ?? tensors[0],
    r = Math.min(row, tensor.shape[0] - 1),
    c = Math.min(col, tensor.shape[1] - 1),
    delta = result.delta;
  let up = 0,
    down = 0;
  if (delta)
    for (let i = 0; i < delta.length; i++) {
      if (delta[i] > delta[up]) up = i;
      if (delta[i] < delta[down]) down = i;
    }
  return (
    <div data-stack>
      <div className="note">
        选定一个实际计算节点，将某个数加上
        Δ，观察变化沿网络传播。差值和最终概率是观测终点，不再进入后续计算。
      </div>
      <div data-controls>
        <Control label="计算节点">
          <select
            aria-label="扰动张量"
            value={tensor.key}
            onChange={(e) => {
              setKey(e.target.value);
              setRow(0);
              setCol(0);
            }}
          >
            {tensors.map((t) => (
              <option key={t.key} value={t.key}>
                {t.key} · {t.label}
              </option>
            ))}
          </select>
        </Control>
        <Control label="行">
          <input
            type="number"
            aria-label="扰动行"
            min={0}
            max={tensor.shape[0] - 1}
            value={r}
            onChange={(e) => setRow(Math.max(0, Number(e.target.value)))}
          />
        </Control>
        <Control label="列">
          <input
            type="number"
            aria-label="扰动列"
            min={0}
            max={tensor.shape[1] - 1}
            value={c}
            onChange={(e) => setCol(Math.max(0, Number(e.target.value)))}
          />
        </Control>
        <Control label="扰动幅度 Δ">
          <input
            type="number"
            aria-label="扰动幅度"
            step="0.1"
            value={magnitude}
            onChange={(e) => setMagnitude(Number(e.target.value))}
          />
        </Control>
      </div>
      <div data-row>
        <button
          className="btn primary"
          onClick={() =>
            onPatch({
              key: tensor.key,
              index: r * tensor.shape[1] + c,
              value: magnitude,
              mode: "add",
            })
          }
        >
          施加扰动
        </button>
        <button
          className="btn"
          disabled={!patch}
          onClick={() => onPatch(undefined)}
        >
          还原
        </button>
      </div>
      {delta && result.baseline && (
        <>
          <div data-grid>
            <div className="delta-i">
              <div className="lb">
                ↑ {tokenLabel(result.vocab[up])} · #{up}
              </div>
              <div className="vl up">{fmt(delta[up], "prob", true)}</div>
            </div>
            <div className="delta-i">
              <div className="lb">
                ↓ {tokenLabel(result.vocab[down])} · #{down}
              </div>
              <div className="vl dn">{fmt(delta[down], "prob", true)}</div>
            </div>
          </div>
          <div className="note">
            {delta[up] === 0 && delta[down] === 0 ? (
              "此次干预没有改变输出概率。该维度可能未被下游使用，或被遮罩、归一化抵消。"
            ) : (
              <>
                最大上升：{tokenLabel(result.vocab[up])} 从{" "}
                {fmt(result.baseline[up], "prob")} 到{" "}
                {fmt(result.sample.probs[up], "prob")}；最大下降：
                {tokenLabel(result.vocab[down])} 从{" "}
                {fmt(result.baseline[down], "prob")} 到{" "}
                {fmt(result.sample.probs[down], "prob")}。共有 {result.affected}{" "}
                / {result.trace.ids.length} 个位置的预测改变。
              </>
            )}
          </div>
          <div data-scroll>
            <table className="mx" aria-label="概率差值热图">
              <tbody>
                {Array.from(
                  { length: Math.min(16, Math.ceil(delta.length / 4)) },
                  (_, r) => (
                    <tr key={r}>
                      {Array.from({ length: 4 }, (_, c) => {
                        const i = r * 4 + c;
                        if (i >= delta.length) return null;
                        return (
                          <td
                            key={i}
                            className={
                              delta[i] > 0 ? "pos" : delta[i] < 0 ? "neg" : ""
                            }
                            title={result.vocab[i]}
                          >
                            <small>{tokenLabel(result.vocab[i])}</small>
                            <br />
                            {fmt(delta[i], "prob", true)}
                          </td>
                        );
                      })}
                    </tr>
                  ),
                )}
              </tbody>
            </table>
          </div>
          {delta.length > 64 && (
            <small>大词表预览前 64 项；最大升降统计覆盖全部词表。</small>
          )}
        </>
      )}
    </div>
  );
}
