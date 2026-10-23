import { useState } from "react";
import type { RunResult } from "../state/store";
import type { Patch } from "../core/model/config";
import { fmt } from "../core/format";
import { Card, Control } from "./Shared";
import { MatrixView } from "./MatrixView";
export function TensorInspector({
  result,
  onPatch,
}: {
  result: RunResult;
  onPatch: (p: Patch) => void;
}) {
  const [key, setKey] = useState("embedding"),
    [cell, setCell] = useState<[number, number]>([0, 0]),
    [mode, setMode] = useState<"add" | "set">("set"),
    [value, setValue] = useState("0.3"),
    t =
      result.trace.tensors.find((t) => t.key === key) ??
      result.trace.tensors[0],
    r = Math.min(cell[0], t.shape[0] - 1),
    c = Math.min(cell[1], t.shape[1] - 1),
    leaf = t.key.endsWith(".delta") || t.key === "probabilities";
  return (
    <Card idx="⌗" title="完整张量探针 · 没有隐藏的数字" tag="Float32Array">
      <div data-controls>
        <Control label="选择任意权重或中间张量">
          <select
            aria-label="完整张量选择"
            value={t.key}
            onChange={(e) => {
              setKey(e.target.value);
              setCell([0, 0]);
            }}
          >
            {result.trace.tensors.map((t) => (
              <option key={t.key} value={t.key}>
                {t.key} · {t.label} [{t.shape.join("×")}]
              </option>
            ))}
          </select>
        </Control>
      </div>
      <MatrixView
        data={t.data}
        rows={t.shape[0]}
        cols={t.shape[1]}
        label={t.label}
        selected={[r, c]}
        onCell={(row, col) => {
          setCell([row, col]);
          const v = t.data[row * t.shape[1] + col];
          setValue(Number.isFinite(v) ? String(v) : "0");
        }}
      />
      <div className="fml">
        {t.key}[{r},{c}] ={" "}
        <span className="res">{fmt(t.data[r * t.shape[1] + c], "matrix")}</span>
      </div>
      {leaf ? (
        <div className="note">
          此张量是用于观察的终点。要改变它，请在它的输入路径施加干预。
        </div>
      ) : (
        <>
          <div data-row>
            <label>
              操作{" "}
              <select
                value={mode}
                onChange={(e) => setMode(e.target.value as "add" | "set")}
              >
                <option value="set">设置为</option>
                <option value="add">增加</option>
              </select>
            </label>
            <input
              aria-label="张量单元格新值"
              type="number"
              step="0.1"
              value={value}
              onChange={(e) => setValue(e.target.value)}
            />
            <button
              className="btn primary"
              disabled={!Number.isFinite(Number(value))}
              onClick={() =>
                onPatch({
                  key: t.key,
                  index: r * t.shape[1] + c,
                  mode,
                  value: Number(value),
                })
              }
            >
              从这个单元格继续计算
            </button>
          </div>
          <div className="note warn">
            直接修改概率权重、遮罩或中间统计量，可能使它们不再满足归一化或因果约束。这是可见的实验干预；按右侧「还原」恢复标准前向。
          </div>
        </>
      )}
    </Card>
  );
}
