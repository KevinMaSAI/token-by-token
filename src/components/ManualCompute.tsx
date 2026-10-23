import { useState } from "react";
import type { RunResult } from "../state/store";
import type { Patch } from "../core/model/config";
import type { AttentionSelection } from "./StageAttention";
import { Card } from "./Shared";
import { fmt } from "../core/format";
import { MatMulView } from "./MatMulView";
import { MatrixView } from "./MatrixView";
export function ManualCompute({
  result,
  selection,
  onPatch,
}: {
  result: RunResult;
  selection: AttentionSelection;
  onPatch: (p: Patch) => void;
}) {
  const f = result.trace,
    l = Math.min(selection.layer, f.c.layers - 1),
    h = Math.min(selection.head, f.c.heads - 1),
    q = Math.min(selection.query, f.ids.length - 1),
    d = f.c.d,
    hd = d / f.c.heads,
    a = f.layers[l].att,
    [step, setStep] = useState(0),
    [value, setValue] = useState("0.3"),
    w = f.tensors.find((t) => t.key === `W:L${l}.q`)!.data,
    headWeight = Float32Array.from(
      { length: d * hd },
      (_, i) => w[Math.floor(i / hd) * d + h * hd + (i % hd)],
    ),
    out = Float32Array.from(
      { length: f.ids.length * hd },
      (_, i) => a.qRaw[Math.floor(i / hd) * d + h * hd + (i % hd)],
    );
  return (
    <Card
      idx="✎"
      title="让我带着你算一次"
      tag={`层 ${l} / 头 ${h} / 位置 ${q}`}
    >
      <div className="note">
        每一步都可以停下来。先取输入，再做每项乘法，最后求和。你也可以在 Q
        的第一个维度写入一个新数，从此处继续整个网络。
      </div>
      <div data-row>
        {["取出 xᵢ", "输入 × 权重", "求和得到 qᵢ", "改一个数，再继续"].map(
          (s, i) => (
            <button
              key={i}
              className={`btn ${step === i ? "primary" : ""}`}
              onClick={() => setStep(i)}
            >
              {i + 1}. {s}
            </button>
          ),
        )}
      </div>
      <div data-space>
        {step === 0 ? (
          <MatrixView
            data={a.input.slice(q * d, (q + 1) * d)}
            rows={1}
            cols={d}
            label="被选位置的完整输入"
          />
        ) : step < 3 ? (
          <MatMulView
            a={a.input}
            b={headWeight}
            out={out}
            m={f.ids.length}
            k={d}
            n={hd}
            label="手动计算 Q"
            initialRow={q}
            bias={f.tensors
              .find((t) => t.key === `W:L${l}.qBias`)!
              .data.slice(h * hd, (h + 1) * hd)}
          />
        ) : (
          <>
            <div className="fml">
              当前 q[{q}][{h * hd}] ={" "}
              <span className="res">
                {fmt(a.qRaw[q * d + h * hd], "matrix")}
              </span>
            </div>
            <div data-row>
              <label>
                改为{" "}
                <input
                  aria-label="手算中间值"
                  type="number"
                  value={value}
                  step="0.1"
                  onChange={(e) => setValue(e.target.value)}
                />
              </label>
              <button
                className="btn primary"
                disabled={!Number.isFinite(Number(value))}
                onClick={() =>
                  onPatch({
                    key: `L${l}.qRaw`,
                    index: q * d + h * hd,
                    value: Number(value),
                    mode: "set",
                  })
                }
              >
                从这里继续
              </button>
            </div>
            <div className="note">
              干预后的 Q 会进入位置旋转、相关分数、softmax、FFN
              与输出。右侧扰动实验报告最终概率变化。
            </div>
          </>
        )}
      </div>
      <div data-row data-space>
        <button
          className="btn"
          disabled={step === 0}
          onClick={() => setStep((s) => s - 1)}
        >
          上一步
        </button>
        <button
          className="btn primary"
          disabled={step === 3}
          onClick={() => setStep((s) => s + 1)}
        >
          继续 →
        </button>
      </div>
    </Card>
  );
}
