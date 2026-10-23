import { useState } from "react";
import type { RunResult } from "../state/store";
import { fmt } from "../core/format";
import { Card } from "./Shared";
import { MatrixView } from "./MatrixView";
export function StageResidual({
  result,
  position,
  onPosition,
}: {
  result: RunResult;
  position: number;
  onPosition: (p: number) => void;
}) {
  const f = result.trace,
    [selected, setSelected] = useState(0),
    l = Math.min(selected, f.c.layers - 1),
    p = Math.min(position, f.ids.length - 1),
    c = f.layers[l];
  return (
    <>
      <Card idx={8} title="残差流 · 每一层都在现有信息上写入">
        <div className="note">
          选择层，再点击任意位置的维度。差值 <b>x_out − x_in</b>{" "}
          显示这一层实际增加或减少了什么；注意力与 FFN 的范数帮助比较贡献大小。
        </div>
        <div data-tracebar>
          <span className="pill">嵌入</span>
          <span data-ink>→</span>
          {f.layers.map((_, i) => (
            <span data-row key={i}>
              <button
                className={`btn ${l === i ? "primary" : ""}`}
                onClick={() => setSelected(i)}
              >
                第 {fmt(i, "integer")} 层
              </button>
              <span data-ink>→</span>
            </span>
          ))}
          <span className="pill">最终归一化</span>
        </div>
        <div data-scroll>
          <table data-table>
            <thead>
              <tr>
                <th>层</th>
                <th>张量形状</th>
                <th>输入范数</th>
                <th>注意力范数</th>
                <th>FFN 范数</th>
                <th>输出范数</th>
                <th>变化范数</th>
              </tr>
            </thead>
            <tbody>
              {f.activationNorms.map((v, i) => (
                <tr
                  key={i}
                  aria-selected={i === l}
                  onClick={() => setSelected(i)}
                >
                  <td>
                    <button className="btn">{i}</button>
                  </td>
                  <td data-mono>
                    {f.ids.length} × {f.c.d}
                  </td>
                  {[v.input, v.attention, v.ffn, v.output, v.delta].map(
                    (n, j) => (
                      <td data-number key={j}>
                        {fmt(n)}
                      </td>
                    ),
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <Card idx="↳" title={`第 ${l} 层展开 · 点击差值探针`}>
        <MatrixView
          data={c.delta}
          rows={f.ids.length}
          cols={f.c.d}
          label="残差流变化 x_out − x_in"
          highlightRows={[p]}
          onCell={(i) => onPosition(i)}
          onRow={onPosition}
        />
        <div className="fml">
          {Array.from({ length: f.c.d }, (_, j) => (
            <div key={j}>
              <span className="lbl">
                位置 {p} 维 {j}：
              </span>
              <span className="num">
                {fmt(c.output[p * f.c.d + j], "matrix")}
              </span>{" "}
              −{" "}
              <span className="num">
                {fmt(c.input[p * f.c.d + j], "matrix")}
              </span>{" "}
              ={" "}
              <span className="res">
                {fmt(c.delta[p * f.c.d + j], "matrix")}
              </span>
            </div>
          ))}
        </div>
        <details>
          <summary>叠看输入、注意力贡献、FFN 贡献与输出</summary>
          {[
            { data: c.input, label: "层输入" },
            { data: c.att.output, label: "注意力写入" },
            { data: c.ff.output, label: "FFN 写入" },
            { data: c.output, label: "层输出" },
          ].map((v) => (
            <MatrixView
              key={v.label}
              data={v.data.slice(p * f.c.d, (p + 1) * f.c.d)}
              rows={1}
              cols={f.c.d}
              label={v.label}
            />
          ))}
        </details>
      </Card>
    </>
  );
}
