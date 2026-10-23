import { useState } from "react";
import type { RunResult } from "../state/store";
import type { Options } from "../core/model/config";
import { fmt } from "../core/format";
import { Card, Control } from "./Shared";
import { MatMulView } from "./MatMulView";
export function StageFFN({
  result,
  onOptions,
  position,
}: {
  result: RunResult;
  onOptions: (o: Options) => void;
  position: number;
}) {
  const f = result.trace,
    o = f.options,
    [layer, setLayer] = useState(0),
    l = Math.min(layer, f.c.layers - 1),
    c = f.c,
    a = f.layers[l].ff,
    p = Math.min(position, f.ids.length - 1),
    w = (key: string) =>
      f.tensors.find((t) => t.key === `W:L${l}.${key}`)!.data;
  const max = Math.max(
    1,
    ...a.pre.slice(p * c.ff, (p + 1) * c.ff).map(Math.abs),
    ...a.gated.slice(p * c.ff, (p + 1) * c.ff).map(Math.abs),
  );
  return (
    <>
      <Card idx={7} title="FFN：每个位置独立通过同一组参数">
        <div className="note">
          FFN 在每个 token 内组合不同维度，<b>不在此处混合不同位置</b>。维度先从{" "}
          {c.d} 扩到 {c.ff}，经过非线性，再压回 {c.d}。
        </div>
        <div data-controls>
          <Control label="层">
            <select
              value={l}
              onChange={(e) => setLayer(Number(e.target.value))}
            >
              {f.layers.map((_, i) => (
                <option key={i} value={i}>
                  {i}
                </option>
              ))}
            </select>
          </Control>
          <Control label="激活函数">
            <select
              value={o.activation}
              onChange={(e) =>
                onOptions({
                  ...o,
                  activation: e.target.value as Options["activation"],
                })
              }
            >
              <option value="relu">线性整流（ReLU）</option>
              <option value="gelu">高斯误差线性（GELU）</option>
              <option value="silu">自门控线性（SiLU）</option>
              <option value="swiglu">门控线性（SwiGLU）</option>
            </select>
          </Control>
          <Control label="参与计算的隐藏维度">
            <input
              type="number"
              min={1}
              max={c.ff}
              value={o.ffWidth}
              onChange={(e) =>
                onOptions({
                  ...o,
                  ffWidth: Math.max(1, Math.min(c.ff, Number(e.target.value))),
                })
              }
            />
          </Control>
        </div>
        <div className="note warn">
          调窄宽度时，后面的隐藏维度置零，参数矩阵形状保留，便于对照。重新训练时，只有启用的维度参与梯度更新。
        </div>
        <MatMulView
          a={a.input}
          b={w("w1")}
          out={a.pre}
          m={f.ids.length}
          k={c.d}
          n={c.ff}
          bias={w("b1")}
          label="FFN 第一层"
          aLabel="FFN 输入"
          bLabel="W1"
          initialRow={p}
        />
      </Card>
      <Card idx="↳" title="激活前后 · 负值去了哪里？">
        <div data-bars>
          {Array.from({ length: c.ff }, (_, j) => (
            <div
              key={j}
              title={`维 ${j}: ${fmt(a.pre[p * c.ff + j])} → ${fmt(a.gated[p * c.ff + j])}`}
            >
              <i
                style={{
                  height: `${(Math.abs(a.pre[p * c.ff + j]) / max) * 100}%`,
                }}
              />
              <i
                style={{
                  height: `${(Math.abs(a.gated[p * c.ff + j]) / max) * 100}%`,
                }}
              />
            </div>
          ))}
        </div>
        <small>
          左条：激活前绝对值；右条：有效激活后绝对值。正负号请看数值表。
        </small>
        <div className="fml">
          {o.activation === "relu"
            ? "ReLU(x) = max(0, x)"
            : o.activation === "gelu"
              ? "GELU(x) ≈ 0.5x × [1 + tanh(√(2/π) × (x + 0.044715x³))]"
              : o.activation === "silu"
                ? "SiLU(x) = x / (1 + exp(−x))"
                : "SwiGLU(a, b) = SiLU(a) × b；b = X · Wgate + bgate"}
        </div>
        <details>
          <summary>查看全部维度的激活前后数值</summary>
          <table data-table>
            <thead>
              <tr>
                <th>维度</th>
                <th>激活前</th>
                <th>激活后</th>
                <th>门值</th>
                <th>有效结果</th>
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: c.ff }, (_, j) => (
                <tr key={j}>
                  {[
                    j,
                    a.pre[p * c.ff + j],
                    a.activated[p * c.ff + j],
                    a.gate[p * c.ff + j],
                    a.gated[p * c.ff + j],
                  ].map((v, k) => (
                    <td key={k} data-number>
                      {fmt(v, k === 0 ? "integer" : "matrix")}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </details>
        {o.activation === "swiglu" && (
          <MatMulView
            a={a.input}
            b={w("gate")}
            out={a.gate}
            m={f.ids.length}
            k={c.d}
            n={c.ff}
            bias={w("gateBias")}
            label="SwiGLU 门投影"
            initialRow={p}
          />
        )}
      </Card>
      <Card idx="↳" title="第二次矩阵乘法：写回残差流">
        <MatMulView
          a={a.gated}
          b={w("w2")}
          out={a.output}
          m={f.ids.length}
          k={c.ff}
          n={c.d}
          bias={w("b2")}
          label="FFN 第二层"
          aLabel="有效激活向量"
          bLabel="W2"
          initialRow={p}
        />
      </Card>
    </>
  );
}
