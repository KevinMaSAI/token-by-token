import { useState } from "react";
import type { RunResult } from "../state/store";
import type { Options } from "../core/model/config";
import { fmt } from "../core/format";
import { Card, Control } from "./Shared";
import { MatrixView } from "./MatrixView";
export function StageLayerNorm({
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
    d = f.c.d,
    [layer, setLayer] = useState(0),
    [which, setWhich] = useState<"n1" | "n2">("n1"),
    l = Math.min(layer, f.c.layers - 1),
    cache = f.layers[l],
    n = cache[which],
    row = Math.min(position, f.ids.length - 1),
    left = which === "n1" ? cache.input : cache.middle,
    right = which === "n1" ? cache.att.output : cache.ff.output,
    sum = which === "n1" ? cache.residual1 : cache.residual2,
    g = f.tensors.find((t) => t.key === `W:L${l}.${which}.gamma`)!.data,
    b = f.tensors.find((t) => t.key === `W:L${l}.${which}.beta`)!.data;
  return (
    <>
      <Card idx={6} title="残差连接：保留输入，再添加这一层的贡献">
        <div data-controls>
          <Control label="观察层">
            <select
              value={l}
              onChange={(e) => setLayer(Number(e.target.value))}
            >
              {f.layers.map((_, i) => (
                <option value={i} key={i}>
                  {i}
                </option>
              ))}
            </select>
          </Control>
          <Control label="子层">
            <select
              value={which}
              onChange={(e) => setWhich(e.target.value as "n1" | "n2")}
            >
              <option value="n1">注意力子层</option>
              <option value="n2">前馈子层</option>
            </select>
          </Control>
          <Control label="归一化">
            <select
              value={o.norm}
              onChange={(e) =>
                onOptions({ ...o, norm: e.target.value as Options["norm"] })
              }
            >
              <option value="layer">层归一化（LayerNorm）</option>
              <option value="rms">均方根归一化（RMSNorm）</option>
            </select>
          </Control>
          <Control label="结构">
            <select
              value={o.normOrder}
              onChange={(e) =>
                onOptions({
                  ...o,
                  normOrder: e.target.value as Options["normOrder"],
                })
              }
            >
              <option value="pre">归一化在前（pre-LN）</option>
              <option value="post">归一化在后（post-LN）</option>
            </select>
          </Control>
          <Control label="ε">
            <input
              type="number"
              min={1e-8}
              max={1}
              step={0.00001}
              value={o.eps}
              onChange={(e) =>
                onOptions({
                  ...o,
                  eps: Math.max(1e-8, Math.min(1, Number(e.target.value))),
                })
              }
            />
          </Control>
        </div>
        <div className="note">
          {o.normOrder === "pre"
            ? "pre-LN：x + 子层(Norm(x))。归一化先进入子层，残差主路保留直接通路。"
            : "post-LN：Norm(x + 子层(x))。先相加，再对整个残差结果归一化。"}
        </div>
        <div className="fml">
          {Array.from({ length: d }, (_, j) => (
            <div key={j}>
              <span className="lbl">维 {fmt(j, "integer")}：</span>{" "}
              <span className="num">{fmt(left[row * d + j], "matrix")}</span> +{" "}
              <span className="num">{fmt(right[row * d + j], "matrix")}</span> ={" "}
              <span className="res">{fmt(sum[row * d + j], "matrix")}</span>
            </div>
          ))}
        </div>
      </Card>
      <Card idx="↳" title="把均值、方差和每一个归一化值展开">
        <div className="fml">
          <span className="lbl">归一化实际输入（位置 {row}）</span>
          <br />
          {Array.from(n.input.slice(row * d, (row + 1) * d), (v, i) => (
            <span key={i}>
              {i ? " + " : ""}
              <span className="num">{fmt(v, "matrix")}</span>
            </span>
          ))}
          <div className="sum">
            Σ x = <b>{fmt(n.sums[row], "matrix")}</b>
            <br />
            {o.norm === "layer" ? (
              <>
                μ = Σ x / {d} = <b>{fmt(n.mean[row], "matrix")}</b>
              </>
            ) : (
              <>
                RMSNorm 不减均值：μ 使用 <b>0</b>，也不加 β。
              </>
            )}
          </div>
        </div>
        <div className="fml">
          {Array.from({ length: d }, (_, j) => (
            <div key={j}>
              (
              <span className="num">{fmt(n.input[row * d + j], "matrix")}</span>{" "}
              − {fmt(n.mean[row], "matrix")})² ={" "}
              <span className="res">
                {fmt(n.squares[row * d + j], "matrix")}
              </span>
            </div>
          ))}
          <div className="sum">
            Σ 偏差平方 / {d} = {fmt(n.squareSums[row], "matrix")} / {d} ={" "}
            <b>{fmt(n.variance[row], "matrix")}</b>
            <br />1 / √(方差 + ε) = <b>{fmt(n.inv[row], "matrix")}</b>
          </div>
        </div>
        <div data-scroll>
          <table data-table>
            <thead>
              <tr>
                <th>维度</th>
                <th>x−μ</th>
                <th>× 标准差倒数</th>
                <th>γ</th>
                <th>β</th>
                <th>结果</th>
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: d }, (_, j) => (
                <tr key={j}>
                  {[
                    j,
                    n.centered[row * d + j],
                    n.normalized[row * d + j],
                    g[j],
                    o.norm === "layer" ? b[j] : 0,
                    n.output[row * d + j],
                  ].map((v, k) => (
                    <td data-number key={k}>
                      {fmt(v, k === 0 ? "integer" : "matrix")}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="fml">y[j] = (x[j] − μ) / √(var + ε) × γ[j] + β[j]</div>
      </Card>
      <Card idx="↳" title="同一组权重，两种结构的输出对照">
        <div className="note warn">
          这里真实计算两种结构的最终概率差异，不能用一次前向运算证明训练稳定性。深层网络中
          post-LN 的梯度更依赖初始化和学习率；训练页可以分别运行并比较损失。
        </div>
        <MatrixView
          data={result.normComparison.pre}
          rows={1}
          cols={f.c.vocabSize}
          label="pre-LN 的输出概率"
          kind="prob"
        />
        <MatrixView
          data={result.normComparison.post}
          rows={1}
          cols={f.c.vocabSize}
          label="post-LN 的输出概率"
          kind="prob"
        />
      </Card>
    </>
  );
}
