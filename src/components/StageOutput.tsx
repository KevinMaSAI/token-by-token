import { useState } from "react";
import type { RunResult } from "../state/store";
import type { Sampling } from "../core/sample";
import { fmt, tokenLabel } from "../core/format";
import { Card } from "./Shared";
import { MatMulView } from "./MatMulView";
import { ProbPanel } from "./ProbPanel";
export function StageOutput({
  result,
  sampling,
}: {
  result: RunResult;
  sampling: Sampling;
}) {
  const f = result.trace,
    s = result.sample,
    v = f.c.vocabSize,
    d = f.c.d,
    p = f.ids.length - 1,
    w = f.tensors.find((t) => t.key === "W:lm")!.data,
    b = f.tensors.find((t) => t.key === "W:lmBias")!.data,
    [page, setPage] = useState(0),
    start = Math.min(page, Math.max(0, Math.ceil(v / 64) - 1)) * 64;
  return (
    <>
      <Card
        idx={9}
        title="输出投影：为词表里的每一个候选打分"
        tag="最终归一化 → LM Head"
      >
        <div className="note">
          最终位置的隐藏向量，和词表中每个候选的权重向量各做一次点积。得到的{" "}
          <b>logit 只是分数</b>
          ，尚不是概率。点击矩阵中的任意候选可展开它的完整算式。
        </div>
        <details>
          <summary>最终 LayerNorm 的均值、方差与逐维计算</summary>
          <div className="fml">
            均值 = (
            {Array.from(f.final.input.slice(p * d, (p + 1) * d), (x) =>
              fmt(x, "matrix"),
            ).join(" + ")}
            ) / {d} ={" "}
            <span className="res">{fmt(f.final.mean[p], "matrix")}</span>
            <br />
            方差 = Σ(x−均值)² / {d} = {fmt(f.final.variance[p], "matrix")}
            <div className="sum">归一化后乘 γ 再加 β（RMSNorm 不加 β）</div>
            {Array.from({ length: d }, (_, j) => (
              <div key={j}>
                维 {j}：{fmt(f.final.centered[p * d + j], "matrix")} ×{" "}
                {fmt(f.final.inv[p], "matrix")} →{" "}
                {fmt(f.final.normalized[p * d + j], "matrix")} →{" "}
                <span className="res">
                  {fmt(f.final.output[p * d + j], "matrix")}
                </span>
              </div>
            ))}
          </div>
        </details>
        <MatMulView
          a={f.final.output.slice(p * d, (p + 1) * d)}
          b={w}
          out={f.logits.slice(p * v, (p + 1) * v)}
          m={1}
          k={d}
          n={v}
          bias={b}
          label="词表输出投影"
          aLabel="最终隐藏向量"
          bLabel="LM Head"
          initialCol={s.selected}
        />
      </Card>
      <Card idx="↳" title="从分数到概率 · 所有候选都在这里">
        <div className="note">
          <b>max = {fmt(s.max, "matrix")} ← 减去它可防止指数溢出。</b>
          <br />
          先把 logit 除以温度 T = {fmt(sampling.temperature)}
          ，再减去最大值，不会改变相对概率。
        </div>
        <div className="fml">
          pᵢ = exp(logitᵢ / T − max) / Σ exp(logitⱼ / T − max)
          <div className="sum">
            指数总和 = <b>{fmt(s.sum, "matrix")}</b>
          </div>
        </div>
        <div data-scroll>
          <table data-table>
            <thead>
              <tr>
                <th>候选 / ID</th>
                <th>logit</th>
                <th>÷T</th>
                <th>减 max</th>
                <th>e^</th>
                <th>概率</th>
                <th>累积</th>
                <th>截断后</th>
              </tr>
            </thead>
            <tbody>
              {s.order.slice(start, start + 64).map((id) => (
                <tr
                  key={id}
                  aria-selected={id === s.selected}
                  data-discarded={s.filtered[id] === 0}
                >
                  <td data-mono>
                    {tokenLabel(result.vocab[id])} <small>#{id}</small>
                    <i
                      data-mini-line
                      style={{
                        width: `${(s.probs[id] / (s.probs[s.order[0]] || 1)) * 100}%`,
                      }}
                    />
                  </td>
                  {[
                    f.logits[p * v + id],
                    s.scaled[id],
                    s.shifted[id],
                    s.exp[id],
                    s.probs[id],
                    s.cumulative[id],
                    s.filtered[id],
                  ].map((x, i) => (
                    <td data-number key={i}>
                      {fmt(x, i === 0 ? "logit" : i >= 4 ? "prob" : "matrix")}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {v > 64 && (
          <div data-row>
            <button
              className="btn"
              disabled={!start}
              onClick={() => setPage((p) => p - 1)}
            >
              上一页
            </button>
            <small>
              {start} / {v}
            </small>
            <button
              className="btn"
              disabled={start + 64 >= v}
              onClick={() => setPage((p) => p + 1)}
            >
              下一页
            </button>
          </div>
        )}
        <div className="note">
          {sampling.method === "greedy"
            ? "贪心直接取 logit 最大项：它只需要排名，不需要先算概率；此页仍显示 softmax 供对照。"
            : sampling.method === "top-k"
              ? `仅保留最高的 ${sampling.k} 项，然后除以保留概率之和重新归一化；暗行被丢弃。`
              : sampling.method === "top-p"
                ? `按概率从高到低累加，保留第一个使累积概率达到 ${fmt(sampling.p)} 的候选，然后截断并重新归一化。`
                : "温度小会放大分数差距，让分布更尖；温度大让分布更平。随机数落入谁的累计概率区间，就选谁。"}
        </div>
      </Card>
      <Card idx="↳" title="同一组 logits，在不同温度下会怎样？">
        <table data-table>
          <thead>
            <tr>
              <th>候选</th>
              {s.comparisons.map((c) => (
                <th key={c.t}>T={fmt(c.t, "logit")}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {s.order.slice(0, 5).map((id) => (
              <tr key={id}>
                <td>{tokenLabel(result.vocab[id])}</td>
                {s.comparisons.map((c) => (
                  <td data-number key={c.t}>
                    {fmt(c.probs[id], "prob")}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        <ProbPanel sample={s} vocab={result.vocab} />
        <div className="fml">
          u = <span className="eq">{fmt(s.draw, "matrix")}</span>（种子{" "}
          {fmt(s.seed, "integer")}，第 {fmt(s.step, "integer")} 次）
          <div className="sum">
            选中 <b>{tokenLabel(result.vocab[s.selected])}</b>，ID ={" "}
            <b>{fmt(s.selected, "integer")}</b>
            <br />
            把这个 ID 接到前缀后，重新运行同一个模型。
          </div>
        </div>
      </Card>
    </>
  );
}
