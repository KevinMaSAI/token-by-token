import { useState } from "react";
import type { RunResult } from "../state/store";
import type { Options } from "../core/model/config";
import { fmt, tokenLabel } from "../core/format";
import { Card, Control, Switch } from "./Shared";
import { MatMulView } from "./MatMulView";
import { MatrixView } from "./MatrixView";
export interface AttentionSelection {
  layer: number;
  head: number;
  query: number;
  key: number;
}
export function AttentionGrid({
  result,
  selection,
  onSelection,
  onHover,
}: {
  result: RunResult;
  selection: AttentionSelection;
  onSelection: (s: AttentionSelection) => void;
  onHover: (p: number) => void;
}) {
  const f = result.trace,
    n = f.ids.length;
  return (
    <div data-grid>
      {f.layers.flatMap((l, li) =>
        l.att.heads.map((h, hi) => (
          <div key={`${li}-${hi}`}>
            <div data-between>
              <h3>
                层 {fmt(li, "integer")} · 头 {fmt(hi, "integer")}
              </h3>
              <small>
                {li === selection.layer && hi === selection.head
                  ? "当前观察"
                  : "点击任意格"}
              </small>
            </div>
            <div data-scroll>
              <table className="att" aria-label={`层 ${li} 头 ${hi} 注意力`}>
                <thead>
                  <tr>
                    <th className="ln">Q↓ K→</th>
                    {f.ids.map((_, j) => (
                      <th className="ln" key={j}>
                        {fmt(j, "integer")}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {f.ids.map((_, i) => (
                    <tr key={i}>
                      <th className="ln">{fmt(i, "integer")}</th>
                      {f.ids.map((__, j) => {
                        const v = h.probs[i * n + j],
                          masked = f.options.causal && j > i;
                        return (
                          <td
                            key={j}
                            data-masked={masked}
                            data-heat={Math.min(4, Math.floor(v * 5))}
                          >
                            <button
                              aria-label={`层 ${li} 头 ${hi} 查询 ${i} 键 ${j} 权重 ${fmt(v, "prob")}`}
                              title={`Q${i} → K${j}: ${fmt(v, "prob")}${masked ? "；未来位置被遮住" : ""}`}
                              onMouseEnter={() => onHover(j)}
                              onMouseLeave={() => onHover(-1)}
                              onFocus={() => onHover(j)}
                              onClick={() =>
                                onSelection({
                                  layer: li,
                                  head: hi,
                                  query: i,
                                  key: j,
                                })
                              }
                            >
                              {n <= 10 ? fmt(v, "prob").replace("0.", ".") : ""}
                            </button>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )),
      )}
    </div>
  );
}
export function StageAttention({
  result,
  selection,
  onSelection,
  onOptions,
  hover,
  onHover,
}: {
  result: RunResult;
  selection: AttentionSelection;
  onSelection: (s: AttentionSelection) => void;
  onOptions: (o: Options) => void;
  hover: number;
  onHover: (p: number) => void;
}) {
  const f = result.trace,
    c = f.c,
    o = f.options,
    n = f.ids.length,
    d = c.d,
    hd = d / c.heads,
    l = Math.min(selection.layer, c.layers - 1),
    h = Math.min(selection.head, c.heads - 1),
    q = Math.min(selection.query, n - 1),
    j = Math.min(selection.key, n - 1),
    a = f.layers[l].att,
    head = a.heads[h],
    [projection, setProjection] = useState<"q" | "k" | "v">("q");
  const s = { layer: l, head: h, query: q, key: j };
  const weight = (key: string) =>
    f.tensors.find((t) => t.key === `W:L${l}.${key}`)!.data;
  const headSlice = (x: Float32Array) =>
    Float32Array.from(
      { length: n * hd },
      (_, i) => x[Math.floor(i / hd) * d + h * hd + (i % hd)],
    );
  const projOut = headSlice(
      projection === "q" ? a.qRaw : projection === "k" ? a.kRaw : a.v,
    ),
    projW = Float32Array.from(
      { length: d * hd },
      (_, i) => weight(projection)[Math.floor(i / hd) * d + h * hd + (i % hd)],
    );
  const qHead = headSlice(a.q),
    kTranspose = Float32Array.from(
      { length: hd * n },
      (_, i) => a.k[(i % n) * d + h * hd + Math.floor(i / n)],
    ),
    vHead = headSlice(a.v);
  const dots = Float32Array.from(
    { length: hd },
    (_, i) => a.q[q * d + h * hd + i] * a.k[j * d + h * hd + i],
  );
  return (
    <>
      <Card
        idx={5}
        title="注意力全景 · 每一层，每一个头"
        tag={`${c.layers} 层 × ${c.heads} 头`}
      >
        <div className="note">
          每行是一个查询位置，每列是一个被读取的位置。亮度表示混入了多少信息。
          <b>点击格子进入下方计算细节</b>
          ，悬停时文字同步高亮。灰暗格子保留了未来位置的结构。
        </div>
        <AttentionGrid
          result={result}
          selection={s}
          onSelection={onSelection}
          onHover={onHover}
        />
        <div className="sec-h">当前查询正在读取哪些词</div>
        <div data-token-link>
          {f.ids.map((id, i) => (
            <button
              key={i}
              aria-current={hover === i}
              data-heat={Math.min(4, Math.floor(head.probs[q * n + i] * 5))}
              onClick={() => onSelection({ ...s, key: i })}
            >
              {tokenLabel(result.vocab[id])}{" "}
              <small>{fmt(head.probs[q * n + i], "prob")}</small>
            </button>
          ))}
        </div>
        <div data-controls>
          <Control label="层">
            <select
              value={l}
              onChange={(e) =>
                onSelection({ ...s, layer: Number(e.target.value) })
              }
            >
              {f.layers.map((_, i) => (
                <option key={i} value={i}>
                  {i}
                </option>
              ))}
            </select>
          </Control>
          <Control label="头">
            <select
              value={h}
              onChange={(e) =>
                onSelection({ ...s, head: Number(e.target.value) })
              }
            >
              {a.heads.map((_, i) => (
                <option key={i} value={i}>
                  {i}
                </option>
              ))}
            </select>
          </Control>
          <Control label="查询位置 i">
            <input
              type="number"
              min={0}
              max={n - 1}
              value={q}
              onChange={(e) =>
                onSelection({
                  ...s,
                  query: Math.max(0, Math.min(n - 1, Number(e.target.value))),
                })
              }
            />
          </Control>
          <Control label="被查询位置 j">
            <input
              type="number"
              min={0}
              max={n - 1}
              value={j}
              onChange={(e) =>
                onSelection({
                  ...s,
                  key: Math.max(0, Math.min(n - 1, Number(e.target.value))),
                })
              }
            />
          </Control>
          <Switch
            label="除以 √头维度"
            value={o.scaled}
            onChange={(v) => onOptions({ ...o, scaled: v })}
          />
          <Switch
            label="因果遮罩：不看未来"
            value={o.causal}
            onChange={(v) => onOptions({ ...o, causal: v })}
          />
          <Control label="注意力温度">
            <input
              type="number"
              min={0.1}
              max={5}
              step={0.1}
              value={o.attentionTemperature}
              onChange={(e) =>
                onOptions({
                  ...o,
                  attentionTemperature: Math.max(
                    0.1,
                    Math.min(5, Number(e.target.value)),
                  ),
                })
              }
            />
          </Control>
          <Control label="演示丢弃率">
            <input
              type="number"
              min={0}
              max={0.8}
              step={0.1}
              value={o.dropout}
              onChange={(e) =>
                onOptions({
                  ...o,
                  dropout: Math.max(0, Math.min(0.8, Number(e.target.value))),
                })
              }
            />
          </Control>
        </div>
        {!o.causal && (
          <div className="note warn">
            遮罩已关闭：早期位置可以看到后面的输入词。训练时这会造成未来信息泄漏；对当前完整前缀的最后一个位置，关闭遮罩不保证每次都降低输出质量。
          </div>
        )}
        {o.dropout > 0 && (
          <div className="note warn">
            固定种子的演示 dropout：被保留的权重除以
            1−p，因此单行权重和不再保证为 1。普通推理应设为 0。
          </div>
        )}
      </Card>
      <Card idx="1" title="输入经过三个投影，得到 Q / K / V">
        <div className="note">
          统一使用行向量约定：<b>Q = X · WQ + bQ</b>。选定头的 WQ 形状是 {d} ×{" "}
          {hd}；输出每一个维度，都是输入行与权重列的点积。
        </div>
        <div data-row>
          {(["q", "k", "v"] as const).map((k) => (
            <button
              key={k}
              className={`btn ${projection === k ? "primary" : ""}`}
              onClick={() => setProjection(k)}
            >
              {k === "q" ? "查询 Q" : k === "k" ? "键 K" : "值 V"}
            </button>
          ))}
        </div>
        <MatMulView
          a={a.input}
          b={projW}
          out={projOut}
          m={n}
          k={d}
          n={hd}
          label={`${projection.toUpperCase()} 投影`}
          aLabel="子层输入 X"
          bLabel={`W${projection.toUpperCase()} 选定头`}
          bias={weight(`${projection}Bias`).slice(h * hd, (h + 1) * hd)}
          initialRow={q}
        />
      </Card>
      <Card idx="2" title="Q · K：逐项相乘，得到相关分数">
        <div className="fml">
          <span className="lbl">
            当前查询 i = {q}，键 j = {j}
          </span>
          <br />
          {Array.from(dots, (v, i) => (
            <div key={i}>
              <span className="num">
                {fmt(a.q[q * d + h * hd + i], "matrix")}
              </span>{" "}
              ×{" "}
              <span className="num">
                {fmt(a.k[j * d + h * hd + i], "matrix")}
              </span>{" "}
              = <span className="res">{fmt(v, "matrix")}</span>
            </div>
          ))}
          <div className="sum">
            Σ = <b>{fmt(head.scores[q * n + j], "matrix")}</b>
            <br />
            除以 {o.scaled ? `√${hd}` : "1"}，再除以温度{" "}
            {fmt(o.attentionTemperature)} →{" "}
            <b>{fmt(head.scaled[q * n + j], "matrix")}</b>
          </div>
        </div>
        <details>
          <summary>查看任意位置的完整 Q · Kᵀ 矩阵乘法</summary>
          <MatMulView
            a={qHead}
            b={kTranspose}
            out={head.scores}
            m={n}
            k={hd}
            n={n}
            label="Q · Kᵀ"
            aLabel="旋转后的 Q"
            bLabel="旋转后的 Kᵀ"
            initialRow={q}
            initialCol={j}
          />
        </details>
      </Card>
      <Card idx="3" title="遮罩 → 稳定 softmax → 混合比例">
        <div className="note">
          <b>max = {fmt(head.max[q], "matrix")} ← 减去它可防止指数溢出。</b>
          未来位置先置为 −∞，所以 exp(−∞) = 0。
        </div>
        <div data-scroll>
          <table data-table>
            <thead>
              <tr>
                <th>键位置</th>
                <th>分数 / √d / T</th>
                <th>减 max</th>
                <th>e^</th>
                <th>归一化权重</th>
                <th>实际使用</th>
              </tr>
            </thead>
            <tbody>
              {f.ids.map((id, i) => (
                <tr key={i} aria-selected={i === j}>
                  <td>
                    {i} {tokenLabel(result.vocab[id])}
                  </td>
                  {[
                    head.scaled[q * n + i],
                    head.shifted[q * n + i],
                    head.exp[q * n + i],
                    head.probs[q * n + i],
                    head.used[q * n + i],
                  ].map((v, k) => (
                    <td data-number key={k}>
                      {fmt(v, k >= 3 ? "prob" : "matrix")}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="fml">
          Σ exp = <span className="res">{fmt(head.sums[q], "matrix")}</span>
          <div className="sum">
            weight[{q},{j}] = {fmt(head.exp[q * n + j], "matrix")} /{" "}
            {fmt(head.sums[q], "matrix")} ={" "}
            <b>{fmt(head.probs[q * n + j], "prob")}</b>
          </div>
        </div>
        {f.ids.map((id, i) => (
          <div className={`pbar ${i === j ? "top" : ""}`} key={i}>
            <span title={result.vocab[id]}>{tokenLabel(result.vocab[id])}</span>
            <div>
              <i style={{ width: `${head.probs[q * n + i] * 100}%` }} />
            </div>
            <output>{fmt(head.probs[q * n + i], "prob")}</output>
          </div>
        ))}
      </Card>
      <Card idx="4" title="加权求和 · 从所有位置混入 V">
        <MatMulView
          a={head.used}
          b={vHead}
          out={head.output}
          m={n}
          k={n}
          n={hd}
          label="注意力权重 · V"
          aLabel="注意力权重"
          bLabel="值向量 V"
          initialRow={q}
        />
      </Card>
      <Card idx="5" title="多头并排拼接，再经过输出投影">
        <div className="note">
          拼接只把每头的 {hd}{" "}
          个维度放到相邻位置，不求平均。不同头有各自的投影参数；在未训练模型中，不应把图案解读为语言规律。
        </div>
        <MatrixView
          data={a.concat}
          rows={n}
          cols={d}
          label="拼接后的各头输出"
          highlightRows={[q]}
        />
        <div className="sec-h">头之间的模式相似度 · 余弦</div>
        <MatrixView
          data={f.headSimilarities[l]}
          rows={c.heads}
          cols={c.heads}
          label="注意力头相似度"
          kind="weight"
        />
        <MatMulView
          a={a.concat}
          b={weight("o")}
          out={a.output}
          m={n}
          k={d}
          n={d}
          bias={weight("oBias")}
          label="多头输出投影"
          aLabel="各头拼接"
          bLabel="WO"
          initialRow={q}
        />
      </Card>
    </>
  );
}
