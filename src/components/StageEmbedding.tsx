import { useEffect, useState } from "react";
import { rpc, inspect, type RunResult } from "../state/store";
import { FormulaView, type Arithmetic } from "./FormulaView";
import { MatrixView } from "./MatrixView";
import { Card } from "./Shared";
import { fmt, tokenLabel } from "../core/format";
export function StageEmbedding({
  result,
  onEdit,
}: {
  result: RunResult;
  onEdit: (key: string, index: number, value: number) => void;
}) {
  const f = result.trace,
    d = f.c.d,
    matrix = f.tensors.find((t) => t.key === "W:embedding")!.data;
  const [row, setRow] = useState(f.ids[0]),
    [col, setCol] = useState(0),
    [value, setValue] = useState("0.125"),
    r = Math.min(row, f.c.vocabSize - 1);
  const [ranking, setRanking] = useState<{
      scores: Float32Array;
      order: number[];
    }>(),
    [page, setPage] = useState(0),
    [target, setTarget] = useState<number>(),
    [arithmetic, setArithmetic] = useState<Arithmetic | null>(null),
    [similarityError, setSimilarityError] = useState("");
  useEffect(() => {
    let alive = true;
    setPage(0);
    rpc<{ scores: Float32Array; order: number[] }>("similarity", {
      row: r,
      matrix,
      width: d,
      count: f.c.vocabSize,
    })
      .then((v) => {
        if (alive) setRanking(v);
      })
      .catch((e) => {
        if (alive) setSimilarityError(String(e));
      });
    return () => {
      alive = false;
    };
  }, [r, result]);
  useEffect(() => {
    let alive = true;
    if (target !== undefined)
      inspect(
        matrix.slice(r * d, (r + 1) * d),
        matrix.slice(target * d, (target + 1) * d),
      )
        .then((v) => {
          if (alive) setArithmetic(v);
        })
        .catch((e) => {
          if (alive) setSimilarityError(String(e));
        });
    return () => {
      alive = false;
    };
  }, [r, target, matrix, d]);
  const similar =
    ranking?.order
      .slice(page * 12, (page + 1) * 12)
      .map((i) => ({ i, s: ranking.scores[i] })) ?? [];
  return (
    <>
      <Card idx={3} title="查表就是取出一整行数字" tag={`词表 × ${d}`}>
        <div className="note">
          词嵌入 <b>E[token_id]</b>{" "}
          是模型的可学习参数。点击任意单元格可选定一个维度，修改后会重新计算整个模型。这里只改变当前浏览器内存。
        </div>
        <MatrixView
          data={matrix}
          rows={f.c.vocabSize}
          cols={d}
          label="嵌入矩阵 E"
          rowLabels={result.vocab.map((v, i) => `${i} ${tokenLabel(v)}`)}
          selected={[r, col]}
          highlightRows={f.ids}
          onCell={(i, j) => {
            setRow(i);
            setCol(j);
            setValue(String(matrix[i * d + j]));
          }}
          onRow={setRow}
        />
        <div className="sec-h">当前向量 · {tokenLabel(result.vocab[r])}</div>
        <MatrixView
          data={matrix.slice(r * d, (r + 1) * d)}
          rows={1}
          cols={d}
          label={`E[${r}] 完整向量`}
          selected={[0, col]}
          onCell={(_, j) => {
            setCol(j);
            setValue(String(matrix[r * d + j]));
          }}
        />
        <div data-row data-space>
          <label>
            维度{" "}
            <input
              aria-label="嵌入维度"
              type="number"
              min={0}
              max={d - 1}
              value={col}
              onChange={(e) =>
                setCol(Math.max(0, Math.min(d - 1, Number(e.target.value))))
              }
            />
          </label>
          <label>
            新值{" "}
            <input
              aria-label="嵌入新值"
              type="number"
              step="0.05"
              value={value}
              onChange={(e) => setValue(e.target.value)}
            />
          </label>
          <button
            className="btn primary"
            disabled={!Number.isFinite(Number(value))}
            onClick={() => onEdit("embedding", r * d + col, Number(value))}
          >
            修改并重新计算
          </button>
        </div>
      </Card>
      <Card idx="↳" title="哪些向量与它方向最接近？" tag="余弦相似度">
        <div className="fml">cos(a,b) = (Σ aᵢbᵢ) / (√Σ aᵢ² · √Σ bᵢ²)</div>
        <div className="note warn">
          教学模型的初始权重未训练，靠近不等于语义相近。训练后再观察关系如何改变。
        </div>
        <table data-table aria-label="嵌入余弦排名">
          <thead>
            <tr>
              <th>词</th>
              <th>相似度</th>
              <th>逐项计算</th>
            </tr>
          </thead>
          <tbody>
            {similar.map((x) => (
              <tr key={x.i}>
                <td>
                  <button className="btn" onClick={() => setRow(x.i)}>
                    {tokenLabel(result.vocab[x.i])}
                  </button>
                </td>
                <td data-number>{fmt(x.s, "weight", true)}</td>
                <td>
                  <button className="btn" onClick={() => setTarget(x.i)}>
                    展开点积
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div data-row data-space>
          <button
            className="btn"
            disabled={!page}
            onClick={() => setPage((p) => p - 1)}
          >
            上一页
          </button>
          <small>
            全部其他词的排名 · {fmt(page * 12 + 1, "integer")}–
            {fmt(Math.min((page + 1) * 12, f.c.vocabSize - 1), "integer")} /{" "}
            {fmt(f.c.vocabSize - 1, "integer")}
          </small>
          <button
            className="btn"
            disabled={(page + 1) * 12 >= f.c.vocabSize - 1}
            onClick={() => setPage((p) => p + 1)}
          >
            下一页
          </button>
        </div>
        {similarityError && <div className="note warn">{similarityError}</div>}
        {target !== undefined && (
          <>
            <FormulaView
              arithmetic={arithmetic}
              label={`E[${r}] 与 E[${target}] 的点积`}
            />
            <div className="fml">
              向量 a 的平方和：
              {Array.from(
                matrix.slice(r * d, (r + 1) * d),
                (v) => `${fmt(v, "matrix")}²`,
              ).join(" + ")}
              <br />
              向量 b 的平方和：
              {Array.from(
                matrix.slice(target * d, (target + 1) * d),
                (v) => `${fmt(v, "matrix")}²`,
              ).join(" + ")}
              <div className="sum">
                点积 / 两个向量长度之积 ={" "}
                <b>{fmt(ranking?.scores[target] ?? 0, "matrix")}</b>
              </div>
            </div>
          </>
        )}
      </Card>
    </>
  );
}
