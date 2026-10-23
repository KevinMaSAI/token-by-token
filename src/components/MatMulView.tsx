import { useEffect, useState } from "react";
import { MatrixView } from "./MatrixView";
import { FormulaView, type Arithmetic } from "./FormulaView";
import { inspect } from "../state/store";
import { fmt } from "../core/format";
export interface MatMulProps {
  a: Float32Array;
  b: Float32Array;
  out: Float32Array;
  m: number;
  k: number;
  n: number;
  label: string;
  aLabel?: string;
  bLabel?: string;
  bias?: Float32Array;
  initialRow?: number;
  initialCol?: number;
}
export function MatMulView({
  a,
  b,
  out,
  m,
  k,
  n,
  label,
  aLabel = "输入 A",
  bLabel = "权重 B",
  bias,
  initialRow = 0,
  initialCol = 0,
}: MatMulProps) {
  const [row, setRow] = useState(initialRow),
    [col, setCol] = useState(initialCol),
    [arithmetic, setArithmetic] = useState<Arithmetic | null>(null),
    [error, setError] = useState("");
  const r = Math.min(row, m - 1),
    c = Math.min(col, n - 1);
  useEffect(() => {
    setRow(initialRow);
  }, [initialRow]);
  useEffect(() => {
    let live = true;
    setArithmetic(null);
    inspect(
      a.slice(r * k, (r + 1) * k),
      Float32Array.from({ length: k }, (_, i) => b[i * n + c]),
      bias?.[c] ?? 0,
    )
      .then((v) => {
        if (live) setArithmetic(v);
      })
      .catch((e) => {
        if (live) setError(String(e));
      });
    return () => {
      live = false;
    };
  }, [a, b, bias, r, c, k, n]);
  return (
    <div data-matmul>
      <div data-between>
        <h3>{label}</h3>
        <span className="tag">
          {fmt(m, "integer")}×{fmt(k, "integer")} · {fmt(k, "integer")}×
          {fmt(n, "integer")}
        </span>
      </div>
      <div data-operand>
        <span>
          {aLabel} 第 <b>{fmt(r, "integer")}</b> 行
        </span>
        <b>→ × →</b>
        <span>
          {bLabel} 第 <b>{fmt(c, "integer")}</b> 列
        </span>
        <b>→ Σ →</b>
        <span className="eq">
          结果 [{fmt(r, "integer")}, {fmt(c, "integer")}]
        </span>
      </div>
      <div data-row>
        <label>
          输出行{" "}
          <input
            type="number"
            min={0}
            max={m - 1}
            value={r}
            onChange={(e) =>
              setRow(Math.max(0, Math.min(m - 1, Number(e.target.value))))
            }
          />
        </label>
        <label>
          输出列{" "}
          <input
            type="number"
            min={0}
            max={n - 1}
            value={c}
            onChange={(e) =>
              setCol(Math.max(0, Math.min(n - 1, Number(e.target.value))))
            }
          />
        </label>
        <small>点结果单元格，追踪这一项</small>
      </div>
      <MatrixView
        data={out}
        rows={m}
        cols={n}
        label={`${label} 结果`}
        selected={[r, c]}
        onCell={(i, j) => {
          setRow(i);
          setCol(j);
        }}
        pageSize={4}
      />
      <details>
        <summary>展开完整输入与权重矩阵</summary>
        <MatrixView
          data={a}
          rows={m}
          cols={k}
          label={aLabel}
          highlightRows={[r]}
        />
        <MatrixView
          data={b}
          rows={k}
          cols={n}
          label={bLabel}
          selected={[0, c]}
        />
      </details>
      {error ? (
        <div data-error>{error}</div>
      ) : (
        <FormulaView
          arithmetic={arithmetic}
          label={`${label}：结果[${r},${c}] 的每一个乘积`}
        />
      )}
    </div>
  );
}
