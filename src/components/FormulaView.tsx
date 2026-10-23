import { fmt } from "../core/format";
export interface Arithmetic {
  left: Float32Array;
  right: Float32Array;
  products: Float32Array;
  sum: number;
  bias: number;
  result: number;
}
export function FormulaView({
  arithmetic: a,
  label = "逐项相乘再求和",
  selected = -1,
}: {
  arithmetic: Arithmetic | null;
  label?: string;
  selected?: number;
}) {
  return (
    <div className="fml" aria-label={label}>
      <div className="lbl">{label} · 显示值经过舍入，运算使用完整精度</div>
      {!a ? (
        <span className="lbl">正在展开乘加…</span>
      ) : (
        <>
          <div data-cell-detail>
            {Array.from(a.left, (_, i) => (
              <div
                data-arithmetic-term
                key={i}
                className={selected === i ? "eq" : undefined}
              >
                <span className="lbl">{fmt(i, "integer")}</span>
                <span className="num">{fmt(a.left[i], "matrix")}</span>
                <span className="op">×</span>
                <span className="num">{fmt(a.right[i], "matrix")}</span>
                <span className="op">=</span>
                <span className="res">{fmt(a.products[i], "matrix")}</span>
              </div>
            ))}
          </div>
          <div className="sum">
            Σ 乘积 = <b>{fmt(a.sum, "matrix")}</b>
            <br />
            加偏置 {fmt(a.bias, "matrix")} →{" "}
            <b className="res">{fmt(a.result, "matrix")}</b>
          </div>
        </>
      )}
    </div>
  );
}
