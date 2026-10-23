import { useEffect, useState } from "react";
import { fmt, type NumberKind } from "../core/format";
export interface MatrixProps {
  data: Float32Array;
  rows: number;
  cols: number;
  label: string;
  kind?: NumberKind;
  selected?: [number, number];
  highlightRows?: number[];
  rowLabels?: string[];
  onCell?: (row: number, col: number) => void;
  onRow?: (row: number) => void;
  pageSize?: number;
}
export function MatrixView({
  data,
  rows,
  cols,
  label,
  kind = "matrix",
  selected,
  highlightRows = [],
  rowLabels,
  onCell,
  onRow,
  pageSize = 8,
}: MatrixProps) {
  const [page, setPage] = useState(0),
    [all, setAll] = useState(false),
    [colPage, setColPage] = useState(0);
  const start =
      Math.min(page, Math.max(0, Math.ceil(rows / pageSize) - 1)) * pageSize,
    end = all ? rows : Math.min(rows, start + pageSize),
    begin = all ? 0 : start,
    colStart =
      cols <= 64
        ? 0
        : Math.min(colPage, Math.max(0, Math.ceil(cols / 32) - 1)) * 32,
    colEnd = cols <= 64 ? cols : Math.min(cols, colStart + 32);
  useEffect(() => {
    if (selected && cols > 64) setColPage(Math.floor(selected[1] / 32));
  }, [selected?.[1], cols]);
  return (
    <div data-stack>
      <div data-between>
        <small>
          {label} ·{" "}
          <span data-number>
            {fmt(rows, "integer")} × {fmt(cols, "integer")}
          </span>
        </small>
        {rows > pageSize && (
          <div data-row>
            <button
              className="btn"
              disabled={all || start === 0}
              onClick={() => setPage((p) => p - 1)}
              aria-label={`${label} 上一组行`}
            >
              ←
            </button>
            <small data-number>
              {fmt(begin, "integer")}–{fmt(end - 1, "integer")}
            </small>
            <button
              className="btn"
              disabled={all || end >= rows}
              onClick={() => setPage((p) => p + 1)}
              aria-label={`${label} 下一组行`}
            >
              →
            </button>
            {rows <= 128 && (
              <button className="btn" onClick={() => setAll(!all)}>
                {all ? "分页" : "全部行"}
              </button>
            )}
          </div>
        )}
      </div>
      {cols > 64 && (
        <div data-row>
          <button
            className="btn"
            disabled={colStart === 0}
            onClick={() => setColPage((p) => p - 1)}
          >
            ← 前 32 列
          </button>
          <small>
            列 {fmt(colStart, "integer")}–{fmt(colEnd - 1, "integer")}
          </small>
          <button
            className="btn"
            disabled={colEnd >= cols}
            onClick={() => setColPage((p) => p + 1)}
          >
            后 32 列 →
          </button>
        </div>
      )}
      <div data-scroll>
        <table className="mx" aria-label={label}>
          <thead>
            <tr>
              <th>行 / 列</th>
              {Array.from({ length: colEnd - colStart }, (_, j) => (
                <th key={j}>{fmt(j + colStart, "integer")}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Array.from({ length: end - begin }, (_, r) => {
              const i = r + begin;
              return (
                <tr className="rl" key={i}>
                  <th>
                    <button
                      onClick={() => onRow?.(i)}
                      aria-label={`选择 ${label} 第 ${i} 行`}
                    >
                      {rowLabels?.[i] ?? fmt(i, "integer")}
                    </button>
                  </th>
                  {Array.from({ length: colEnd - colStart }, (_, offset) => {
                    const j = offset + colStart,
                      v = data[i * cols + j] ?? 0,
                      sel = selected?.[0] === i && selected[1] === j;
                    return (
                      <td
                        key={j}
                        className={`${v > 0 ? "pos" : v < 0 ? "neg" : ""} ${sel ? "hi2" : highlightRows.includes(i) ? "hi" : ""}`}
                      >
                        <button
                          title={`${label}[${i},${j}] = ${v}`}
                          aria-label={`${label} 第 ${i} 行第 ${j} 列 ${fmt(v, kind)}`}
                          onClick={() => onCell?.(i, j)}
                        >
                          {fmt(v, kind, true)}
                        </button>
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
