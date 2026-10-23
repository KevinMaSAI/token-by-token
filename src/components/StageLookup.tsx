import { useState } from "react";
import { fmt, tokenLabel } from "../core/format";
import { Card } from "./Shared";
export function StageLookup({
  vocab,
  ids,
}: {
  vocab: string[];
  ids: number[];
}) {
  const [search, setSearch] = useState(""),
    [page, setPage] = useState(0),
    found = vocab
      .map((v, i) => ({ v, i }))
      .filter((x) => x.v.includes(search) || String(x.i) === search),
    start = Math.min(page, Math.max(0, Math.ceil(found.length / 64) - 1)) * 64;
  return (
    <Card
      idx={2}
      title="每个 token 对应词表里的一行"
      tag={`${vocab.length} 项`}
    >
      <div className="note">
        ID 只是行号，不代表大小或相似性。<b>模型只接收 ID</b>
        ，下一步用它提取嵌入矩阵的一行。微型词表外的片段映射到 ID
        0（未知），原文仍保留在分词显示中。
      </div>
      <input
        type="search"
        placeholder="搜索 token 或精确 ID…"
        aria-label="搜索词表"
        value={search}
        onChange={(e) => {
          setSearch(e.target.value);
          setPage(0);
        }}
      />
      <div data-scroll style={{ maxHeight: 600 }}>
        <table data-table>
          <thead>
            <tr>
              <th>词表 ID</th>
              <th>符号</th>
              <th>当前序列出现位置</th>
            </tr>
          </thead>
          <tbody>
            {found.slice(start, start + 64).map(({ v, i }) => (
              <tr key={i} aria-selected={ids.includes(i)}>
                <td data-number>{fmt(i, "integer")}</td>
                <td data-mono>{tokenLabel(v)}</td>
                <td data-number>
                  {ids
                    .map((id, p) => (id === i ? p : -1))
                    .filter((p) => p >= 0)
                    .join(", ") || "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {found.length > 64 && (
        <div data-row>
          <button
            className="btn"
            disabled={!start}
            onClick={() => setPage((p) => p - 1)}
          >
            上一页
          </button>
          <span data-number>
            {fmt(start, "integer")} / {fmt(found.length, "integer")}
          </span>
          <button
            className="btn"
            disabled={start + 64 >= found.length}
            onClick={() => setPage((p) => p + 1)}
          >
            下一页
          </button>
        </div>
      )}
    </Card>
  );
}
