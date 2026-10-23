import type { ReactNode } from "react";
export function Card({
  idx,
  title,
  tag,
  children,
}: {
  idx: string | number;
  title: string;
  tag?: string;
  children: ReactNode;
}) {
  return (
    <section className="card">
      <div className="card-h">
        <span className="idx">{idx}</span>
        <h2>{title}</h2>
        {tag && <span className="tag">{tag}</span>}
      </div>
      {children}
    </section>
  );
}
export function Switch({
  label,
  value,
  onChange,
}: {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div data-between>
      <label>{label}</label>
      <button
        type="button"
        className={`sw ${value ? "on" : ""}`}
        role="switch"
        aria-checked={value}
        aria-label={label}
        onClick={() => onChange(!value)}
      />
    </div>
  );
}
export function Control({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <label data-between>
      {label}
      {children}
    </label>
  );
}
export const STAGES = [
  "字节",
  "分词",
  "查表",
  "嵌入",
  "位置",
  "注意力",
  "归一化",
  "FFN",
  "堆叠",
  "输出",
];
export const STAGE_NAMES = [
  "原始文字 → UTF-8 字节",
  "字节 → BPE 分词",
  "Token ID → 词表",
  "词表 → 嵌入向量",
  "把位置带进计算",
  "谁在看谁 · 自注意力",
  "残差相加与归一化",
  "逐位置的非线性变换",
  "穿过每一层残差流",
  "从分数到下一个词",
];
