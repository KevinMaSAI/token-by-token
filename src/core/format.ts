export type NumberKind = "prob" | "logit" | "weight" | "matrix" | "integer";
const decimals: Record<NumberKind, number> = {
  prob: 3,
  logit: 2,
  weight: 3,
  matrix: 4,
  integer: 0,
};
const formatters = new Map<string, Intl.NumberFormat>();
export function fmt(
  value: number,
  kind: NumberKind = "weight",
  signed = false,
) {
  if (!Number.isFinite(value))
    return value === -Infinity ? "−∞" : value === Infinity ? "+∞" : "—";
  const key = kind + signed;
  if (!formatters.has(key))
    formatters.set(
      key,
      new Intl.NumberFormat("en-US", {
        minimumFractionDigits: decimals[kind],
        maximumFractionDigits: decimals[kind],
        signDisplay: signed ? "exceptZero" : "auto",
        useGrouping: true,
      }),
    );
  return formatters.get(key)!.format(Object.is(value, -0) ? 0 : value);
}
export const tokenLabel = (s: string) =>
  s === " "
    ? "␠"
    : s === "\n"
      ? "↵"
      : s === "<unk>"
        ? "未知"
        : s === "<bos>"
          ? "起始"
          : s === "<eos>"
            ? "结束"
            : s;
