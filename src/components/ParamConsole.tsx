import type { Sampling } from "../core/sample";
import { fmt } from "../core/format";
import { Control } from "./Shared";
export function ParamConsole({
  value: s,
  onChange,
  compact = false,
}: {
  value: Sampling;
  onChange: (s: Sampling) => void;
  compact?: boolean;
}) {
  return (
    <div data-controls>
      <Control label="采样方式">
        <select
          aria-label={compact ? "对照采样方式" : "采样方式"}
          value={s.method}
          onChange={(e) =>
            onChange({ ...s, method: e.target.value as Sampling["method"] })
          }
        >
          <option value="greedy">贪心（Greedy）</option>
          <option value="temperature">温度采样</option>
          <option value="top-k">前 K 项（Top-k）</option>
          <option value="top-p">累积概率（Top-p）</option>
        </select>
      </Control>
      <Control label="温度 T">
        <output>{fmt(s.temperature)}</output>
      </Control>
      <input
        aria-label={compact ? "对照温度" : "输出温度"}
        type="range"
        min={0.1}
        max={2}
        step={0.1}
        value={s.temperature}
        onChange={(e) =>
          onChange({ ...s, temperature: Number(e.target.value) })
        }
      />
      <div data-row>
        {[0.1, 0.5, 1, 2].map((t) => (
          <button
            key={t}
            className={`btn ${s.temperature === t ? "primary" : ""}`}
            onClick={() => onChange({ ...s, temperature: t })}
          >
            {fmt(t, "logit")}
          </button>
        ))}
      </div>
      {s.method === "top-k" && (
        <Control label="保留数量 K">
          <input
            type="number"
            min={1}
            max={64}
            value={s.k}
            onChange={(e) =>
              onChange({
                ...s,
                k: Math.max(1, Math.min(64, Number(e.target.value))),
              })
            }
          />
        </Control>
      )}
      {s.method === "top-p" && (
        <Control label="累积阈值 p">
          <input
            type="number"
            min={0.01}
            max={1}
            step={0.05}
            value={s.p}
            onChange={(e) =>
              onChange({
                ...s,
                p: Math.max(0.01, Math.min(1, Number(e.target.value))),
              })
            }
          />
        </Control>
      )}
      <Control label="随机种子">
        <input
          aria-label={compact ? "对照随机种子" : "随机种子"}
          type="number"
          min={0}
          max={2147483647}
          value={s.seed}
          onChange={(e) =>
            onChange({
              ...s,
              seed: Math.max(
                0,
                Math.min(2147483647, Math.floor(Number(e.target.value))),
              ),
            })
          }
        />
      </Control>
    </div>
  );
}
