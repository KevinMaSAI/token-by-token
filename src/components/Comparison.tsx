import { useState } from "react";
import { run, type RunRequest, type RunResult } from "../state/store";
import { DEFAULT_SAMPLING, type Sampling } from "../core/sample";
import { Card, Control } from "./Shared";
import { ParamConsole } from "./ParamConsole";
import { GenText } from "./GenText";
import { ProbPanel } from "./ProbPanel";
export function Comparison({
  request,
  onError,
}: {
  request: RunRequest;
  onError: (s: string) => void;
}) {
  const [second, setSecond] = useState<Sampling>({
      ...DEFAULT_SAMPLING,
      method: "top-p",
      temperature: 0.5,
      seed: 73,
    }),
    [count, setCount] = useState(6),
    [busy, setBusy] = useState(false),
    [results, setResults] = useState<{ ids: number[]; r: RunResult }[]>([]);
  const generate = async () => {
    setBusy(true);
    setResults([]);
    try {
      const out = [];
      for (const sampling of [request.sampling, second]) {
        let r = await run({
            ...request,
            ids: undefined,
            patch: undefined,
            sampling,
            step: 0,
          }),
          ids = [...r.trace.ids],
          generated: number[] = [];
        for (let i = 0; i < count; i++) {
          if (i)
            r = await run({
              ...request,
              ids,
              sampling,
              patch: undefined,
              step: i,
            });
          const id = r.sample.selected;
          generated.push(id);
          ids.push(id);
          if (id === 2 && r.track !== "real") break;
        }
        out.push({ ids: generated, r });
        setResults([...out]);
      }
    } catch (e) {
      onError(String(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <Card idx="⇄" title="同一段输入，两种采样配置">
      <div className="note">
        A 使用右侧当前采样设置；B
        使用下面的设置。两组共享完全相同的模型权重和提示词，按各自的 token
        前缀逐步计算。
      </div>
      <ParamConsole value={second} onChange={setSecond} compact />
      <Control label="比较生成步数">
        <input
          type="number"
          min={1}
          max={16}
          value={count}
          onChange={(e) =>
            setCount(Math.max(1, Math.min(16, Number(e.target.value))))
          }
        />
      </Control>
      <button className="btn primary" disabled={busy} onClick={generate}>
        {busy ? "正在计算两组结果…" : "运行两组对比"}
      </button>
      <div data-grid data-space>
        {results.map((v, i) => (
          <div key={i}>
            <h3>
              配置 {i === 0 ? "A" : "B"} ·{" "}
              {i === 0 ? request.sampling.method : second.method}
            </h3>
            <GenText prompt={request.text} ids={v.ids} vocab={v.r.vocab} />
            <ProbPanel sample={v.r.sample} vocab={v.r.vocab} />
          </div>
        ))}
      </div>
    </Card>
  );
}
