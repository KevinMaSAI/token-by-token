import { useEffect, useState } from "react";
import type { BPEOptions, Tokenization } from "../core/tokenizer/bpe";
import { fmt } from "../core/format";
import { Card, Control, Switch } from "./Shared";
import { TokenBar } from "./TokenBar";
export function StageBPE({
  data,
  options,
  onOptions,
  choices,
  onChoices,
  selected,
  onSelect,
  locked = false,
}: {
  data: Tokenization;
  options: BPEOptions;
  onOptions: (o: BPEOptions) => void;
  choices: number[];
  onChoices: (v: number[]) => void;
  selected: number;
  onSelect: (i: number) => void;
  locked?: boolean;
}) {
  const [round, setRound] = useState(0),
    [playing, setPlaying] = useState(false),
    [speed, setSpeed] = useState(800);
  useEffect(() => {
    setRound(0);
    setPlaying(false);
  }, [data.text, options]);
  const r = Math.min(round, data.rounds.length),
    state = data.rounds[r],
    pieces = state?.before ?? data.tokens;
  useEffect(() => {
    if (!playing) return;
    if (r >= data.rounds.length) {
      setPlaying(false);
      return;
    }
    const t = setTimeout(() => setRound((x) => x + 1), speed);
    return () => clearTimeout(t);
  }, [playing, r, data.rounds.length, speed]);
  const update = (o: Partial<BPEOptions>) => onOptions({ ...options, ...o });
  return (
    <>
      <Card idx={1} title="一步一步，合并最高优先级的相邻对" tag="手写字节 BPE">
        <div className="note">
          每一轮只合并 <b>rank 最小</b>{" "}
          的合法相邻对。下表可手动选择另一个合法对，之后继续按优先级合并；这属于算法干预，会改变分词路径。
        </div>
        <div data-controls>
          <Control label="合并表">
            <select
              disabled={locked}
              value={options.set}
              onChange={(e) =>
                update({ set: e.target.value as BPEOptions["set"] })
              }
            >
              <option value="english">英文词优先</option>
              <option value="chinese">中文逐字</option>
              <option value="character">字符级</option>
            </select>
          </Control>
          {locked ? (
            <div className="note warn">
              预训练模型固定使用随权重提供的原始 BPE 规则。
            </div>
          ) : (
            <>
              <Switch
                label="允许跨标点边界"
                value={options.crossPunctuation}
                onChange={(v) =>
                  update({
                    crossPunctuation: v,
                    splitPunctuation: v ? false : options.splitPunctuation,
                  })
                }
              />
              <Switch
                label="数字处强制分开"
                value={options.splitDigits}
                onChange={(v) => update({ splitDigits: v })}
              />
              <Switch
                label="标点处强制分开"
                value={options.splitPunctuation}
                onChange={(v) => update({ splitPunctuation: v })}
              />
              <Switch
                label="空格处强制分开"
                value={options.splitSpaces}
                onChange={(v) => update({ splitSpaces: v })}
              />
            </>
          )}
        </div>
        <div data-row>
          <button
            className="btn primary"
            disabled={r >= data.rounds.length}
            onClick={() => setPlaying((v) => !v)}
          >
            {playing ? "暂停合并" : "慢速合并"}
          </button>
          <button
            className="btn"
            disabled={r === 0}
            onClick={() => {
              setPlaying(false);
              setRound((x) => x - 1);
            }}
          >
            上一步
          </button>
          <button
            className="btn"
            disabled={!state}
            onClick={() => {
              setPlaying(false);
              setRound((x) => x + 1);
            }}
          >
            下一步
          </button>
          <button
            className="btn"
            onClick={() => {
              setRound(0);
              setPlaying(false);
              onChoices([]);
            }}
          >
            重放
          </button>
          <small>
            第 <span data-number>{fmt(r, "integer")}</span> /{" "}
            {fmt(data.rounds.length, "integer")} 轮
          </small>
        </div>
        <label data-between>
          每步间隔 <output>{fmt(speed, "integer")} 毫秒</output>
        </label>
        <input
          type="range"
          min={150}
          max={2000}
          step={50}
          value={speed}
          onChange={(e) => setSpeed(Number(e.target.value))}
          aria-label="合并播放速度"
        />
        <div data-round data-token-list>
          {pieces.map((p, i) => (
            <span
              className={`tok ${state && (i === state.selected || i === state.selected + 1) ? "pick" : ""}`}
              key={i}
            >
              {p.raw}
              <span className="id">
                {fmt(p.start, "integer")}–{fmt(p.end, "integer")}
              </span>
            </span>
          ))}
        </div>
        {state ? (
          <>
            <div data-scroll style={{ maxHeight: 260 }}>
              <table data-table>
                <thead>
                  <tr>
                    <th>相邻符号</th>
                    <th>优先级</th>
                    <th>本轮选择</th>
                  </tr>
                </thead>
                <tbody>
                  {state.pairs.map((p) => (
                    <tr
                      key={p.index}
                      aria-selected={p.index === state.selected}
                    >
                      <td data-mono>
                        {p.left} + {p.right}
                      </td>
                      <td data-number>
                        {p.rank === null ? "无规则" : fmt(p.rank, "integer")}
                      </td>
                      <td>
                        {!p.allowed ? (
                          "边界禁止"
                        ) : p.rank === null ? (
                          "不合并"
                        ) : (
                          <button
                            className="btn"
                            onClick={() => {
                              setPlaying(false);
                              onChoices([
                                ...choices.slice(0, r),
                                ...Array.from(
                                  { length: Math.max(0, r - choices.length) },
                                  () => -1,
                                ),
                                p.index,
                              ]);
                            }}
                          >
                            {p.index === state.selected
                              ? "当前选中"
                              : "改合并此对"}
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="fml">
              <span className="lbl">合并前</span>{" "}
              {state.before.map((p) => p.raw).join(" | ")}
              <br />
              <span className="res">合并后</span>{" "}
              {state.after.map((p) => p.raw).join(" | ")}
              <div className="sum">
                {state.manual ? "手动干预轮次" : "按最小 rank 选择"} · 合并了第{" "}
                <b>{fmt(state.selected, "integer")}</b> 对
              </div>
            </div>
          </>
        ) : (
          <div className="note">
            已经没有可合并的相邻对。现在按最终符号查模型词表。
          </div>
        )}
      </Card>
      <Card idx="↳" title="最终分词 · 每块宽度反映字节长度">
        <TokenBar data={data} selected={selected} onSelect={onSelect} />
        <div data-grid data-space>
          <div className="delta-i">
            <div className="lb">Token / 字符</div>
            <div className="vl">
              {fmt(data.tokens.length, "integer")} /{" "}
              {fmt(data.charCount, "integer")}
            </div>
          </div>
          <div className="delta-i">
            <div className="lb">每字符平均 token</div>
            <div className="vl">{fmt(data.ratio, "prob")}</div>
          </div>
        </div>
        <div className="note">
          字节长度分布：
          {Object.entries(data.histogram)
            .map(([b, n]) => `${b} 字节 × ${n}`)
            .join("；") || "空输入"}
          。中文逐字只合并汉字内部字节；字符级另外禁止任何多字符词合并。
        </div>
      </Card>
    </>
  );
}
