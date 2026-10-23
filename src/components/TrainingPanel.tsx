import { useEffect, useRef, useState } from "react";
import type { Options } from "../core/model/config";
import type { Weights } from "../core/model/weights";
import { CORPUS } from "../core/model/train";
import { rpc, downloadJSON, type TrainPoint } from "../state/store";
import { fmt } from "../core/format";
import { Card, Control } from "./Shared";
import { MatrixView } from "./MatrixView";
export function TrainingPanel({
  options,
  onComplete,
  onError,
}: {
  options: Options;
  onComplete: (w: Weights, o: Options) => Promise<void>;
  onError: (s: string) => void;
}) {
  const [points, setPoints] = useState<TrainPoint[]>([]),
    [running, setRunning] = useState(false),
    [steps, setSteps] = useState(500),
    [lr, setLr] = useState(0.003),
    [clip, setClip] = useState(1),
    [status, setStatus] = useState("准备好后，所有层的参数都会参与真实训练。"),
    ref = useRef<Worker>();
  useEffect(() => () => ref.current?.terminate(), []);
  const start = async () => {
    if (running) return;
    try {
      const snapshot = await rpc<{
        weights: Weights;
        config: {
          vocabSize: number;
          d: number;
          layers: number;
          heads: number;
          ff: number;
        };
      }>("snapshot", {});
      if (
        snapshot.config.vocabSize !== 64 ||
        snapshot.config.d !== 32 ||
        snapshot.config.layers !== 2 ||
        snapshot.config.heads !== 2 ||
        snapshot.config.ff !== 64
      )
        throw new Error("请先在左侧切回微型模型，再训练内置语料。");
      ref.current?.terminate();
      const worker = new Worker(
        new URL("../workers/train.worker.ts", import.meta.url),
        { type: "module" },
      );
      ref.current = worker;
      setRunning(true);
      setPoints([]);
      setStatus("正在训练：交叉熵 → 手写反向传播 → 梯度裁剪 → Adam");
      const trainingOptions = { ...options };
      worker.onmessage = async (e) => {
        if (e.data.type === "point") setPoints((p) => [...p, e.data.point]);
        if (e.data.type === "error") {
          setRunning(false);
          setStatus("训练已停止");
          onError(e.data.error);
        }
        if (e.data.type === "done") {
          setRunning(false);
          setStatus(
            `${e.data.stopped ? "已暂停" : "训练完成"} · ${e.data.steps} 步 · ${fmt(e.data.elapsed / 1000, "logit")} 秒；参数已应用到解剖台。`,
          );
          try {
            await onComplete(e.data.weights, trainingOptions);
          } catch (err) {
            onError(String(err));
          }
        }
      };
      worker.onerror = (e) => {
        setRunning(false);
        onError(e.message);
      };
      worker.postMessage({
        type: "start",
        payload: {
          weights: snapshot.weights,
          options: trainingOptions,
          steps,
          lr,
          clip,
          seed: 42,
        },
      });
    } catch (e) {
      onError(String(e));
    }
  };
  const last = points.at(-1),
    max = Math.max(1, ...points.map((p) => p.loss)),
    normMax = Math.max(1, ...points.map((p) => p.gradNorm)),
    x = (i: number) => 34 + (i / Math.max(steps - 1, 1)) * 620,
    y = (v: number, scale: number) => 140 - (v / scale) * 115;
  return (
    <>
      <Card
        idx="训"
        title="在这个浏览器里，真正训练一个 Transformer"
        tag="2 层 / 2 头 / d=32"
      >
        <div className="note">
          没有深度学习框架，没有自动微分。
          <b>每一层的反向传播、梯度裁剪和 Adam 都由 TypeScript 手写</b>，在独立
          Worker
          中运行。初始教学随机权重会学习这个小语料，效果不等同于大型语言模型。
        </div>
        <div data-controls>
          <Control label="训练步数">
            <input
              aria-label="训练步数"
              type="number"
              min={1}
              max={5000}
              value={steps}
              disabled={running}
              onChange={(e) =>
                setSteps(
                  Math.max(
                    1,
                    Math.min(5000, Math.floor(Number(e.target.value))),
                  ),
                )
              }
            />
          </Control>
          <Control label="学习率">
            <input
              type="number"
              min={0.00001}
              max={0.1}
              step={0.001}
              value={lr}
              disabled={running}
              onChange={(e) =>
                setLr(Math.max(0.00001, Math.min(0.1, Number(e.target.value))))
              }
            />
          </Control>
          <Control label="全局梯度裁剪上限">
            <input
              type="number"
              min={0.01}
              max={100}
              step={0.5}
              value={clip}
              disabled={running}
              onChange={(e) =>
                setClip(Math.max(0.01, Math.min(100, Number(e.target.value))))
              }
            />
          </Control>
        </div>
        <div data-row>
          <button className="btn primary" disabled={running} onClick={start}>
            开始真实训练
          </button>
          <button
            className="btn"
            disabled={!running}
            onClick={() => {
              ref.current?.postMessage({ type: "stop" });
              setStatus("将在当前训练步结束后暂停并应用参数…");
            }}
          >
            暂停并应用
          </button>
          <button
            className="btn"
            disabled={running || !last}
            onClick={async () => {
              try {
                downloadJSON(
                  await rpc("export", options),
                  "token-by-token-trained.json",
                );
              } catch (e) {
                onError(String(e));
              }
            }}
          >
            导出训练权重 JSON
          </button>
        </div>
        <div className="note" role="status">
          {status}
        </div>
        {last && (
          <div data-grid>
            <div className="delta-i">
              <div className="lb">训练步 / 目标</div>
              <div className="vl">
                {fmt(last.step, "integer")} / {fmt(steps, "integer")}
              </div>
            </div>
            <div className="delta-i">
              <div className="lb">交叉熵损失</div>
              <div className="vl">{fmt(last.loss, "matrix")}</div>
            </div>
            <div className="delta-i">
              <div className="lb">梯度范数 → 裁剪后</div>
              <div className="vl">
                {fmt(last.gradNorm)} → {fmt(last.clippedNorm)}
              </div>
            </div>
            <div className="delta-i">
              <div className="lb">已用时间（秒）</div>
              <div className="vl">{fmt(last.elapsed / 1000, "logit")}</div>
            </div>
          </div>
        )}
      </Card>
      <Card idx="↳" title="损失随每一步生长" tag="真实观测值">
        <svg
          data-chart
          viewBox="0 0 680 170"
          role="img"
          aria-label="训练损失曲线"
        >
          <line x1="34" y1="140" x2="654" y2="140" />
          <line x1="34" y1="20" x2="34" y2="140" />
          <text x="4" y="26">
            {fmt(max, "logit")}
          </text>
          <text x="12" y="143">
            0
          </text>
          <text x="34" y="160">
            0 步
          </text>
          <text x="606" y="160">
            {steps} 步
          </text>
          <polyline
            points={points.map((p, i) => `${x(i)},${y(p.loss, max)}`).join(" ")}
          />
        </svg>
        <div className="note">
          每步随机抽取一条内置句子，损失会波动。下降表示模型更能预测这份小语料，不能据此推断泛化能力。
        </div>
        <div className="sec-h">每步梯度范数</div>
        <svg
          data-chart
          viewBox="0 0 680 170"
          role="img"
          aria-label="梯度范数曲线"
        >
          <line x1="34" y1="140" x2="654" y2="140" />
          <text x="4" y="26">
            {fmt(normMax, "logit")}
          </text>
          <polyline
            points={points
              .map((p, i) => `${x(i)},${y(p.gradNorm, normMax)}`)
              .join(" ")}
          />
        </svg>
        {last && (
          <>
            <div className="sec-h">实时嵌入参数热图</div>
            <MatrixView
              data={last.embedding}
              rows={64}
              cols={32}
              label="训练中的嵌入权重"
            />
          </>
        )}
      </Card>
      <Card idx="↳" title="训练语料与本次结构">
        <p data-small data-muted>
          预归一化 /
          后归一化、位置编码与激活函数沿用点击开始时的实验设置。训练固定使用内置英文优先
          BPE。暂停后的再次开始会保留权重并重置 Adam 动量。
        </p>
        <div className="fml">
          {CORPUS.map((line, i) => (
            <div key={i}>{line}</div>
          ))}
        </div>
      </Card>
    </>
  );
}
