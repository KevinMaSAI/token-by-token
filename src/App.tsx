import { useCallback, useEffect, useRef, useState } from "react";
import { DEFAULT_OPTIONS, type Options, type Patch } from "./core/model/config";
import type { Weights } from "./core/model/weights";
import { DEFAULT_BPE, type BPEOptions } from "./core/tokenizer/bpe";
import { DEFAULT_SAMPLING, type Sampling } from "./core/sample";
import { fmt } from "./core/format";
import {
  run,
  rpc,
  downloadJSON,
  type RunResult,
  type RunRequest,
  type SavedStep,
  type SerializedModel,
} from "./state/store";
import { STAGES, STAGE_NAMES, Control } from "./components/Shared";
import { StageBytes } from "./components/StageBytes";
import { StageBPE } from "./components/StageBPE";
import { StageLookup } from "./components/StageLookup";
import { StageEmbedding } from "./components/StageEmbedding";
import { StagePosEnc } from "./components/StagePosEnc";
import {
  StageAttention,
  type AttentionSelection,
} from "./components/StageAttention";
import { StageLayerNorm } from "./components/StageLayerNorm";
import { StageFFN } from "./components/StageFFN";
import { StageResidual } from "./components/StageResidual";
import { StageOutput } from "./components/StageOutput";
import { TokenBar } from "./components/TokenBar";
import { GenText } from "./components/GenText";
import { Timeline } from "./components/Timeline";
import { ParamConsole } from "./components/ParamConsole";
import { ProbPanel } from "./components/ProbPanel";
import { Perturbation } from "./components/Perturbation";
import { ManualCompute } from "./components/ManualCompute";
import { TrainingPanel } from "./components/TrainingPanel";
import { Comparison } from "./components/Comparison";
import { TensorInspector } from "./components/TensorInspector";

export default function App() {
  const [text, setText] = useState("你好 hello"),
    [stage, setStage] = useState(0),
    [view, setView] = useState<"stages" | "training">("stages");
  const [options, setOptions] = useState<Options>({ ...DEFAULT_OPTIONS }),
    [bpe, setBpe] = useState<BPEOptions>({ ...DEFAULT_BPE }),
    [sampling, setSampling] = useState<Sampling>({ ...DEFAULT_SAMPLING });
  const [result, setResult] = useState<RunResult>(),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(true),
    [revision, setRevision] = useState(0),
    [maxSeq, setMaxSeq] = useState(32),
    [patch, setPatch] = useState<Patch>(),
    [choices, setChoices] = useState<number[]>([]);
  const [position, setPosition] = useState(0),
    [selection, setSelection] = useState<AttentionSelection>({
      layer: 0,
      head: 0,
      query: 0,
      key: 0,
    }),
    [hover, setHover] = useState(-1),
    [manual, setManual] = useState(false);
  const [generated, setGenerated] = useState<number[]>([]),
    [contextIds, setContextIds] = useState<number[]>(),
    [history, setHistory] = useState<SavedStep[]>([]),
    [replay, setReplay] = useState(-1),
    [playing, setPlaying] = useState(false),
    [speed, setSpeed] = useState(1000),
    [maxSteps, setMaxSteps] = useState(12),
    [ended, setEnded] = useState(false);
  const requestRef = useRef(0),
    runRef = useRef(0),
    trained = useRef<{ weights: Weights; options: Options }>(),
    fileInput = useRef<HTMLInputElement>(null);
  const clearGeneration = () => {
    setPlaying(false);
    setGenerated([]);
    setContextIds(undefined);
    setHistory([]);
    setReplay(-1);
    setEnded(false);
  };
  const changeText = (s: string) => {
    setText(s);
    setChoices([]);
    setPatch(undefined);
    clearGeneration();
  };
  const request: RunRequest = {
    text,
    options,
    bpe,
    sampling,
    maxSeq,
    ids: contextIds,
    patch,
    choices,
    step: generated.length,
  };
  const requestCurrent = useRef(request);
  requestCurrent.current = request;
  useEffect(() => {
    if (replay >= 0) {
      setBusy(false);
      return;
    }
    const serial = ++requestRef.current;
    setBusy(true);
    const timer = setTimeout(() => {
      run({
        text,
        options,
        bpe,
        sampling,
        maxSeq,
        ids: contextIds,
        patch,
        choices,
        step: generated.length,
      })
        .then((r) => {
          if (serial === requestRef.current) {
            setResult(r);
            setBusy(false);
            setError("");
            setSelection((s) => ({
              ...s,
              query: Math.min(s.query, r.trace.ids.length - 1),
              key: Math.min(s.key, r.trace.ids.length - 1),
            }));
          }
        })
        .catch((e) => {
          if (serial === requestRef.current) {
            setError(String(e));
            setBusy(false);
            setPlaying(false);
          }
        });
    }, 140);
    return () => clearTimeout(timer);
  }, [
    text,
    options,
    bpe,
    sampling,
    maxSeq,
    contextIds,
    patch,
    choices,
    revision,
    generated.length,
    replay,
  ]);
  const generate = useCallback(() => {
    if (!result || busy || ended || generated.length >= maxSteps) return;
    setBusy(true);
    const id = result.sample.selected,
      next = [...generated, id];
    setHistory((h) => [
      ...h.slice(0, replay >= 0 ? replay : h.length),
      {
        request: { ...requestCurrent.current },
        result,
        generatedIds: next,
        sampling: { ...sampling },
        options: { ...options },
      },
    ]);
    setReplay(-1);
    setGenerated(next);
    setContextIds([
      ...(contextIds ??
        (result.tokenization.ids.length
          ? result.tokenization.ids
          : [result.trace.c.bosId ?? 1])),
      id,
    ]);
    setSelection((s) => ({
      ...s,
      query: Math.min(result.trace.ids.length, result.trace.c.maxSeq - 1),
    }));
    if (id === (result.trace.c.eosId ?? 2) || next.length >= maxSteps) {
      setPlaying(false);
      setEnded(id === (result.trace.c.eosId ?? 2));
    }
  }, [
    result,
    busy,
    ended,
    generated,
    contextIds,
    maxSteps,
    replay,
    sampling,
    options,
  ]);
  useEffect(() => {
    if (!playing || busy) return;
    const timer = setTimeout(generate, speed);
    return () => clearTimeout(timer);
  }, [playing, busy, speed, generate]);
  const restore = (i: number) => {
    const saved = history[i];
    if (!saved) return;
    setPlaying(false);
    requestRef.current++;
    setResult(saved.result);
    setSampling(saved.sampling);
    setOptions(saved.options);
    setPatch(saved.request.patch);
    setBpe(saved.request.bpe);
    setChoices(saved.request.choices ?? []);
    setMaxSeq(saved.request.maxSeq);
    setGenerated(saved.generatedIds.slice(0, -1));
    setContextIds(saved.request.ids);
    setReplay(i);
    setStage(9);
    setView("stages");
    setBusy(false);
    setEnded(false);
  };
  const changeOptions = (o: Options) => {
    setOptions(o);
    setReplay(-1);
    setPlaying(false);
  };
  const edit = async (key: string, index: number, value: number) => {
    try {
      await rpc("edit", { key, index, value });
      setPatch(undefined);
      clearGeneration();
      setRevision((v) => v + 1);
    } catch (e) {
      setError(String(e));
    }
  };
  const chooseTrack = async (track: string) => {
    setPlaying(false);
    setBusy(true);
    try {
      if (track === "mini") {
        await rpc("reset", {});
        setOptions({ ...DEFAULT_OPTIONS });
        setMaxSeq(32);
      } else if (track === "trained") {
        if (!trained.current) {
          setView("training");
          setBusy(false);
          return;
        }
        await rpc("replace", trained.current.weights);
        setOptions({ ...trained.current.options });
      } else {
        const loaded = await rpc<{ options?: Options }>("loadReal", {
          base: new URL(
            "models/",
            new URL(import.meta.env.BASE_URL, location.href),
          ).href,
        });
        if (loaded.options) setOptions(loaded.options);
      }
      setPatch(undefined);
      clearGeneration();
      setRevision((v) => v + 1);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  };
  const trainingComplete = async (w: Weights, o: Options) => {
    trained.current = { weights: w, options: o };
    await rpc("replace", w);
    setOptions({ ...o });
    setBpe({ ...DEFAULT_BPE });
    setPatch(undefined);
    clearGeneration();
    setRevision((v) => v + 1);
  };
  const importModel = async (file?: File) => {
    if (!file) return;
    try {
      if (file.size > 250 * 1024 * 1024)
        throw new Error("模型文件超过 250 MB，请使用更小模型");
      setBusy(true);
      const loaded = await rpc<{ options?: Options }>(
        "import",
        JSON.parse(await file.text()),
      );
      if (loaded.options) setOptions(loaded.options);
      setPatch(undefined);
      clearGeneration();
      setRevision((v) => v + 1);
      setError("");
    } catch (e) {
      setError(String(e));
      setBusy(false);
    }
    if (fileInput.current) fileInput.current.value = "";
  };
  const exportModel = async () => {
    try {
      downloadJSON(
        await rpc<SerializedModel>("export", options),
        "token-by-token-weights.json",
      );
    } catch (e) {
      setError(String(e));
    }
  };
  const navigate = (i: number) => {
    setStage(i);
    setView("stages");
    setPlaying(false);
    runRef.current++;
    window.scrollTo({ top: 0, behavior: "instant" });
  };
  const currentPosition = Math.min(
      position,
      (result?.trace.ids.length ?? 1) - 1,
    ),
    displayIds =
      replay >= 0 ? (history[replay]?.generatedIds ?? generated) : generated;
  return (
    <>
      <header className="topbar">
        <div data-row>
          <span className="brand-mark">L</span>
          <strong data-brand-name>Token by Token</strong>
        </div>
        <nav aria-label="处理阶段">
          {STAGES.map((name, i) => (
            <span data-row key={name} style={{ gap: 4 }}>
              {i > 0 && <span className="step-sep" />}
              <button
                className={`step ${stage === i && view === "stages" ? "on" : i < stage ? "done" : ""}`}
                onClick={() => navigate(i)}
                aria-current={
                  stage === i && view === "stages" ? "step" : undefined
                }
              >
                {i} {name}
              </button>
            </span>
          ))}
        </nav>
        <div data-row>
          <span className="pill">
            <b>{result?.trace.c.layers ?? 2}</b> 层
          </span>
          <span className="pill">
            d = <b>{result?.trace.c.d ?? 32}</b>
          </span>
          <span className="pill">
            <b>{result?.trace.c.heads ?? 2}</b> 头
          </span>
        </div>
      </header>
      <div data-layout>
        <aside data-sidebar="left">
          <div className="sec-h">实验输入</div>
          <textarea
            className="ta"
            aria-label="输入文字"
            value={text}
            maxLength={512}
            onChange={(e) => changeText(e.target.value)}
            placeholder="写下任何文字…"
          />
          <div data-row data-space>
            <button className="btn" onClick={() => changeText("你好 hello")}>
              中英混合
            </button>
            <button className="btn" onClick={() => changeText("0 1 2 3")}>
              数字序列
            </button>
          </div>
          <div className="sec-h">分词结果</div>
          {result && (
            <TokenBar
              data={result.tokenization}
              selected={currentPosition}
              onSelect={(p) => {
                setPosition(Math.max(0, p - result.contextStart));
                setSelection((s) => ({
                  ...s,
                  query: Math.min(
                    Math.max(0, p - result.contextStart),
                    result.trace.ids.length - 1,
                  ),
                }));
              }}
            />
          )}
          <div data-between data-space>
            <small>字符 / token</small>
            <output data-small>
              {fmt(result?.tokenization.charCount ?? 0, "integer")} /{" "}
              {fmt(result?.tokenization.tokens.length ?? 0, "integer")}
            </output>
          </div>
          <div className="sec-h" data-nav-label>
            解剖路径
          </div>
          <div data-stages>
            {STAGES.map((s, i) => (
              <button
                key={s}
                onClick={() => navigate(i)}
                aria-current={stage === i && view === "stages"}
              >
                <span
                  className={`tl-n ${i === stage ? "cur" : i < stage ? "done" : ""}`}
                >
                  {i}
                </span>
                <span>{s}</span>
                <small>{i === stage ? "正在观察" : "↗"}</small>
              </button>
            ))}
          </div>
          <div className="sec-h">逐词生成</div>
          <GenText prompt={text} ids={displayIds} vocab={result?.vocab ?? []} />
          <div data-row data-space>
            <button
              className="btn primary"
              disabled={busy || ended || generated.length >= maxSteps}
              onClick={generate}
            >
              生成一步
            </button>
            <button
              className="btn"
              disabled={ended || generated.length >= maxSteps}
              onClick={() => setPlaying((v) => !v)}
            >
              {playing ? "暂停" : "连续生成"}
            </button>
            <button className="btn" onClick={clearGeneration}>
              重置
            </button>
          </div>
          <div data-controls>
            <Control label="生成速度">
              <select
                aria-label="生成速度"
                value={speed}
                onChange={(e) => setSpeed(Number(e.target.value))}
              >
                <option value={1000}>慢速 · 每秒一步</option>
                <option value={100}>连续 · 计算完成即继续</option>
              </select>
            </Control>
            <Control label="最多生成步数">
              <input
                type="number"
                min={1}
                max={32}
                value={maxSteps}
                onChange={(e) =>
                  setMaxSteps(Math.max(1, Math.min(32, Number(e.target.value))))
                }
              />
            </Control>
          </div>
          <Timeline steps={history} selected={replay} onSelect={restore} />
          {replay >= 0 && (
            <div className="note">
              回放第 {replay + 1}{" "}
              步：数值、参数和随机数均已恢复。再次生成会从此处分支。
            </div>
          )}
          {ended && <div className="note">模型生成了结束标记，已停止。</div>}
          <div className="sec-h">模型来源</div>
          <select
            data-full
            aria-label="模型来源"
            value={result?.track ?? "mini"}
            onChange={(e) => chooseTrack(e.target.value)}
          >
            <option value="mini">① 微型教学模型 · 零下载</option>
            <option value="trained">② 浏览器训练的模型</option>
            <option value="real">③ 加载预训练模型</option>
          </select>
          <div data-row data-space>
            <button
              className="btn"
              onClick={() => {
                setView("training");
                setPlaying(false);
              }}
            >
              训练一个模型
            </button>
            <button className="btn" onClick={() => fileInput.current?.click()}>
              导入 JSON
            </button>
          </div>
          <input
            type="file"
            accept=".json,application/json"
            hidden
            ref={fileInput}
            onChange={(e) => importModel(e.target.files?.[0])}
          />
          <p data-small data-faint data-space>
            所有文字与计算都留在浏览器。教学初始权重未训练，生成内容没有语言质量保证。
          </p>
        </aside>
        <main>
          <div data-hero>
            <div data-between>
              <div data-eyebrow>可检查的语言模型 · 计算解剖台</div>
              <span className="pill">
                {busy ? "正在计算" : "计算完成"}{" "}
                <b>{result ? fmt(result.elapsed, "logit") : "—"}</b> 毫秒
              </span>
            </div>
            <h1>把下一个词，拆到每一次乘加。</h1>
            <p>选择一个 token，追踪每一层；点击一个数，看它从哪里来。</p>
            <div data-row data-space>
              <span className="pill">
                参数 <b>{fmt(result?.parameterCount ?? 0, "integer")}</b>
              </span>
              <span className="pill">
                数值张量{" "}
                <b>{fmt(result?.trace.tensors.length ?? 0, "integer")}</b>
              </span>
              <span className="pill">
                {result?.track === "trained"
                  ? "已训练"
                  : result?.track === "real"
                    ? "已载入模型"
                    : "固定种子 · 教学随机权重"}
              </span>
            </div>
          </div>
          {error && (
            <div className="note warn" role="alert">
              {error}
            </div>
          )}
          {result?.truncated ? (
            <div className="note warn">
              输入超出上下文限制，当前计算最后 {result.trace.ids.length} 个
              token；省略了前 {result.truncated} 个。生成始终追加真实 token
              ID，不重新分词。
            </div>
          ) : null}
          {view === "stages" ? (
            <>
              <div data-stage-nav>
                <span>
                  {String(stage).padStart(2, "0")} / 09 · {STAGE_NAMES[stage]}
                </span>
                <button
                  className="btn"
                  disabled={stage === 0}
                  onClick={() => navigate(stage - 1)}
                >
                  ← 上一步
                </button>
                <button
                  className="btn"
                  disabled={stage === 9}
                  onClick={() => navigate(stage + 1)}
                >
                  下一步 →
                </button>
              </div>
              {!result ? (
                <section className="card">
                  <div className="note" role="status">
                    正在初始化字节分词器与两层 Transformer…
                  </div>
                </section>
              ) : (
                <>
                  {stage === 0 && <StageBytes data={result.tokenization} />}
                  {stage === 1 && (
                    <StageBPE
                      data={result.tokenization}
                      options={bpe}
                      onOptions={(o) => {
                        setBpe(o);
                        setChoices([]);
                        clearGeneration();
                      }}
                      choices={choices}
                      onChoices={(v) => {
                        setChoices(v);
                        clearGeneration();
                      }}
                      selected={currentPosition}
                      onSelect={setPosition}
                      locked={result.track === "real"}
                    />
                  )}
                  {stage === 2 && (
                    <StageLookup vocab={result.vocab} ids={result.trace.ids} />
                  )}
                  {stage === 3 && (
                    <StageEmbedding result={result} onEdit={edit} />
                  )}
                  {stage === 4 && (
                    <StagePosEnc
                      result={result}
                      onOptions={changeOptions}
                      maxSeq={maxSeq}
                      onMaxSeq={setMaxSeq}
                    />
                  )}
                  {stage === 5 && (
                    <StageAttention
                      result={result}
                      selection={selection}
                      onSelection={(s) => {
                        setSelection(s);
                        setPosition(s.query);
                      }}
                      onOptions={changeOptions}
                      hover={hover}
                      onHover={setHover}
                    />
                  )}
                  {stage === 6 && (
                    <StageLayerNorm
                      result={result}
                      onOptions={changeOptions}
                      position={currentPosition}
                    />
                  )}
                  {stage === 7 && (
                    <StageFFN
                      result={result}
                      onOptions={changeOptions}
                      position={currentPosition}
                    />
                  )}
                  {stage === 8 && (
                    <StageResidual
                      result={result}
                      position={currentPosition}
                      onPosition={setPosition}
                    />
                  )}
                  {stage === 9 && (
                    <>
                      <StageOutput result={result} sampling={sampling} />
                      <Comparison
                        request={{ ...request, ids: undefined }}
                        onError={setError}
                      />
                    </>
                  )}
                  {manual && (
                    <ManualCompute
                      result={result}
                      selection={selection}
                      onPatch={setPatch}
                    />
                  )}
                  <details>
                    <summary>查看任意张量 / 单元格干预</summary>
                    <TensorInspector
                      result={result}
                      onPatch={(p) => {
                        setPatch(p);
                        setReplay(-1);
                      }}
                    />
                  </details>
                </>
              )}
            </>
          ) : (
            <TrainingPanel
              options={options}
              onComplete={trainingComplete}
              onError={setError}
            />
          )}
          <footer>
            Token by Token · React 18 / TypeScript / 手写 Transformer ·
            纯浏览器实验
            <br />
            数值按固定小数显示；底层张量使用
            Float32Array。每次显示都来自实际计算。
          </footer>
        </main>
        <aside data-sidebar="right">
          <div className="sec-h">下一个 token</div>
          {result && <ProbPanel sample={result.sample} vocab={result.vocab} />}
          <div className="sec-h">采样控制台</div>
          <ParamConsole
            value={sampling}
            onChange={(s) => {
              setSampling(s);
              setReplay(-1);
              setPlaying(false);
            }}
          />
          <div className="note">
            这里的 T 作用于输出 logits；注意力页的温度作用于 Q·K
            分数，两者独立。
          </div>
          <div className="sec-h">动手验证</div>
          <button
            data-full
            className={`btn ${manual ? "primary" : ""}`}
            onClick={() => {
              setManual((v) => !v);
              setView("stages");
              setStage(5);
            }}
          >
            {manual ? "关闭逐步手算" : "进入逐步手算"}
          </button>
          <div className="sec-h">一个数的蝴蝶效应</div>
          {result && (
            <Perturbation
              result={result}
              patch={patch}
              onPatch={(p) => {
                setPatch(p);
                setReplay(-1);
                setPlaying(false);
              }}
            />
          )}
          <div className="sec-h">实验工具</div>
          <button className="btn" data-full onClick={exportModel}>
            导出当前权重 JSON
          </button>
          <div className="note warn">
            被截断、被遮罩或未启用的路径可能不会传播扰动。零变化也是有效的实验结果。
          </div>
          <small>
            正数用蓝底与 + 号，负数用红底与 − 号；选中项加粗并使用琥珀色。
          </small>
        </aside>
      </div>
    </>
  );
}
