import { useState } from "react";
import type { RunResult } from "../state/store";
import type { Options } from "../core/model/config";
import { fmt } from "../core/format";
import { Card, Control } from "./Shared";
import { MatrixView } from "./MatrixView";
export function StagePosEnc({
  result,
  onOptions,
  maxSeq,
  onMaxSeq,
}: {
  result: RunResult;
  onOptions: (o: Options) => void;
  maxSeq: number;
  onMaxSeq: (n: number) => void;
}) {
  const f = result.trace,
    o = f.options,
    d = f.c.d,
    hd = d / f.c.heads,
    [pair, setPair] = useState(0),
    p = Math.min(pair, Math.max(0, Math.floor(hd / 2) - 1));
  return (
    <>
      <Card
        idx={4}
        title="位置不是一个标签，而是一次变换"
        tag={o.position === "rope" ? "旋转位置编码" : "位置编码对照"}
      >
        <div data-controls>
          <Control label="位置编码">
            <select
              value={o.position}
              onChange={(e) =>
                onOptions({
                  ...o,
                  position: e.target.value as Options["position"],
                })
              }
            >
              <option value="rope">旋转编码（RoPE）</option>
              <option value="sinusoidal">正弦 / 余弦</option>
              <option value="learned">可学习位置向量</option>
              <option value="none">无位置编码</option>
            </select>
          </Control>
          <Control label="上下文最大长度">
            <input
              type="number"
              min={1}
              max={f.c.maxSeq}
              value={maxSeq}
              onChange={(e) =>
                onMaxSeq(
                  Math.max(1, Math.min(f.c.maxSeq, Number(e.target.value))),
                )
              }
            />
          </Control>
        </div>
        <div className="note">
          RoPE 在每个注意力头里旋转 <b>Q 和 K</b>
          ，不直接加到词嵌入上。本模型每头 {fmt(hd, "integer")} 维，因此角度为 θ
          = pos / 10000^(2i / {fmt(hd, "integer")})。
        </div>
        {o.position === "none" && (
          <div className="note warn">
            移除位置编码会丢失显式位置信息，但因果遮罩仍携带方向和前缀长度线索。只有同时取消遮罩等位置相关因素，自注意力才具有置换等变性，不能笼统说输出必然完全无序。
          </div>
        )}
        {hd < 2 && (
          <div className="note warn">
            当前检查点每头只有一维，没有可旋转的维度对。RoPE
            对照在此退化为恒等变换；原始模型应使用可学习位置编码。
          </div>
        )}
        <Control label="观察哪一对维度">
          <input
            type="number"
            min={0}
            max={Math.max(0, Math.floor(hd / 2) - 1)}
            disabled={hd < 2}
            value={p}
            onChange={(e) =>
              setPair(Math.max(0, Math.floor(Number(e.target.value))))
            }
          />
        </Control>
        {hd >= 2 && (
          <div data-row>
            {[0, 1, 2, 3].map((pos) => {
              const theta = pos / 10000 ** ((2 * p) / hd);
              return (
                <svg
                  data-vector
                  key={pos}
                  viewBox="0 0 140 132"
                  width="140"
                  height="132"
                  role="img"
                  aria-label={`位置 ${pos} 的旋转角度`}
                >
                  <circle cx="70" cy="60" r="40" />
                  <line x1="22" y1="60" x2="118" y2="60" />
                  <line x1="70" y1="12" x2="70" y2="108" />
                  <line
                    data-arrow
                    x1="70"
                    y1="60"
                    x2={70 + 40 * Math.cos(theta)}
                    y2={60 - 40 * Math.sin(theta)}
                  />
                  <text x="25" y="122">
                    位置 {pos} · θ {fmt(theta)}
                  </text>
                </svg>
              );
            })}
          </div>
        )}
        <div className="fml">
          a′ = a × cos θ − b × sin θ<br />
          b′ = a × sin θ + b × cos θ
          <div className="sum">
            qᵢ′ · kⱼ′ = qᵢᵀ R(θⱼ − θᵢ) kⱼ
            <br />
            <b>旋转的相对角度，让位置差进入相关分数。</b>
          </div>
        </div>
      </Card>
      <Card idx="↳" title="沿着具体的数字追踪位置">
        <div className="note">
          下表显示第一层、第一头的选定维度对；注意力页还可展开旋转后的 Q·K
          每项乘积。
        </div>
        {o.position === "rope" && hd >= 2 ? (
          <div data-scroll>
            <table data-table>
              <thead>
                <tr>
                  <th>位置</th>
                  <th>q[2i]</th>
                  <th>q[2i+1]</th>
                  <th>θ</th>
                  <th>cos θ</th>
                  <th>sin θ</th>
                  <th>q′[2i]</th>
                  <th>q′[2i+1]</th>
                </tr>
              </thead>
              <tbody>
                {f.ids.map((_, i) => {
                  const a = f.layers[0].att,
                    angle = i / 10000 ** ((2 * p) / hd);
                  return (
                    <tr key={i}>
                      {[
                        i,
                        a.qRaw[i * d + 2 * p],
                        a.qRaw[i * d + 2 * p + 1],
                        angle,
                        Math.cos(angle),
                        Math.sin(angle),
                        a.q[i * d + 2 * p],
                        a.q[i * d + 2 * p + 1],
                      ].map((v, j) => (
                        <td key={j} data-number>
                          {fmt(v, j === 0 ? "integer" : "matrix")}
                        </td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <>
            <MatrixView
              data={f.positional}
              rows={f.ids.length}
              cols={d}
              label="位置向量 P"
            />
            <div className="fml">
              初始残差流 = E[token_id] + P[position]
              <div className="sum">
                无位置模式中 P 全为 0；正弦模式使用 sin /
                cos；学习模式按位置查参数表。
              </div>
            </div>
            <MatrixView
              data={f.initial}
              rows={f.ids.length}
              cols={d}
              label="E + P"
            />
          </>
        )}
      </Card>
    </>
  );
}
