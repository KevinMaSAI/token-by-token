import { useState } from "react";
import type { Tokenization } from "../core/tokenizer/bpe";
import { fmt, tokenLabel } from "../core/format";
import { Card } from "./Shared";
export function StageBytes({ data }: { data: Tokenization }) {
  const [hover, setHover] = useState(-1);
  return (
    <>
      <Card idx={0} title="计算机先读到的是字节" tag="UTF-8">
        <div className="note">
          一个常用汉字通常占 <b>3 个字节</b>，英文字母占 1 个；表情可能占 4
          个。把鼠标放在文字上，看它对应哪些字节。
        </div>
        <div data-token-list>
          {data.bytes.map((c) => (
            <button
              className={`tok ${hover === c.index ? "pick" : ""}`}
              key={c.index}
              onMouseEnter={() => setHover(c.index)}
              onFocus={() => setHover(c.index)}
              onMouseLeave={() => setHover(-1)}
            >
              {tokenLabel(c.char)}
              <span className="id">
                {fmt(c.index, "integer")} · {fmt(c.bytes.length, "integer")}{" "}
                字节
              </span>
            </button>
          ))}
        </div>
        <div className="sec-h">十六进制字节序列</div>
        <div data-token-list>
          {data.bytes.flatMap((c) =>
            c.bytes.map((b, i) => (
              <button
                className={`tok ${hover === c.index ? "pick" : ""}`}
                key={c.start + i}
                onMouseEnter={() => setHover(c.index)}
                onFocus={() => setHover(c.index)}
              >
                {b.toString(16).toUpperCase().padStart(2, "0")}
                <span className="id">
                  {fmt(c.start + i, "integer")} · {c.mapped[i]}
                </span>
              </button>
            )),
          )}
        </div>
        <div className="fml">
          <span className="lbl">原始输入</span>
          <br />
          {data.text || "（空输入：模型使用起始标记）"}
          <div className="sum">
            {fmt(data.charCount, "integer")} 个 Unicode 码点 →{" "}
            <b>
              {fmt(
                data.bytes.reduce((s, c) => s + c.bytes.length, 0),
                "integer",
              )}{" "}
              个字节
            </b>
          </div>
        </div>
      </Card>
      <Card idx="↳" title="为什么要先映射成可打印字符？">
        <p data-small data-muted>
          空格、换行等字节无法直接显示为普通字母。GPT-2 式映射把 256
          个字节一一映射到可打印字符，例如空格常显示为
          Ġ。它只改变显示符号，不改变信息，也不是加密。
        </p>
        <div className="note warn">
          字符、字节和 token 是三种不同单位。后面的 BPE 在字节符号上合并，64
          词教学词表再把最终符号映射为模型 ID。
        </div>
      </Card>
    </>
  );
}
