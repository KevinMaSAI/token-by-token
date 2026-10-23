/** Contiguous row-major tensors. Every numeric payload is a real Float32Array. */
export class Tensor {
  readonly data: Float32Array;
  readonly shape: number[];
  constructor(shape: number[], data?: ArrayLike<number>) {
    if (!shape.length || shape.some((d) => !Number.isInteger(d) || d < 1))
      throw new Error("张量维度必须为正整数");
    this.shape = [...shape];
    const size = shape.reduce((a, b) => a * b, 1);
    if (data && data.length !== size) throw new Error("张量数据与形状不匹配");
    this.data = data ? Float32Array.from(data) : new Float32Array(size);
  }
  index(...indices: number[]) {
    if (indices.length !== this.shape.length) throw new Error("索引维度不匹配");
    return indices.reduce((a, n, i) => {
      if (!Number.isInteger(n) || n < 0 || n >= this.shape[i])
        throw new Error("索引越界");
      return a * this.shape[i] + n;
    }, 0);
  }
  get(...indices: number[]) {
    return this.data[this.index(...indices)];
  }
  row(i: number) {
    if (this.shape.length !== 2 || i < 0 || i >= this.shape[0])
      throw new Error("行索引越界");
    return this.data.slice(i * this.shape[1], (i + 1) * this.shape[1]);
  }
  reshape(shape: number[]) {
    return new Tensor(shape, this.data);
  }
  broadcastRows(rows: number) {
    if (this.shape.length !== 1) throw new Error("广播需要向量");
    const t = new Tensor([rows, this.data.length]);
    for (let r = 0; r < rows; r++) t.data.set(this.data, r * this.data.length);
    return t;
  }
}
