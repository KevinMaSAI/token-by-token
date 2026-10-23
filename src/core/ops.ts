export type Activation = "relu" | "gelu" | "silu" | "swiglu";
export type NormKind = "layer" | "rms";
export const zeros = (n: number) => new Float32Array(n);

/** C[i,j] = sum_k A[i,k] * B[k,j]; row vectors multiply matrices on the right. */
export function matmul(
  a: Float32Array,
  b: Float32Array,
  m: number,
  k: number,
  n: number,
) {
  if (a.length !== m * k || b.length !== k * n)
    throw new Error("矩阵乘法维度不匹配");
  const c = zeros(m * n);
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let q = 0; q < k; q++) s += a[i * k + q] * b[q * n + j];
      c[i * n + j] = s;
    }
  return c;
}
/** dA = dC B^T, dB = A^T dC; explicit local derivatives, no autodiff. */
export function matmulBackward(
  a: Float32Array,
  b: Float32Array,
  dc: Float32Array,
  m: number,
  k: number,
  n: number,
) {
  const da = zeros(a.length),
    db = zeros(b.length);
  for (let i = 0; i < m; i++)
    for (let q = 0; q < k; q++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += dc[i * n + j] * b[q * n + j];
      da[i * k + q] = s;
    }
  for (let q = 0; q < k; q++)
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let i = 0; i < m; i++) s += a[i * k + q] * dc[i * n + j];
      db[q * n + j] = s;
    }
  return { da, db };
}
export function add(a: Float32Array, b: Float32Array) {
  if (a.length !== b.length) throw new Error("逐元素加法维度不匹配");
  return a.map((x, i) => x + b[i]);
}
export const mul = (a: Float32Array, b: Float32Array) =>
  a.map((x, i) => x * b[i]);
export const scale = (a: Float32Array, n: number) => a.map((x) => x * n);
export const dot = (a: Float32Array, b: Float32Array) =>
  a.reduce((s, x, i) => s + x * b[i], 0);
export const norm = (a: Float32Array) => Math.sqrt(dot(a, a));
export const cosine = (a: Float32Array, b: Float32Array) =>
  dot(a, b) / (norm(a) * norm(b) || 1);
export function biasAdd(x: Float32Array, b: Float32Array) {
  return x.map((v, i) => v + b[i % b.length]);
}
export function biasGrad(d: Float32Array, width: number) {
  const b = zeros(width);
  for (let j = 0; j < width; j++) {
    let s = 0;
    for (let i = j; i < d.length; i += width) s += d[i];
    b[j] = s;
  }
  return b;
}

/** p_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x)); -Infinity denotes a mask. */
export function softmaxDetails(x: Float32Array, temperature = 1) {
  if (!(temperature > 0) || !Number.isFinite(temperature))
    throw new Error("温度必须为有限正数");
  const scaled = x.map((v) => v / temperature);
  let max = -Infinity;
  for (const v of scaled) if (v > max) max = v;
  const shifted = scaled.map((v) =>
    Number.isFinite(max) ? v - max : -Infinity,
  );
  const exp = shifted.map((v) => Math.exp(v));
  const sum = exp.reduce((a, b) => a + b, 0);
  const probs = exp.map((v) => (sum ? v / sum : 0));
  return { scaled, max, shifted, exp, sum, probs };
}
export const softmax = (x: Float32Array) => softmaxDetails(x).probs;
export const softmaxWithTemp = (x: Float32Array, t: number) =>
  softmaxDetails(x, t).probs;
/** dx_i = p_i * (dp_i - sum_j dp_j*p_j). */
export function softmaxBackward(p: Float32Array, dp: Float32Array) {
  const s = dot(p, dp);
  return p.map((v, i) => v * (dp[i] - s));
}

export interface NormCache {
  input: Float32Array;
  normalized: Float32Array;
  output: Float32Array;
  mean: Float32Array;
  variance: Float32Array;
  inv: Float32Array;
  kind: NormKind;
  width: number;
}
/** LN(x) = (x-mean)/sqrt(var+eps)*gamma+beta; RMS(x) uses mean(x*x), no beta. */
export function normalize(
  x: Float32Array,
  gamma: Float32Array,
  beta: Float32Array,
  eps = 1e-5,
  kind: NormKind = "layer",
): NormCache {
  const width = gamma.length,
    rows = x.length / width;
  const mean = zeros(rows),
    variance = zeros(rows),
    inv = zeros(rows),
    normalized = zeros(x.length),
    output = zeros(x.length);
  for (let r = 0; r < rows; r++) {
    let mu = 0;
    if (kind === "layer")
      for (let j = 0; j < width; j++) mu += x[r * width + j] / width;
    let v = 0;
    for (let j = 0; j < width; j++) v += (x[r * width + j] - mu) ** 2 / width;
    const iv = 1 / Math.sqrt(v + eps);
    mean[r] = mu;
    variance[r] = v;
    inv[r] = iv;
    for (let j = 0; j < width; j++) {
      const z = (x[r * width + j] - mu) * iv;
      normalized[r * width + j] = z;
      output[r * width + j] = z * gamma[j] + (kind === "layer" ? beta[j] : 0);
    }
  }
  return { input: x, normalized, output, mean, variance, inv, kind, width };
}
export const layernorm = (x: Float32Array, eps = 1e-5) =>
  normalize(x, new Float32Array(x.length).fill(1), zeros(x.length), eps).output;
export const rmsnorm = (x: Float32Array, eps = 1e-5) =>
  normalize(x, new Float32Array(x.length).fill(1), zeros(x.length), eps, "rms")
    .output;
/** dx = inv*(g - mean(g) - z*mean(g*z)); RMS omits mean(g). */
export function normBackward(
  c: NormCache,
  dy: Float32Array,
  gamma: Float32Array,
) {
  const dx = zeros(dy.length),
    dg = zeros(c.width),
    db = zeros(c.width),
    w = c.width;
  for (let r = 0; r < dy.length / w; r++) {
    let s = 0,
      sz = 0;
    for (let j = 0; j < w; j++) {
      const i = r * w + j,
        g = dy[i] * gamma[j];
      s += g / w;
      sz += (g * c.normalized[i]) / w;
      dg[j] += dy[i] * c.normalized[i];
      if (c.kind === "layer") db[j] += dy[i];
    }
    for (let j = 0; j < w; j++) {
      const i = r * w + j;
      dx[i] =
        c.inv[r] *
        (dy[i] * gamma[j] -
          (c.kind === "layer" ? s : 0) -
          c.normalized[i] * sz);
    }
  }
  return { dx, dg, db };
}
export const relu = (x: number) => Math.max(0, x);
export const silu = (x: number) => x / (1 + Math.exp(-x));
/** GELU tanh approximation used by GPT-2: .5*x*(1+tanh(sqrt(2/pi)*(x+.044715*x^3))). */
export const gelu = (x: number) =>
  0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
export const swiglu = (x: number, gate: number) => silu(x) * gate;
export function activate(x: number, kind: Activation) {
  return kind === "relu" ? relu(x) : kind === "gelu" ? gelu(x) : silu(x);
}
export function activationDerivative(x: number, kind: Activation) {
  if (kind === "relu") return x > 0 ? 1 : 0;
  if (kind === "gelu") {
    const s = Math.sqrt(2 / Math.PI),
      u = Math.tanh(s * (x + 0.044715 * x ** 3));
    return (
      0.5 * (1 + u) + 0.5 * x * (1 - u * u) * s * (1 + 3 * 0.044715 * x * x)
    );
  }
  const s = 1 / (1 + Math.exp(-x));
  return s + x * s * (1 - s);
}
/** RoPE: [a,b] -> [a*cos(theta)-b*sin(theta), a*sin(theta)+b*cos(theta)]. */
export function rope(
  x: Float32Array,
  seq: number,
  heads: number,
  headDim: number,
  inverse = false,
) {
  const out = x.slice(),
    d = heads * headDim;
  for (let p = 0; p < seq; p++)
    for (let h = 0; h < heads; h++)
      for (let j = 0; j + 1 < headDim; j += 2) {
        const i = p * d + h * headDim + j,
          t = ((inverse ? -1 : 1) * p) / 10000 ** (j / headDim),
          c = Math.cos(t),
          s = Math.sin(t);
        out[i] = x[i] * c - x[i + 1] * s;
        out[i + 1] = x[i] * s + x[i + 1] * c;
      }
  return out;
}
export function sinusoidal(seq: number, d: number) {
  const x = zeros(seq * d);
  for (let p = 0; p < seq; p++)
    for (let j = 0; j < d; j++)
      x[p * d + j] = (j % 2 ? Math.cos : Math.sin)(
        p / 10000 ** ((2 * Math.floor(j / 2)) / d),
      );
  return x;
}
