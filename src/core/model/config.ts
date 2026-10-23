import type { Activation, NormKind } from "../ops";
export interface Config {
  vocabSize: number;
  layers: number;
  heads: number;
  d: number;
  ff: number;
  maxSeq: number;
  name: string;
  bosId?: number;
  eosId?: number;
}
export const MINI: Config = {
  vocabSize: 64,
  layers: 2,
  heads: 2,
  d: 32,
  ff: 64,
  maxSeq: 32,
  name: "微型教学模型",
};
export type PositionKind = "rope" | "sinusoidal" | "learned" | "none";
export interface Options {
  position: PositionKind;
  causal: boolean;
  scaled: boolean;
  attentionTemperature: number;
  dropout: number;
  norm: NormKind;
  normOrder: "pre" | "post";
  eps: number;
  activation: Activation;
  ffWidth: number;
}
export const DEFAULT_OPTIONS: Options = {
  position: "rope",
  causal: true,
  scaled: true,
  attentionTemperature: 1,
  dropout: 0,
  norm: "layer",
  normOrder: "pre",
  eps: 1e-5,
  activation: "gelu",
  ffWidth: 64,
};
export interface Patch {
  key: string;
  index: number;
  value: number;
  mode: "add" | "set";
}
export interface TraceTensor {
  key: string;
  label: string;
  shape: [number, number];
  data: Float32Array;
  kind: "weight" | "activation";
}
