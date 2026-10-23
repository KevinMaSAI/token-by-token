/// <reference lib="webworker" />
import {
  Adam,
  trainingData,
  trainStep,
  trainerPicker,
} from "../core/model/train";
import { MINI, type Options } from "../core/model/config";
import type { Weights } from "../core/model/weights";
let stopped = false,
  running = false;
self.onmessage = async (e: MessageEvent) => {
  if (e.data.type === "stop") {
    stopped = true;
    return;
  }
  if (e.data.type !== "start" || running) return;
  running = true;
  stopped = false;
  try {
    const { weights, options, steps, lr, clip, seed } = e.data.payload as {
      weights: Weights;
      options: Options;
      steps: number;
      lr: number;
      clip: number;
      seed: number;
    };
    if (
      !Number.isInteger(steps) ||
      steps < 1 ||
      steps > 5000 ||
      !(lr > 0 && lr <= 0.1) ||
      !(clip > 0)
    )
      throw new Error("训练参数无效");
    const data = trainingData(),
      adam = new Adam(weights),
      pick = trainerPicker(seed),
      start = performance.now();
    let completed = 0;
    for (let step = 1; step <= steps && !stopped; step++) {
      const metrics = trainStep(
        weights,
        MINI,
        { ...options, ffWidth: Math.min(64, options.ffWidth) },
        adam,
        data[pick(data.length)],
        lr,
        clip,
      );
      completed = step;
      self.postMessage({
        type: "point",
        point: {
          step,
          total: steps,
          ...metrics,
          elapsed: performance.now() - start,
          embedding: weights.embedding.slice(),
        },
      });
      if (step % 2 === 0)
        await new Promise((resolve) => setTimeout(resolve, 0));
    }
    self.postMessage({
      type: "done",
      weights,
      steps: completed,
      stopped,
      elapsed: performance.now() - start,
    });
  } catch (error) {
    self.postMessage({
      type: "error",
      error: error instanceof Error ? error.message : String(error),
    });
  } finally {
    running = false;
  }
};
