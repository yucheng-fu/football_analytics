import { DEFAULT_ARROW_PROPERTIES } from "../config/constants";
import type { ArrowProperties } from "../types";
import { computeDistance } from "../utils/dist";

interface PassArrowParams {
  id: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  properties?: Partial<ArrowProperties>;
}

export class PassArrow {
  id: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  height: string;
  bodyPart: string;
  underPressure: boolean;
  distance: number;
  duration: number;
  prediction: number | null;
  probability: number | null;
  predictedAt: string | null;

  constructor({ id, x1, y1, x2, y2, properties = {} }: PassArrowParams) {
    this.id = id;
    this.x1 = x1;
    this.y1 = y1;
    this.x2 = x2;
    this.y2 = y2;
    this.height = properties.height ?? DEFAULT_ARROW_PROPERTIES.height;
    this.bodyPart = properties.bodyPart ?? DEFAULT_ARROW_PROPERTIES.bodyPart;
    this.underPressure = properties.underPressure ?? DEFAULT_ARROW_PROPERTIES.underPressure;
    this.distance = computeDistance(x1, y1, x2, y2);
    this.duration = properties.duration ?? DEFAULT_ARROW_PROPERTIES.duration;
    this.prediction = null;
    this.probability = null;
    this.predictedAt = null;
  }

  getSuccessProbability() {
    if (this.prediction === null || this.probability === null) {
      return null;
    }

    return this.prediction === 1 ? this.probability : 1 - this.probability;
  }
}
