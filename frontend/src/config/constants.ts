import type { ArrowProperties } from "../types";

export const APP_CONFIG = {
  scale: 5,
  pitchWidth: 120,
  pitchHeight: 80,
  pageSize: 4,
} as const;

export const DEFAULT_ARROW_PROPERTIES: ArrowProperties = {
  height: "Ground pass",
  bodyPart: "Right foot",
  underPressure: false,
  duration: 1.0,
};
