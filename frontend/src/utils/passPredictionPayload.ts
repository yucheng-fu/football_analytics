import type { PassArrow } from "../models/PassArrow";
import type { PassPredictionRequestPayload } from "../types";

const HEIGHT_MAP: Record<string, string> = {
  "Ground pass": "Ground Pass",
  "Low pass": "Low Pass",
  "High pass": "High Pass",
};

const BODY_PART_MAP: Record<string, string> = {
  "Right foot": "Right Foot",
  "Left foot": "Left Foot",
  Head: "Head",
  "Keeper arm": "Other",
  "Drop kick": "Other",
  Other: "Other",
};

export function toPassPredictionPayload(arrow: PassArrow): PassPredictionRequestPayload {
  const angle = Math.atan2(arrow.y2 - arrow.y1, arrow.x2 - arrow.x1);

  return {
    start_x: arrow.x1,
    start_y: arrow.y1,
    end_x: arrow.x2,
    end_y: arrow.y2,
    length: arrow.distance,
    height: HEIGHT_MAP[arrow.height] ?? "Ground Pass",
    angle,
    duration: arrow.duration,
    body_part: BODY_PART_MAP[arrow.bodyPart] ?? "Other",
    under_pressure: arrow.underPressure ? 1 : 0,
  };
}
