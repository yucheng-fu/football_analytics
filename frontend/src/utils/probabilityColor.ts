import { interpolateViridis } from "d3";

export function probabilityToArrowColor(successProbability: number | null) {
  if (successProbability === null) {
    return "#4b5563";
  }

  const clamped = Math.max(0, Math.min(1, successProbability));
  return interpolateViridis(clamped);
}
