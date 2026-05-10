export interface Point {
  x: number;
  y: number;
}

export interface ArrowProperties {
  height: string;
  bodyPart: string;
  underPressure: boolean;
  duration: number;
}

export interface PassPredictionRequestPayload {
  start_x: number;
  start_y: number;
  end_x: number;
  end_y: number;
  length: number;
  height: string;
  angle: number;
  duration: number;
  body_part: string;
  under_pressure: number | null;
}

export interface PassPredictionResponsePayload {
  prediction: number;
  probability: number;
  timestamp: string;
}
