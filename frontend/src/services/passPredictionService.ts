import type { PassPredictionRequestPayload, PassPredictionResponsePayload } from "../types";

const PREDICTION_URL = import.meta.env.VITE_PASS_PREDICTION_URL;

export class PassPredictionService {
  async predictPass(payload: PassPredictionRequestPayload): Promise<PassPredictionResponsePayload> {
    if (!PREDICTION_URL) {
      throw new Error("Missing VITE_PASS_PREDICTION_URL in frontend/.env.local");
    }

    const response = await fetch(PREDICTION_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const message = await response.text();
      throw new Error(message || "Failed to get pass prediction");
    }

    return response.json();
  }
}
