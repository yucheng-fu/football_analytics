/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_PASS_PREDICTION_URL?: string;
  readonly VITE_PASS_API_KEY?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
