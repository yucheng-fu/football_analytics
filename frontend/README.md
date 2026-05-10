# Frontend

## Run
1. Install dependencies:
`npm install`
2. Start dev server:
`npm run dev`
3. Build for production:
`npm run build`

## Prediction API Configuration
Create `frontend/.env.local` and set:
`VITE_PASS_PREDICTION_URL=https://yuch0001-football-api.hf.space/predict`
`VITE_PASS_API_KEY=<YOUR_API_KEY>`

The frontend sends the request directly to the hosted API.
