# Frontend
This folder contains the code for the frontend components for the interactive pass classifier.

## Run locally
Install dependencies:
```bash 
npm install
```
Create .env.local file and add the following:
```
VITE_PASS_PREDICTION_URL=https://yuch0001-football-api.hf.space/api/v1/inference/predict
```

Start dev server:
```bash
npm run dev
```

## Deploy to production
Run the Github Actions workflow defined in `.github/workflows/deploy-frontend-pages.yml`