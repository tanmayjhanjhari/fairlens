# ByUs

> **Detect. Explain. Mitigate.** AI-powered bias detection for datasets and ML models using Google Gemini 2.0 Flash.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-green.svg)](https://python.org)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com)

---

## What is ByUs?

ByUs is an open-source web platform that helps you detect, understand, and fix bias in your datasets and machine learning models. Upload your dataset (CSV, Excel, JSON, ZIP, and more), select your target variable and sensitive attributes, and ByUs will:

- **Compute Industry-Standard Metrics**: Instantly calculate Statistical Parity Difference (SPD), Disparate Impact (DI), Equal Opportunity Difference (EOD), and Average Odds Difference (AOD).
- **Deep Contextual Explanations**: Beyond numbers, ByUs provides plain-English narratives explaining *why* bias exists, including proxy feature detection and data imbalance analysis.
- **AI-Powered Domain Detection**: Automatically identifies the dataset's scenario (Hiring, Lending, Healthcare, etc.) to tailor the analysis.
- **Bias Mitigation Suite**: Run Reweighing and Threshold Adjustment strategies side-by-side and compare results with interactive trade-off charts.
- **Gemini Bias Copilot**: A real-time AI assistant to answer questions about fairness metrics and provide remediation guidance.
- **AI Resilience**: Robust rule-based fallbacks ensure the platform remains fully functional even during AI service quotas or downtime.
- **Professional PDF Reports**: Export high-fidelity audit reports with automated findings and action plans.

---

## Prerequisites

- **Python 3.11+**
- **Node.js 18+** and **npm 9+**
- **Google Gemini API Key** (get one at [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey))

---

## Backend Setup

### 1. Navigate to the backend directory

```bash
cd byus/backend
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

Open `.env` and set:

```
GEMINI_API_KEY=your_google_gemini_api_key_here
ALLOWED_ORIGINS=http://localhost:5173
SESSION_TTL_SECONDS=3600
MAX_UPLOAD_SIZE_MB=50
```

> **Never commit your `.env` file.** It is included in `.gitignore`.

### 5. Start the backend server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.
Interactive API docs: `http://localhost:8000/docs`

---

## Frontend Setup

### 1. Navigate to the frontend directory

```bash
cd byus/frontend
```

### 2. Install dependencies

```bash
npm install
```

### 3. Configure environment variables

Create a `.env.local` file in the `frontend/` directory:

```
VITE_API_BASE_URL=http://localhost:8000
```

> This file is also excluded from git via `.gitignore`.

### 4. Start the development server

```bash
npm run dev
```

The app will be available at `http://localhost:5173`.

---

## Environment Variables Reference

### Backend (`backend/.env`)

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Your Google Gemini API key |
| `ALLOWED_ORIGINS` | Yes | Comma-separated list of allowed CORS origins |
| `SESSION_TTL_SECONDS` | No | Session data TTL in seconds (default: 3600) |
| `MAX_UPLOAD_SIZE_MB` | No | Max CSV/model upload size in MB (default: 50) |

### Frontend (`frontend/.env.local`)

| Variable | Required | Description |
|---|---|---|
| `VITE_API_BASE_URL` | Yes | Backend API base URL |

---

## Running Both Simultaneously

Open two terminal windows:

**Terminal 1 (Backend):**
```bash
cd byus/backend
venv\Scripts\activate   # or source venv/bin/activate on macOS/Linux
uvicorn main:app --reload
```

**Terminal 2 (Frontend):**
```bash
cd byus/frontend
npm run dev
```

---

## Project Structure

```
byus/
+-- frontend/          # React 18 + Vite frontend
+-- backend/           # FastAPI backend
+-- deploy-gcp.ps1      # Automated GCP Deployment (PowerShell)
+-- deploy-gcp.sh       # Automated GCP Deployment (Bash)
+-- REQUIREMENTS.md    # Full requirements documentation
+-- DEMO.md            # 3-minute demo script for judges
+-- README.md          # This file
+-- .gitignore
```

See [REQUIREMENTS.md](./REQUIREMENTS.md) for the full project folder structure and feature documentation.

---

## API Documentation

Once the backend is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

Key endpoints:

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/upload` | Upload dataset (CSV, Excel, JSON, ZIP, Parquet, etc.) |
| POST | `/api/upload-model` | Upload ML model (.pkl/.joblib) |
| POST | `/api/validate` | Validate dataset compatibility |
| POST | `/api/detect-scenario` | AI scenario detection |
| POST | `/api/analyze` | Run fairness analysis |
| POST | `/api/explain` | Run bias explanation engine |
| POST | `/api/mitigate` | Run mitigation strategies |
| POST | `/api/gemini-chat` | Chat with Bias Copilot |
| GET | `/api/report/{session_id}` | Download PDF report |

---

## Deployment

### Google Cloud Run (Recommended)

ByUs is optimized for **Google Cloud**. You can deploy the entire stack using the provided automation scripts:

1. **Install Google Cloud SDK** and run `gcloud init`.
2. Run the deployment script from the root directory:
   - **Windows**: `./deploy-gcp.ps1`
   - **Linux/macOS**: `./deploy-gcp.sh`
3. The script will build your Docker containers, push them to Google Artifact Registry, and deploy them to Cloud Run.

### Manual Deployment (Vercel/Railway)

**Frontend (Vercel)**:
1. Connect your GitHub repository to Vercel.
2. Set root directory to `frontend/`.
3. Add `VITE_API_BASE_URL` environment variable.

**Backend (Railway)**:
1. Connect your GitHub repository to Railway.
2. Set root directory to `backend/`.
3. Add all variables from `backend/.env.example`.

---

## Sample Data

The `backend/sample_data/` directory contains:
- `credit_bias.csv` — Lending dataset with gender and age bias (for demo)
- `hiring_bias.csv` — Hiring dataset with race and gender attributes

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature dev`
3. Commit using Conventional Commits: `git commit -m "feat: add your feature"`
4. Push and open a Pull Request to `dev`

See [REQUIREMENTS.md](./REQUIREMENTS.md#6-git-strategy) for the full Git strategy.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Google Gemini](https://deepmind.google/technologies/gemini/) for AI-powered explanations
- [Fairlearn](https://fairlearn.org/) for fairness metrics and mitigation algorithms
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [Recharts](https://recharts.org/) for data visualization
