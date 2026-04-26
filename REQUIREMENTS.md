# ByUs — Requirements Document

> **Version:** 1.0.0 | **Date:** 2026-04-26 | **Status:** Active Development

---

## 1. Project Overview

**ByUs** is an open-source web platform designed to detect, explain, and mitigate bias in datasets and machine learning models. It leverages **Google Gemini 2.0 Flash** as its AI backbone to provide plain-English explanations, scenario auto-detection, and a conversational Bias Copilot — making algorithmic fairness accessible to data scientists, auditors, and non-technical stakeholders alike.

ByUs bridges the gap between raw fairness metrics and actionable insights by combining established fairness libraries (Fairlearn, scikit-learn) with large language model reasoning, delivering results through an intuitive visual dashboard and exportable PDF audit reports.

**Target Users:**
- Data Scientists auditing model fairness pre-deployment
- ML Engineers integrating fairness checks into CI pipelines
- Compliance Officers requiring documented bias audits
- Researchers studying algorithmic discrimination

---

## 2. Core Features

### Feature 1 — Multi-format Dataset Upload with Instant Column Detection and Type Analysis
- Drag-and-drop or click-to-upload interface supporting `.csv`, `.xlsx`, `.xls`, `.json`, `.zip`, `.parquet`, `.tsv`, and `.txt` files
- Automatic detection of column names, data types (numeric, categorical, boolean, datetime)
- Instant preview table showing first 10 rows
- Column cardinality stats (unique values, null %, distribution summary)
- Dataset shape displayed (rows x columns)

### Feature 2 — ML Model Upload (.pkl / .joblib) — Prediction Bias Analysis
- Accepts serialized scikit-learn-compatible models in `.pkl` or `.joblib` format
- Automatically runs model predictions against the uploaded dataset
- Computes prediction-level bias (group-level prediction rate disparities)
- Surfaces model-specific bias separate from dataset-level bias
- Gracefully handles unsupported model types with user-friendly error messages

### Feature 3 — Data Validation Layer
- Validates dataset compatibility before any analysis begins
- Checks for: missing target column, insufficient rows (<50), all-null columns, unsupported target types
- Warns user with descriptive, actionable error messages
- Automatically falls back to **Fairlearn** metrics for datasets with:
  - Continuous (regression) target variables
  - Multi-class (3+ class) classification targets
- Displays validation status badge: `Full Analysis`, `Partial Analysis (Fairlearn Fallback)`, or `Unsupported`

### Feature 4 — Target Variable + Sensitive Attribute Selection
- Dropdown menus for selecting the target column and one or more sensitive attribute columns
- Auto-highlights likely demographic columns using keyword matching (gender, race, age, nationality, religion, disability)
- Color-coded column type badges in the selector UI
- Multi-select support for sensitive attributes (analyze multiple simultaneously)
- Real-time preview of selected column distributions before running analysis

### Feature 5 — Gemini 2.0 Flash Scenario Auto-Detection
- Sends column names, dataset metadata, and sample rows to Gemini 2.0 Flash
- Classifies dataset into one of: **Hiring**, **Lending**, **Healthcare**, **Education**, or **Other**
- Returns scenario label with a confidence percentage (e.g., "Lending — 94% confidence")
- Scenario context influences subsequent Gemini explanations and recommendations
- User can manually override the detected scenario

### Feature 6 — Fairness Metrics: SPD, DI, EOD, AOD
- Computes the following metrics per sensitive attribute group:
  - **Statistical Parity Difference (SPD)**: P(Yhat=1|A=privileged) minus P(Yhat=1|A=unprivileged)
  - **Disparate Impact (DI)**: P(Yhat=1|A=unprivileged) / P(Yhat=1|A=privileged)
  - **Equal Opportunity Difference (EOD)**: difference in true positive rates between groups
  - **Average Odds Difference (AOD)**: average of TPR and FPR differences
- Each metric tagged with severity level: **Low** (green), **Medium** (yellow), **High** (red)
- Severity thresholds follow legal and academic standards (e.g., DI < 0.8 = high severity)
- Metrics computed independently for each sensitive attribute selected

### Feature 7 — Fairlearn Fallback Metrics for Continuous Targets and Multi-Class Outcomes
- Activates automatically when standard binary classification metrics cannot be applied
- For **regression targets**: uses Fairlearn's MetricFrame with mean absolute error per group
- For **multi-class targets**: computes per-class, per-group accuracy and selection rate disparities
- Results displayed in a dedicated "Extended Metrics" section of the dashboard
- Clear labeling indicates which metrics are from Fairlearn vs. the primary engine

### Feature 8 — Statistical Significance Check
- Runs bootstrapped confidence intervals (1,000 resamples) on all fairness metrics
- Displays 95% CI bands on metric values in the dashboard
- Issues a prominent warning if dataset has fewer than 200 rows per sensitive group
- Warning message: "Small sample detected — results may not be statistically reliable"
- Confidence interval overlapping zero for SPD flagged as "Not Statistically Significant"

### Feature 9 — Bias Audit Score (0-100, Graded A/B/C/F)
- Composite score computed from weighted combination of all fairness metrics
- Weighting: SPD (30%), DI (30%), EOD (20%), AOD (20%)
- Grade scale:
  - **A**: 80-100 (Low bias, minimal disparity)
  - **B**: 60-79 (Moderate bias, monitor)
  - **C**: 40-59 (Significant bias, action recommended)
  - **F**: 0-39 (Severe bias, immediate remediation required)
- Animated circular gauge on the dashboard with grade letter and color-coded ring
- Score breakdown tooltip showing per-metric contributions

### Feature 10 — Visual Dashboard
- **Bar Charts**: side-by-side group comparison bars for each fairness metric
- **Radar Chart**: multi-axis view of all four metrics simultaneously (overall fairness profile)
- **Group Heatmap**: color-coded matrix of metric values per sensitive attribute x metric
- **Distribution Charts**: overlaid histograms of target variable distribution per sensitive group
- All charts built with **Recharts** for interactivity and responsiveness
- Charts exportable as PNG via right-click or dedicated export button

### Feature 11 — Bias Explanation Engine
- **Correlation Analysis**: Pearson/Cramer's V between sensitive attributes and target variable
- **Proxy Feature Detection**: identifies non-sensitive columns correlated with sensitive attributes
- **Data Imbalance Detection**: flags groups with significantly fewer samples than others
- **Historical Skew Analysis**: identifies label distribution skew per group
- **Gemini Plain-English Explanation**: synthesizes all findings into a 3-5 sentence narrative
- Explanation panel with expandable sections per explanation type

### Feature 12 — Mitigation: Reweighing + Threshold Adjustment
- Runs both mitigation techniques **simultaneously** (parallel execution):
  - **Reweighing**: assigns sample weights to balance group representation before retraining
  - **Threshold Adjustment**: adjusts decision thresholds per group to equalize outcomes
- Side-by-side comparison panel: original vs. reweighing vs. threshold adjustment results
- Each technique shows updated fairness metrics post-mitigation
- Best technique highlighted with a "Recommended" badge based on overall metric improvement

### Feature 13 — Mitigation Effects: Before/After Performance Metrics + Trade-off Chart
- Computes and displays for each mitigation technique:
  - **Accuracy**, **Precision**, **Recall**, **F1 Score** — before and after
- Color-coded delta indicators (green = improved or maintained, red = degraded)
- **Trade-off Chart**: scatter plot of Bias Score (x-axis) vs. Accuracy (y-axis)
- Three data points: Original, Reweighing, Threshold Adjustment
- Enables informed decision-making on fairness vs. performance tradeoff

### Feature 14 — PDF Report Generation
- Generates a multi-page professional PDF audit report containing:
  - **Page 1 — Dataset Summary**: shape, column types, target/sensitive selections, scenario detected
  - **Page 2 — Bias Findings**: all fairness metrics with values, severity labels, Audit Score with grade
  - **Page 3 — Before/After Table**: mitigation comparison table for all metrics
  - **Page 4 — Recommendations**: Gemini-generated remediation guidance, proxy features list, next steps
- Generated using **ReportLab** (Python backend)
- Download triggered via `GET /api/report/{session_id}` endpoint
- Report named: `byus_audit_{session_id}_{timestamp}.pdf`

### Feature 15 — Gemini Bias Copilot (Floating Chat Assistant)
- Persistent floating chat widget accessible from any page
- Context-aware: Copilot has access to the full session analysis results
- Remembers conversation history within the session (multi-turn dialogue)
- Suggested starter prompts: "Why is gender biased?", "Which mitigation should I use?", "Explain DI to me"
- Powered by **Gemini 2.0 Flash** with a custom system prompt injecting session analysis context
- Supports markdown rendering in responses (bold, lists, code blocks)

### Feature 16 — Guided Onboarding with Tooltips
- First-time user flow with a 5-step onboarding overlay
- Tooltips on all technical terms (SPD, DI, EOD, AOD, Reweighing, Threshold Adjustment)
- Tooltip content is concise (1-2 sentences) with a "Learn more" link to relevant documentation
- Onboarding state persisted in `localStorage` (skipped on subsequent visits)
- "Reset Onboarding" option in the app settings/help menu

---

## 3. Tech Stack

### Frontend
| Technology | Version | Purpose |
|---|---|---|
| React | 18.x | UI framework |
| Vite | 5.x | Build tool and dev server |
| TailwindCSS | 3.x | Utility-first styling |
| Framer Motion | 11.x | Animations and transitions |
| Recharts | 2.x | Data visualization charts |
| Zustand | 4.x | Global state management |
| React Router | v6 | Client-side routing |
| Axios | 1.x | HTTP client for API calls |
| react-dropzone | 14.x | Drag-and-drop file uploads |
| react-hot-toast | 2.x | Toast notifications |
| lucide-react | 0.x | Icon library |

### Backend
| Technology | Version | Purpose |
|---|---|---|
| Python | 3.11 | Runtime |
| FastAPI | 0.111.x | API framework |
| pandas | 2.x | Data manipulation |
| scikit-learn | 1.4.x | ML metrics and models |
| fairlearn | 0.10.x | Fairness metrics and mitigation |
| shap | 0.45.x | Feature importance / explainability |
| reportlab | 4.x | PDF generation |
| python-multipart | 0.0.9 | File upload handling |
| joblib | 1.4.x | Model serialization |
| scipy | 1.13.x | Statistical computations |
| google-generativeai | 0.7.x | Gemini 2.0 Flash API |
| python-dotenv | 1.x | Environment variable management |
| uvicorn | 0.29.x | ASGI server |

### AI
| Component | Details |
|---|---|
| Model | Google Gemini 2.0 Flash (gemini-2.0-flash-exp) |
| Use Cases | Scenario detection, bias explanation, PDF recommendations, Copilot chat |
| API | Google Generative AI Python SDK |

### Deployment
| Layer | Platform |
|---|---|
| Frontend | Vercel (auto-deploy from `main` branch) |
| Backend | Railway (Dockerfile or nixpacks, auto-deploy from `main` branch) |
| Environment Variables | Set in Vercel/Railway project dashboards (never committed to repo) |

---

## 4. API Endpoints

All endpoints are prefixed with `/api`. Base URL in development: `http://localhost:8000`.

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/upload` | Upload a CSV dataset |
| POST | `/api/upload-model` | Upload a serialized ML model (.pkl/.joblib) |
| POST | `/api/validate` | Validate dataset compatibility for analysis |
| POST | `/api/detect-scenario` | Auto-detect dataset domain using Gemini |
| POST | `/api/analyze` | Run full fairness analysis (SPD, DI, EOD, AOD) |
| POST | `/api/explain` | Run bias explanation engine |
| POST | `/api/gemini-explain` | Get Gemini plain-English explanation of bias findings |
| POST | `/api/mitigate` | Run reweighing and threshold adjustment simultaneously |
| POST | `/api/gemini-chat` | Interact with the Gemini Bias Copilot |
| GET | `/api/report/{session_id}` | Download the generated PDF audit report |

### Request/Response Details

**`POST /api/upload`**
- Request: `multipart/form-data` with field `file` (.csv)
- Response: `{ session_id, columns, dtypes, shape, preview_rows, null_summary }`

**`POST /api/upload-model`**
- Request: `multipart/form-data` with fields `file` (.pkl/.joblib) and `session_id`
- Response: `{ model_type, feature_names, classes, status }`

**`POST /api/validate`**
- Request: `{ session_id, target_column, sensitive_columns }`
- Response: `{ valid, analysis_mode, warnings, validation_details }`

**`POST /api/detect-scenario`**
- Request: `{ session_id, columns, sample_rows }`
- Response: `{ scenario, confidence, reasoning }`

**`POST /api/analyze`**
- Request: `{ session_id, target_column, sensitive_columns, scenario }`
- Response: `{ audit_score, grade, metrics_per_attribute, statistical_significance, group_distributions }`

**`POST /api/explain`**
- Request: `{ session_id, target_column, sensitive_columns }`
- Response: `{ correlation_analysis, proxy_features, imbalance_report, historical_skew, gemini_explanation }`

**`POST /api/gemini-explain`**
- Request: `{ session_id, metrics, scenario, sensitive_columns }`
- Response: `{ explanation, key_factors, recommendations }`

**`POST /api/mitigate`**
- Request: `{ session_id, target_column, sensitive_columns, techniques: ["reweighing", "threshold"] }`
- Response: `{ original_metrics, reweighing_metrics, threshold_metrics, performance_comparison }`

**`POST /api/gemini-chat`**
- Request: `{ session_id, message, conversation_history, analysis_context }`
- Response: `{ reply, sources }`

**`GET /api/report/{session_id}`**
- Response: Binary PDF stream with `Content-Disposition: attachment` header
- File Name: `byus_audit_{session_id}_{timestamp}.pdf`

---

## 5. Project Folder Structure

```
byus/
|
+-- frontend/
|   +-- public/
|   |   +-- favicon.ico
|   +-- src/
|   |   +-- components/
|   |   |   +-- Layout/
|   |   |   |   +-- Navbar.jsx
|   |   |   |   +-- Sidebar.jsx
|   |   |   |   +-- Footer.jsx
|   |   |   +-- Upload/
|   |   |   |   +-- CSVDropzone.jsx
|   |   |   |   +-- ModelUpload.jsx
|   |   |   |   +-- ColumnPreview.jsx
|   |   |   +-- Dashboard/
|   |   |   |   +-- AuditScoreGauge.jsx
|   |   |   |   +-- MetricsBarChart.jsx
|   |   |   |   +-- RadarChart.jsx
|   |   |   |   +-- GroupHeatmap.jsx
|   |   |   |   +-- DistributionChart.jsx
|   |   |   |   +-- MetricCard.jsx
|   |   |   +-- Mitigation/
|   |   |   |   +-- MitigationPanel.jsx
|   |   |   |   +-- BeforeAfterTable.jsx
|   |   |   |   +-- TradeoffChart.jsx
|   |   |   +-- Copilot/
|   |   |   |   +-- CopilotWidget.jsx
|   |   |   |   +-- ChatBubble.jsx
|   |   |   |   +-- SuggestedPrompts.jsx
|   |   |   +-- Onboarding/
|   |   |   |   +-- OnboardingOverlay.jsx
|   |   |   |   +-- Tooltip.jsx
|   |   |   +-- Report/
|   |   |       +-- ReportDownloadButton.jsx
|   |   +-- pages/
|   |   |   +-- LandingPage.jsx
|   |   |   +-- UploadPage.jsx
|   |   |   +-- DashboardPage.jsx
|   |   |   +-- MitigationPage.jsx
|   |   |   +-- ReportPage.jsx
|   |   +-- store/
|   |   |   +-- useAnalysisStore.js
|   |   +-- api/
|   |   |   +-- byusApi.js
|   |   +-- utils/
|   |   |   +-- formatMetrics.js
|   |   |   +-- severityHelpers.js
|   |   +-- hooks/
|   |   |   +-- useSessionId.js
|   |   |   +-- useAnalysis.js
|   |   +-- App.jsx
|   |   +-- main.jsx
|   |   +-- index.css
|   +-- index.html
|   +-- vite.config.js
|   +-- tailwind.config.js
|   +-- package.json
|
+-- backend/
|   +-- routers/
|   |   +-- upload.py
|   |   +-- validate.py
|   |   +-- analyze.py
|   |   +-- explain.py
|   |   +-- mitigate.py
|   |   +-- gemini.py
|   |   +-- report.py
|   +-- services/
|   |   +-- fairness_engine.py
|   |   +-- explanation_engine.py
|   |   +-- mitigation_engine.py
|   |   +-- gemini_service.py
|   |   +-- pdf_generator.py
|   |   +-- session_store.py
|   +-- utils/
|   |   +-- data_utils.py
|   |   +-- metric_utils.py
|   |   +-- validation_utils.py
|   +-- sample_data/
|   |   +-- credit_bias.csv
|   |   +-- hiring_bias.csv
|   +-- main.py
|   +-- requirements.txt
|   +-- .env.example
|
+-- .gitignore
+-- README.md
+-- REQUIREMENTS.md
+-- DEMO.md
```

---

## 6. Git Strategy

### Branch Model

| Branch | Purpose |
|---|---|
| `main` | Production-ready code. Auto-deploys to Vercel and Railway. Protected branch. |
| `dev` | Integration branch. All features merged here first, then to `main` via PR. |
| `feature/*` | Individual feature branches. Branch off `dev`, merge back into `dev`. |
| `fix/*` | Bug fix branches. |
| `chore/*` | Non-code changes (docs, config, dependencies). |

### Branch Naming Convention
```
feature/part-0     -> Initial project setup
feature/part-1     -> CSV upload + column detection
feature/part-2     -> Fairness analysis engine
feature/part-3     -> Gemini integration
feature/part-4     -> Mitigation engine
feature/part-5     -> Visual dashboard
feature/part-6     -> PDF report + Copilot
fix/metric-calc    -> Bug fixes
chore/update-deps  -> Dependency updates
```

### Commit Message Convention (Conventional Commits)
```
chore: initial project setup and requirements
feat(upload): add CSV drag-and-drop with column detection
feat(analysis): implement SPD, DI, EOD, AOD metrics
feat(gemini): add scenario auto-detection with Gemini 2.0 Flash
feat(dashboard): add radar chart and group heatmap
feat(mitigation): implement reweighing and threshold adjustment
feat(copilot): add Gemini Bias Copilot floating widget
feat(report): add PDF generation with ReportLab
fix(metrics): correct EOD calculation for edge cases
docs: update README with environment variable setup
```

### Workflow
1. Create feature branch from `dev`: `git checkout -b feature/part-X dev`
2. Develop and commit locally using Conventional Commits
3. Push branch: `git push origin feature/part-X`
4. Open Pull Request to `dev`
5. After testing on `dev`, open PR to `main` for deployment

### Initial Repository Setup Commands
```bash
git init
git remote add origin https://github.com/tanmayjhanjhari/byus.git
git checkout -b dev
git checkout -b feature/part-0
git add .
git commit -m "chore: initial project setup and requirements"
git push origin feature/part-0
```
