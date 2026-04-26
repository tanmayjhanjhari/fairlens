# FairLens — Demo Script

> **3-Minute Demo for Judges**
> Showcases all major platform features end-to-end using the included `credit_bias.csv` sample dataset.

---

## Setup Before Demo

1. Start backend: `cd backend && uvicorn main:app --reload`
2. Start frontend: `cd frontend && npm run dev`
3. Open browser to `http://localhost:5173`
4. Have `backend/sample_data/credit_bias.csv` ready

---

## Demo Flow

---

### Step 1 — Upload Dataset (0:00 – 0:20)

**Action:** Drag and drop `credit_bias.csv` onto the upload dropzone on the Upload page.

**What to point out:**
- File accepted instantly; no page reload
- Column list appears immediately: `age`, `gender`, `income`, `credit_score`, `loan_approved`, `loan_amount`, `employment_status`
- Data types auto-detected: `gender` = categorical, `age` = numeric, `loan_approved` = binary integer
- Preview table shows first 10 rows
- Dataset shape: **1,000 rows x 7 columns**

---

### Step 2 — Validation Layer (0:20 – 0:35)

**Action:** Click "Validate Dataset" — the system runs compatibility checks before analysis.

**What to point out:**
- Green success banner appears:

  > "Binary classification dataset detected. Full analysis supported."

- Validation details expand: binary target confirmed (values 0/1), no null columns, sufficient rows (1,000 >= 50 minimum)
- If the dataset were a regression target or had 3+ classes, the banner would instead read "Partial Analysis — Fairlearn Fallback Active"

---

### Step 3 — Select Target and Sensitive Attributes (0:35 – 0:55)

**Action:** In the attribute selector:
- Set **Target Variable** = `loan_approved`
- Set **Sensitive Attributes** = `gender`, `age`

**What to point out:**
- `gender` and `age` are auto-highlighted in blue before selection — Gemini's keyword matcher recognized them as likely demographic columns
- Distribution preview appears below the selector: male vs. female approval rates visible in mini histogram
- Multi-select confirms both attributes will be analyzed independently

---

### Step 4 — Gemini Scenario Auto-Detection (0:55 – 1:10)

**Action:** Click "Detect Scenario" — Gemini 2.0 Flash analyzes the dataset.

**What to point out:**
- Spinner runs for ~2 seconds
- Result card appears:

  > "**Lending Dataset** — 94% confidence"
  > "Column names 'loan_approved', 'credit_score', and 'loan_amount' indicate a consumer lending context."

- User can override this if needed (dropdown with Hiring / Healthcare / Education / Other)
- Scenario will be used to tailor all subsequent Gemini explanations

---

### Step 5 — Bias Audit Score (1:10 – 1:25)

**Action:** Click "Run Analysis" — full fairness analysis executes.

**What to point out:**
- Animated circular gauge fills to **34 / 100**
- Grade letter **F** displayed in red at the center
- Score breakdown tooltip: SPD contributed 40% weight, DI contributed 35%, EOD 15%, AOD 10%
- Grade F = "Severe bias — immediate remediation required"

---

### Step 6 — Fairness Metrics Deep-Dive (1:25 – 1:45)

**Action:** Scroll to the Metrics Dashboard. Point to the metric cards and bar charts.

**What to point out:**

**For `gender` attribute:**
- **SPD = 0.34** — tagged **HIGH** severity (red badge)
  - Males are approved 34 percentage points more often than females
- **DI = 0.54** — tagged **HIGH** severity
  - DI < 0.8 breaches the legal 80% rule (EEOC / EU AI Act standard)
  - Tooltip: "A disparate impact below 0.8 may indicate illegal discrimination"

**For `age` attribute:**
- SPD = 0.18 (MEDIUM severity)
- DI = 0.71 (HIGH severity — still below 0.8)

- Radar chart shows the overall bias profile across all four metrics simultaneously
- Group heatmap color-codes every metric x attribute combination at a glance

---

### Step 7 — Gemini Bias Copilot (1:45 – 2:05)

**Action:** Click the floating chat bubble in the bottom-right corner to open the Bias Copilot. Type:

> "Why is gender biased in this dataset?"

**What to point out:**
- Copilot responds in ~3 seconds with a plain-English explanation, e.g.:

  > "The gender attribute shows high bias because female applicants receive loan approvals at a rate 34% lower than male applicants (SPD = 0.34). The Disparate Impact of 0.54 is well below the legal 0.8 threshold, which may expose the lender to regulatory risk. This disparity is likely driven by historical lending patterns reflected in the training data, and the 'income' column may act as a proxy for gender due to its high correlation."

- Copilot is context-aware: it knows the session's metrics, scenario, and selected attributes
- Ask a follow-up: "What does DI mean?" — Copilot explains Disparate Impact without losing context

---

### Step 8 — Run Mitigation (2:05 – 2:25)

**Action:** Navigate to the Mitigation page. Click "Run Both Techniques."

**What to point out:**
- Both **Reweighing** and **Threshold Adjustment** run in parallel (~4 seconds)
- Side-by-side comparison table appears:

  | Metric | Original | Reweighing | Threshold Adj. |
  |---|---|---|---|
  | SPD (gender) | 0.34 | **0.06** | 0.14 |
  | DI (gender) | 0.54 | **0.91** | 0.79 |
  | EOD (gender) | 0.28 | **0.04** | 0.11 |
  | AOD (gender) | 0.26 | **0.05** | 0.12 |

- **Reweighing** wins across all metrics — highlighted with a green "Recommended" badge
- SPD drops from 0.34 to **0.06** — a 82% reduction in gender disparity

---

### Step 9 — Trade-off Chart (2:25 – 2:40)

**Action:** Scroll to the Trade-off Chart below the mitigation table.

**What to point out:**
- Scatter plot: X-axis = Bias Score (lower = less biased), Y-axis = Accuracy
- Three labeled data points:
  - **Original**: Bias=0.34, Accuracy=88%
  - **Reweighing**: Bias=0.06, Accuracy=**86%** — only 2% accuracy loss for 82% bias reduction
  - **Threshold Adj.**: Bias=0.14, Accuracy=84% — larger accuracy tradeoff, less bias reduction
- Reweighing clearly dominates: nearly as accurate as the original while being dramatically fairer
- Before/After table below shows full Accuracy, Precision, Recall, F1 with color-coded deltas

---

### Step 10 — Download PDF Report (2:40 – 3:00)

**Action:** Click the "Download Audit Report" button (or navigate to the Report page).

**What to point out:**
- PDF generates server-side in ~2 seconds
- File downloads as: `fairlens_audit_{session_id}_{timestamp}.pdf`
- Report contains 4 pages:
  - **Page 1**: Dataset summary — 1,000 rows, 7 columns, Lending scenario, binary classification
  - **Page 2**: Bias findings — all metric values with severity labels, Audit Score 34/100 grade F
  - **Page 3**: Before/After mitigation comparison table
  - **Page 4**: Gemini recommendations — actionable next steps, proxy features identified, regulatory notes
- This PDF is ready to hand to a compliance officer or include in a model card

---

## Key Talking Points for Judges

1. **End-to-End**: FairLens covers the full fairness audit lifecycle — detect, explain, mitigate, report
2. **Gemini Integration**: Three distinct Gemini use cases: scenario detection, explanation engine, Copilot chat
3. **Legal Alignment**: DI threshold of 0.8 aligns with EEOC 80% rule and EU AI Act requirements
4. **No Bias Expertise Required**: Plain-English explanations and tooltips make the platform accessible to non-technical users
5. **Actionable Output**: Not just metrics — mitigation runs automatically with performance trade-off analysis
6. **Production-Ready Design**: Session management, validation layer, Fairlearn fallback, statistical significance checks

---

## Git Commands Used for This Demo Branch

```bash
git init
git remote add origin https://github.com/tanmayjhanjhari/fairlens.git
git checkout -b dev
git checkout -b feature/part-0
git add .
git commit -m "chore: initial project setup and requirements"
git push origin feature/part-0
```
