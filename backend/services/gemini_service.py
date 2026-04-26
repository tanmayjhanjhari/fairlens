"""
ByUs — Gemini AI Service

Wraps Google Gemini 2.0 Flash for:
  - Dataset scenario detection
  - Plain-English bias explanation for managers
  - Multi-turn Bias Copilot chat
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

_api_key = os.getenv("GEMINI_API_KEY", "")
if _api_key:
    genai.configure(api_key=_api_key)

_model = genai.GenerativeModel("gemini-2.0-flash")


class GeminiService:
    """Stateless wrapper around the Gemini 2.0 Flash model."""

    # ── Scenario detection ────────────────────────────────────────────────────

    def detect_scenario(self, columns: list[str]) -> dict[str, Any]:
        """
        Classify the dataset into one of: hiring, lending, healthcare,
        education, other.

        Returns
        -------
        dict with keys: scenario, confidence_pct, reason
        """
        prompt = (
            f"Dataset columns: {columns}. "
            "Classify into exactly one category: hiring, lending, healthcare, education, other. "
            'Return ONLY valid JSON with no markdown formatting: '
            '{"scenario": string, "confidence_pct": number, "reason": string}'
        )

        try:
            response = _model.generate_content(prompt)
            raw = response.text.strip()
            return self._parse_json(raw)
        except Exception:
            return {
                "scenario": "other",
                "confidence_pct": 50,
                "reason": "Could not detect scenario automatically.",
            }

    # ── Bias explanation ──────────────────────────────────────────────────────

    def explain_bias(
        self,
        metrics: dict[str, Any],
        sensitive_attr: str,
        scenario: str,
        plain_reason: str,
    ) -> str:
        """
        Generate a manager-friendly 3-paragraph explanation of the bias findings.

        Falls back to ``plain_reason`` if the Gemini API fails.
        """
        spd = metrics.get("spd", metrics.get("SPD", 0)) or 0
        di = metrics.get("di", metrics.get("DI", 1)) or 1
        severity = metrics.get("severity", "unknown")

        prompt = f"""You are ByUs AI Copilot explaining bias findings to a non-technical business manager.

Context:
- Dataset scenario: {scenario}
- Sensitive attribute: '{sensitive_attr}'
- SPD = {metrics.get('SPD', 'N/A')} (outcome gap between groups)
- DI = {metrics.get('DI', 'N/A')} (ratio of positive outcomes, legal threshold is 0.8)
- Severity: {metrics.get('severity', 'unknown')}
- Top proxy feature: {plain_reason}

Write 3 SHORT paragraphs. Each paragraph max 2 sentences. Use plain English.
Paragraph 1: What this bias MEANS in the real world for this specific scenario (hiring/lending/healthcare etc). Be specific, not generic.
Paragraph 2: WHY it likely exists — explain the historical or societal reason, not just the math.
Paragraph 3: What HARM it causes to real people if not fixed. Give a concrete example.

Do NOT repeat the proxy feature statistics already shown above.
Do NOT use technical jargon like SPD, DI, EOD, AOD, logistic regression.
Do NOT start with 'Sure' or 'Certainly' or 'Of course'."""

        try:
            response = _model.generate_content(prompt)
            return response.text.strip()
        except Exception:
            return plain_reason

    # ── Action Plan ───────────────────────────────────────────────────────────

    def get_action_plan(self, session_data: dict) -> str:
        scenario = session_data.get("scenario", "unknown")
        metrics_summary = session_data.get("metrics_per_attr", {})
        mitigation = session_data.get("mitigation", {})
        winner = mitigation.get("winner", "reweigh")
        
        # Build specific context
        attr_summaries = []
        for attr, m in metrics_summary.items():
            if "error" in m: continue
            spd = m.get("spd", m.get("SPD", "N/A"))
            di = m.get("di", m.get("DI", "N/A"))
            severity = m.get("severity", "unknown")
            proxy = m.get("proxy_features", [{}])
            top_proxy = proxy[0].get("feature", "unknown") if proxy else "unknown"
            attr_summaries.append(f"- {attr}: SPD={spd}, DI={di}, severity={severity}, top proxy='{top_proxy}'")
        
        attrs_text = "\n".join(attr_summaries)
        
        prompt = f"""You are ByUs AI. Generate a specific 3-step action plan for this exact dataset.

Dataset scenario: {scenario}
Bias findings:
{attrs_text}
Recommended mitigation technique: {winner}

Write exactly 3 numbered action steps. Each step must:
- Be specific to THIS dataset and scenario (not generic advice)
- Mention the actual attribute name and proxy feature found
- Tell the organization exactly what to do and why
- Be 2-3 sentences maximum
- Use plain English, no jargon

Do NOT give generic advice like 'collect more data' or 'apply reweighing'.
Make each step actionable and specific to the findings above."""

        try:
            response = _model.generate_content(prompt)
            return response.text.strip()
        except Exception:
            # Fallback: generate rule-based specific plan
            lines = []
            for attr, m in metrics_summary.items():
                if "error" in m: continue
                proxy_list = m.get("proxy_features", [])
                top_proxy = proxy_list[0].get("feature", "a correlated feature") if proxy_list else "a correlated feature"
                spd = m.get("spd", m.get("SPD", 0))
                lines.append(f"1. For '{attr}': Remove or neutralize '{top_proxy}' from your model features — it has {spd:.0%} outcome gap and acts as a hidden discriminator.")
            lines.append(f"2. Apply {winner} mitigation technique before retraining your model to rebalance group representation.")
            lines.append("3. Re-audit after retraining to confirm bias reduction before deployment.")
            return "\n".join(lines)

    # ── Copilot chat ──────────────────────────────────────────────────────────

    def chat(
        self,
        user_message: str,
        history: list[dict[str, str]],
        session_context: dict[str, Any],
    ) -> str:
        """
        Multi-turn Bias Copilot conversation.

        Parameters
        ----------
        user_message : str
            The latest message from the user.
        history : list[dict]
            Previous turns as [{"role": "user"|"model", "content": str}, ...]
        session_context : dict
            Condensed analysis context injected as a system preamble.

        Returns
        -------
        str — Gemini's reply.
        """
        system_preamble = (
            "You are ByUs Bias Copilot, an expert AI assistant specialising in "
            "algorithmic fairness, bias detection, and ML ethics. "
            "Be concise, helpful, and explain fairness concepts in simple language. "
            f"Current analysis context: {json.dumps(session_context, default=str)}"
        )

        # Build Gemini-format history
        gemini_history: list[dict] = []

        # Inject system preamble as first model turn to simulate system instruction
        gemini_history.append(
            {"role": "user", "parts": ["Please confirm you understand your role."]}
        )
        gemini_history.append(
            {"role": "model", "parts": [system_preamble + " Understood — ready to assist."]}
        )

        for turn in history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "model") and content:
                gemini_history.append({"role": role, "parts": [content]})

        try:
            chat_session = _model.start_chat(history=gemini_history)
            response = chat_session.send_message(user_message)
            return response.text.strip()
        except Exception:
            return "I'm having trouble connecting. Please try again."

    # ── JSON parsing helper ───────────────────────────────────────────────────

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        """Strip markdown code fences then parse JSON."""
        # Remove ```json ... ``` or ``` ... ``` fences
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to extract the first {...} block
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {
            "scenario": "other",
            "confidence_pct": 50,
            "reason": "Could not parse Gemini response.",
        }
