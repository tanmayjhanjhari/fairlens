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

    # ── Fallback Methods ──────────────────────────────────────────────────────

    def _fallback_explanation(self, sensitive_attr: str, metrics: dict,
                               plain_reason: str, scenario: str) -> str:
        """Generate plain-English explanation from data when Gemini unavailable."""
        spd = metrics.get("SPD", 0) or 0
        di = metrics.get("DI", 1) or 1
        severity = metrics.get("severity", "unknown")
        group_stats = metrics.get("group_stats", {})
        
        # Find best and worst groups
        if group_stats:
            sorted_groups = sorted(group_stats.items(),
                                   key=lambda x: x[1].get("positive_rate", 0))
            worst_group = sorted_groups[0]
            best_group = sorted_groups[-1]
            worst_name = worst_group[0]
            worst_rate = worst_group[1].get("positive_rate", 0)
            best_name = best_group[0]
            best_rate = best_group[1].get("positive_rate", 0)
            gap_pct = round((best_rate - worst_rate) * 100, 1)
            
            para1 = (
                f"In this {scenario} dataset, '{sensitive_attr}' shows a "
                f"{severity}-severity bias. The group '{best_name}' receives "
                f"positive outcomes {best_rate:.0%} of the time, while "
                f"'{worst_name}' receives them only {worst_rate:.0%} of the time "
                f"— a gap of {gap_pct} percentage points."
            )
        else:
            gap_pct = round(abs(spd) * 100, 1)
            para1 = (
                f"In this {scenario} dataset, '{sensitive_attr}' shows a "
                f"{severity}-severity bias with a {gap_pct}% outcome gap "
                f"between demographic groups."
            )
        
        # Legal flag
        if di < 0.8:
            legal_note = (
                f"The Disparate Impact ratio is {di:.3f}, which falls below "
                f"the legal threshold of 0.8 (the '80% rule' used in employment "
                f"and lending law). This level of disparity may expose the "
                f"organization to regulatory risk."
            )
        else:
            legal_note = (
                f"The Disparate Impact ratio is {di:.3f}, which is above the "
                f"legal 0.8 threshold, meaning the disparity is less likely to "
                f"trigger regulatory concern — but the outcome gap still warrants "
                f"attention."
            )
        
        # Root cause
        if plain_reason and "correlates" in plain_reason.lower():
            cause = (
                f"The likely cause is that another feature in the dataset acts "
                f"as a hidden stand-in for '{sensitive_attr}'. {plain_reason} "
                f"This means even if '{sensitive_attr}' is removed from the model, "
                f"the bias can persist through this proxy feature."
            )
        else:
            cause = (
                f"The bias likely reflects historical patterns in the training data "
                f"where '{sensitive_attr}' was correlated with outcomes — possibly "
                f"due to systemic factors outside the model itself. "
                f"{plain_reason}"
            )
        
        return f"{para1}\n\n{legal_note}\n\n{cause}"

    def _fallback_action_plan(self, session_data: dict) -> str:
        """Generate specific action plan from data when Gemini unavailable."""
        metrics_per_attr = session_data.get("metrics_per_attr", {})
        mitigation = session_data.get("mitigation", {})
        winner = mitigation.get("winner", "reweigh")
        scenario = session_data.get("scenario", "this dataset")
        
        lines = []
        for i, (attr, m) in enumerate(metrics_per_attr.items(), 1):
            spd = abs(m.get("SPD", 0) or 0)
            di = m.get("DI", 1) or 1
            severity = m.get("severity", "unknown")
            proxies = m.get("proxy_features", [])
            top_proxy = proxies[0].get("feature") if proxies else None
            group_stats = m.get("group_stats", {})
            
            if group_stats:
                sorted_g = sorted(group_stats.items(),
                                  key=lambda x: x[1].get("positive_rate", 0))
                worst = sorted_g[0][0] if sorted_g else "minority group"
                best = sorted_g[-1][0] if sorted_g else "majority group"
            else:
                worst, best = "disadvantaged group", "advantaged group"
            
            if severity == "high":
                action = (
                    f"Step {i} — Fix '{attr}' bias (HIGH severity, SPD={spd:.3f}): "
                    f"The group '{worst}' is significantly disadvantaged compared to "
                    f"'{best}' in {scenario}."
                )
                if top_proxy:
                    action += (
                        f" Remove or transform '{top_proxy}' from your feature set "
                        f"— it is acting as a hidden proxy for '{attr}'."
                    )
            elif severity == "medium":
                action = (
                    f"Step {i} — Review '{attr}' bias (MEDIUM severity, SPD={spd:.3f}): "
                    f"A {spd:.0%} outcome gap exists between groups."
                )
                if di < 0.8:
                    action += (
                        f" DI={di:.3f} is below the legal 0.8 threshold — "
                        f"document this finding for compliance records."
                    )
            else:
                action = (
                    f"Step {i} — Monitor '{attr}' (LOW severity, SPD={spd:.3f}): "
                    f"Bias is within acceptable range but should be tracked "
                    f"over time as the model is retrained."
                )
            lines.append(action)
        
        # Mitigation step
        mitigation_step = (
            f"Step {len(lines)+1} — Apply {winner} mitigation technique "
            f"before retraining. This technique was identified as best for "
            f"this dataset based on bias reduction vs accuracy trade-off analysis."
        )
        lines.append(mitigation_step)
        
        # Monitoring step  
        lines.append(
            f"Step {len(lines)+1} — Re-audit after retraining using ByUs "
            f"to verify bias reduction. Set up quarterly fairness reviews "
            f"as part of your model governance process."
        )
        
        return "\n\n".join(lines)

    def _fallback_chat(self, user_message: str, session_context: dict) -> str:
        """Rule-based chat response when Gemini unavailable."""
        msg = user_message.lower()
        metrics = session_context.get("metrics_per_attr", {})
        scenario = session_context.get("scenario", "this dataset")
        audit_score = session_context.get("audit_score", "N/A")
        
        if any(w in msg for w in ["spd","statistical parity","parity difference"]):
            attr_info = "; ".join([
                f"{a}: SPD={m.get('SPD','N/A')}"
                for a, m in metrics.items()
            ])
            return (f"Statistical Parity Difference (SPD) measures the outcome "
                    f"rate gap between groups. Ideal value is 0. "
                    f"For your dataset: {attr_info}. "
                    f"Values above 0.1 indicate meaningful bias.")
        
        elif any(w in msg for w in ["di","disparate impact","legal","80%","80 percent"]):
            flags = [a for a, m in metrics.items() if (m.get("DI") or 1) < 0.8]
            if flags:
                return (f"Disparate Impact (DI) measures the ratio of positive "
                        f"outcomes between groups. The legal threshold is 0.8. "
                        f"Your dataset fails this threshold for: {', '.join(flags)}. "
                        f"This means these attributes may create legal liability.")
            else:
                return ("Your dataset passes the legal 0.8 DI threshold for all "
                        "attributes. However, passing the legal threshold doesn't "
                        "mean no bias exists — SPD may still show meaningful gaps.")
        
        elif any(w in msg for w in ["mitigation","fix","reduce","reweigh","threshold"]):
            return (f"Two mitigation techniques were applied to your dataset. "
                    f"Reweighing adjusts training data weights so each group "
                    f"is represented fairly. Threshold adjustment finds per-group "
                    f"decision thresholds that equalize true positive rates. "
                    f"Check the Remediation page to compare their results "
                    f"and accuracy trade-offs.")
        
        elif any(w in msg for w in ["score","grade","audit"]):
            return (f"Your Bias Audit Score is {audit_score}/100. "
                    f"This composite score combines SPD, DI, and EOD penalties "
                    f"across all sensitive attributes. Grade A (85+) is fair, "
                    f"B (70-84) has minor issues, C (50-69) has moderate bias, "
                    f"F (below 50) requires immediate action.")
        
        elif any(w in msg for w in ["why","cause","reason","proxy","how"]):
            explanations = []
            for attr, m in metrics.items():
                proxies = m.get("proxy_features", [])
                if proxies:
                    top = proxies[0]
                    explanations.append(
                        f"For '{attr}': '{top.get('feature')}' is the strongest "
                        f"proxy feature (r={top.get('correlation','N/A'):.2f})"
                    )
            if explanations:
                return ("Bias often exists because other features act as hidden "
                        f"stand-ins for sensitive attributes. " +
                        ". ".join(explanations) + ". Removing or transforming "
                        "these proxy features can reduce indirect discrimination.")
            else:
                return ("Bias in this dataset likely reflects historical patterns "
                        "in how outcomes were recorded, rather than a single "
                        "proxy feature. The training data itself may embed "
                        "systemic inequalities from the real world.")
        
        else:
            # Generic helpful response
            attrs = list(metrics.keys())
            high_attrs = [a for a,m in metrics.items() if m.get("severity")=="high"]
            if high_attrs:
                return (f"Your dataset shows HIGH severity bias in: "
                        f"{', '.join(high_attrs)}. The Bias Audit Score is "
                        f"{audit_score}/100. I can explain specific metrics "
                        f"(try asking about SPD, DI, or mitigation techniques), "
                        f"or why bias exists in your data.")
            else:
                return (f"Your dataset analysis is complete. Bias Audit Score: "
                        f"{audit_score}/100. Sensitive attributes analyzed: "
                        f"{', '.join(attrs)}. Ask me about specific metrics, "
                        f"mitigation results, or what actions to take.")

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
        except Exception as e:
            # Rule-based scenario detection from column names
            cols_lower = [c.lower() for c in columns]
            if any(w in cols_lower for w in ["loan","credit","approved","default","risk","debt"]):
                return {"scenario": "Lending", "confidence_pct": 80,
                        "reason": "Detected lending-related columns"}
            elif any(w in cols_lower for w in ["hired","job","salary","occupation","employed"]):
                return {"scenario": "Hiring", "confidence_pct": 80,
                        "reason": "Detected employment-related columns"}
            elif any(w in cols_lower for w in ["diagnosis","disease","patient","hospital","medical"]):
                return {"scenario": "Healthcare", "confidence_pct": 80,
                        "reason": "Detected healthcare-related columns"}
            elif any(w in cols_lower for w in ["recid","crime","arrest","prison","sentence"]):
                return {"scenario": "Criminal Justice", "confidence_pct": 80,
                        "reason": "Detected criminal justice columns"}
            elif any(w in cols_lower for w in ["grade","gpa","score","admit","student"]):
                return {"scenario": "Education", "confidence_pct": 80,
                        "reason": "Detected education-related columns"}
            else:
                return {"scenario": "General Classification", "confidence_pct": 60,
                        "reason": "Could not determine specific scenario from column names"}

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
        except Exception as e:
            # Use rule-based fallback — never show error to user
            return self._fallback_explanation(
                sensitive_attr, metrics, plain_reason, scenario
            )

    # ── Action Plan ───────────────────────────────────────────────────────────

    def get_action_plan(self, session_data: dict) -> str:
        scenario_dict = session_data.get("scenario", {})
        scenario = scenario_dict.get("scenario", "unknown") if isinstance(scenario_dict, dict) else str(scenario_dict)
        
        bias_results = session_data.get("bias_results", {})
        metrics_summary = bias_results.get("metrics_per_attr", {})
        
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
        except Exception as e:
            return self._fallback_action_plan(session_data)

    # ── Copilot chat ──────────────────────────────────────────────────────────

    def chat(self, user_message: str, history: list, session_context: dict) -> str:
        import json

        try:
            # Build a clean context summary (not full raw dict which can be huge)
            ctx_summary = {
                "scenario": session_context.get("scenario", "unknown"),
                "audit_score": session_context.get("audit_score", "N/A"),
                "overall_severity": session_context.get("overall_severity", "unknown"),
                "sensitive_attrs": list(session_context.get("metrics_per_attr", {}).keys()),
                "top_metrics": {
                    attr: {
                        "SPD": m.get("SPD"),
                        "DI": m.get("DI"),
                        "severity": m.get("severity")
                    }
                    for attr, m in session_context.get("metrics_per_attr", {}).items()
                }
            }

            system_prompt = f"""You are ByUs AI Bias Copilot, an expert in AI fairness and ethics.
You have analyzed a dataset with these findings:
{json.dumps(ctx_summary, indent=2)}

Rules:
- Answer in plain English, no jargon
- Be specific to the actual findings above
- Keep responses under 150 words
- If asked about metrics, refer to the actual numbers above
- Be helpful and actionable"""

            # Format history for Gemini
            gemini_history = []
            for msg in history[-6:]:  # last 6 messages only to save tokens
                role = "user" if msg.get("role") == "user" else "model"
                gemini_history.append({
                    "role": role,
                    "parts": [{"text": msg.get("content", "")}]
                })

            chat_session = _model.start_chat(history=gemini_history)
            full_message = f"{system_prompt}\n\nUser question: {user_message}"
            response = chat_session.send_message(full_message)
            return response.text.strip()

        except Exception as e:
            error_str = str(e).lower()
            if "quota" in error_str or "429" in error_str or "exhausted" in error_str:
                # Use rule-based chat — never show quota error to user
                return self._fallback_chat(user_message, session_context)
            elif "api_key" in error_str or "403" in error_str:
                return ("I'm having trouble with my AI connection. "
                        "Your analysis results are still fully available — "
                        "check the metrics and mitigation pages for complete findings.")
            else:
                return self._fallback_chat(user_message, session_context)

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
