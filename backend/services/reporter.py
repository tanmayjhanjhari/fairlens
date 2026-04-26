"""
ByUs — PDF Report Generator Service

Generates a 3-page professional audit report using ReportLab.

Page 1 — Dataset Overview + Audit Score + Metric Summary Table
Page 2 — Detailed Findings per Attribute (group stats, proxy features, Gemini explanation)
Page 3 — Mitigation Results (before/after table, effects, winner, footer)
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Brand colours ──────────────────────────────────────────────────────────────
TEAL = colors.HexColor("#14B8A6")
NAVY = colors.HexColor("#0F172A")
WHITE = colors.white
LIGHT_BG = colors.HexColor("#F0FDFA")
SCORE_GREEN = colors.HexColor("#22C55E")
SCORE_ORANGE = colors.HexColor("#F59E0B")
SCORE_RED = colors.HexColor("#EF4444")
SEVERITY_COLORS = {
    "high": colors.HexColor("#EF4444"),
    "medium": colors.HexColor("#F59E0B"),
    "low": colors.HexColor("#22C55E"),
}


def _score_color(score: float) -> colors.Color:
    if score > 75:
        return SCORE_GREEN
    if score > 50:
        return SCORE_ORANGE
    return SCORE_RED


def _grade(score: float) -> str:
    if score >= 80:
        return "A"
    if score >= 60:
        return "B"
    if score >= 40:
        return "C"
    return "F"


def _fmt(val: Any, decimals: int = 4) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def _delta_str(val: float | None) -> str:
    if val is None:
        return "N/A"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.4f}"


class ReportGenerator:
    """Generate a multi-page PDF audit report from a ByUs session dict."""

    def generate(self, session_data: dict[str, Any]) -> bytes:
        """
        Build and return PDF bytes.

        Parameters
        ----------
        session_data : dict
            The raw session dict from ``app.state.sessions[session_id]``.
        """
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
            title="ByUs Bias Audit Report",
            author="ByUs",
        )

        styles = getSampleStyleSheet()
        # Custom styles
        h1 = ParagraphStyle(
            "fl_h1", parent=styles["Heading1"],
            textColor=NAVY, fontSize=20, spaceAfter=4, leading=24,
        )
        h2 = ParagraphStyle(
            "fl_h2", parent=styles["Heading2"],
            textColor=TEAL, fontSize=13, spaceBefore=10, spaceAfter=4,
        )
        body = ParagraphStyle(
            "fl_body", parent=styles["BodyText"],
            textColor=NAVY, fontSize=9, leading=13,
        )
        small = ParagraphStyle(
            "fl_small", parent=styles["BodyText"],
            textColor=colors.HexColor("#64748B"), fontSize=8, leading=11,
        )
        bold_body = ParagraphStyle(
            "fl_bold", parent=body, fontName="Helvetica-Bold",
        )

        story: list = []

        # ═════════════════════════════════════════════════════════════════════
        # PAGE 1 — Dataset Overview
        # ═════════════════════════════════════════════════════════════════════
        story += self._page1(session_data, h1, h2, body, small, bold_body)
        story.append(PageBreak())

        # ═════════════════════════════════════════════════════════════════════
        # PAGE 2 — Detailed Findings per Attribute
        # ═════════════════════════════════════════════════════════════════════
        story += self._page2(session_data, h1, h2, body, small)
        story.append(PageBreak())

        # ═════════════════════════════════════════════════════════════════════
        # PAGE 3 — Mitigation Results
        # ═════════════════════════════════════════════════════════════════════
        story += self._page3(session_data, h1, h2, body, small, bold_body)

        doc.build(story)
        buf.seek(0)
        return buf.read()

    # ── Page 1 ────────────────────────────────────────────────────────────────

    def _page1(
        self,
        sd: dict,
        h1: ParagraphStyle,
        h2: ParagraphStyle,
        body: ParagraphStyle,
        small: ParagraphStyle,
        bold: ParagraphStyle,
    ) -> list:
        story = []
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        bias_results: dict = sd.get("bias_results", {})
        validation: dict = sd.get("validation", {})
        scenario_data: dict = sd.get("scenario", {})

        audit_score = float(bias_results.get("audit_score", 0) or 0)
        grade_letter = _grade(audit_score)
        score_col = _score_color(audit_score)

        story.append(Paragraph("ByUs Bias Audit Report", h1))
        story.append(HRFlowable(width="100%", thickness=2, color=TEAL, spaceAfter=8))
        story.append(Paragraph(f"Generated: {generated_at}", small))
        story.append(Spacer(1, 0.3 * cm))

        # ── Summary info ──────────────────────────────────────────────────────
        story.append(Paragraph("Dataset Overview", h2))
        overview_rows = [
            ["Field", "Value"],
            ["File", sd.get("filename", "N/A")],
            ["Rows", str(sd.get("row_count", "N/A"))],
            ["Target Column", sd.get("target_col", "N/A")],
            ["Sensitive Attributes", ", ".join(sd.get("sensitive_attrs", []))],
            ["Target Type", validation.get("target_type", "N/A")],
            ["Analysis Engine", validation.get("engine", "N/A")],
            ["Scenario Detected", scenario_data.get("scenario", "N/A").capitalize()
             + f" ({scenario_data.get('confidence_pct', 'N/A')}% confidence)"],
        ]
        story.append(self._table(overview_rows))
        story.append(Spacer(1, 0.4 * cm))

        # ── Audit Score banner ────────────────────────────────────────────────
        story.append(Paragraph("Bias Audit Score", h2))
        score_html = (
            f'<font color="{score_col.hexval()}" size="42"><b>{grade_letter}</b></font>'
            f'&nbsp;&nbsp;<font size="20" color="{score_col.hexval()}"><b>{audit_score:.1f} / 100</b></font>'
            f'&nbsp;&nbsp;<font size="11" color="#64748B">Overall Severity: '
            f'<b>{bias_results.get("overall_severity", "N/A").upper()}</b></font>'
        )
        score_style = ParagraphStyle("fl_score", leading=48)
        story.append(Paragraph(score_html, score_style))
        story.append(Spacer(1, 0.4 * cm))

        # ── Validation warnings ───────────────────────────────────────────────
        warnings: list[str] = validation.get("warnings", [])
        if warnings:
            story.append(Paragraph("Validation Warnings", h2))
            for w in warnings:
                story.append(Paragraph(f"⚠  {w}", small))
            story.append(Spacer(1, 0.3 * cm))

        # ── Metrics summary table ─────────────────────────────────────────────
        story.append(Paragraph("Fairness Metrics Summary", h2))
        metrics_header = ["Sensitive Attribute", "Severity", "SPD", "DI", "EOD", "Legal Flag (DI<0.8)"]
        metrics_rows = [metrics_header]

        for attr, m in bias_results.get("metrics_per_attr", {}).items():
            if isinstance(m, dict) and "spd" in m:
                metrics_rows.append([
                    attr,
                    (m.get("severity") or "N/A").upper(),
                    _fmt(m.get("spd")),
                    _fmt(m.get("di")),
                    _fmt(m.get("eod")),
                    "YES ⚠" if m.get("legal_flag") else "No",
                ])

        if len(metrics_rows) > 1:
            story.append(self._table(metrics_rows, severity_col=1))
        else:
            story.append(Paragraph("No metric data available. Run /api/analyze first.", small))

        return story

    # ── Page 2 ────────────────────────────────────────────────────────────────

    def _page2(
        self,
        sd: dict,
        h1: ParagraphStyle,
        h2: ParagraphStyle,
        body: ParagraphStyle,
        small: ParagraphStyle,
    ) -> list:
        story = []
        story.append(Paragraph("Detailed Findings", h1))
        story.append(HRFlowable(width="100%", thickness=2, color=TEAL, spaceAfter=8))

        bias_results: dict = sd.get("bias_results", {})
        explanations: dict = sd.get("explanations", {})
        gemini_explanations: dict = sd.get("gemini_explanations", {})

        for attr, m in bias_results.get("metrics_per_attr", {}).items():
            story.append(Paragraph(f"Attribute: {attr}", h2))

            if "error" in m:
                story.append(Paragraph(f"Analysis error: {m['error']}", small))
                story.append(Spacer(1, 0.2 * cm))
                continue

            # Group stats table
            group_stats: dict = m.get("group_stats", {})
            if group_stats:
                story.append(Paragraph("Group Statistics", h2))
                gs_header = ["Group", "Count", "% of Total", "Positive Rate"]
                gs_rows = [gs_header]
                for group_val, gs in group_stats.items():
                    gs_rows.append([
                        str(group_val),
                        str(gs.get("count", "N/A")),
                        _fmt(gs.get("pct_of_total"), 1) + "%",
                        _fmt(gs.get("positive_rate")),
                    ])
                story.append(self._table(gs_rows))
                story.append(Spacer(1, 0.2 * cm))

            # Privileged/unprivileged
            priv = m.get("privileged_group", "N/A")
            unpriv = m.get("unprivileged_group", "N/A")
            story.append(Paragraph(
                f"Privileged group: <b>{priv}</b> | "
                f"Unprivileged group: <b>{unpriv}</b> | "
                f"Statistically significant: <b>{'Yes' if m.get('statistically_significant') else 'No'}</b>",
                small,
            ))
            story.append(Spacer(1, 0.2 * cm))

            # Proxy features
            expl: dict = explanations.get(attr, {})
            proxy_features: list = expl.get("proxy_features", [])
            if proxy_features:
                story.append(Paragraph("Proxy Feature Detection", h2))
                pf_header = ["Feature", "Correlation", "Strength", "Interpretation"]
                pf_rows = [pf_header]
                for pf in proxy_features:
                    pf_rows.append([
                        pf.get("feature", ""),
                        _fmt(pf.get("correlation")),
                        pf.get("strength", "").capitalize(),
                        Paragraph(pf.get("interpretation", ""), small),
                    ])
                story.append(self._table(pf_rows))
                story.append(Spacer(1, 0.2 * cm))

            # Gemini explanation
            gemini_expl = gemini_explanations.get(attr) or expl.get("plain_reason")
            if gemini_expl:
                story.append(Paragraph("AI Explanation", h2))
                story.append(Paragraph(gemini_expl.replace("\n\n", "<br/><br/>"), body))
                story.append(Spacer(1, 0.3 * cm))

        if not bias_results.get("metrics_per_attr"):
            story.append(Paragraph("No detailed findings available.", small))

        return story

    # ── Page 3 ────────────────────────────────────────────────────────────────

    def _page3(
        self,
        sd: dict,
        h1: ParagraphStyle,
        h2: ParagraphStyle,
        body: ParagraphStyle,
        small: ParagraphStyle,
        bold: ParagraphStyle,
    ) -> list:
        story = []
        story.append(Paragraph("Mitigation Results", h1))
        story.append(HRFlowable(width="100%", thickness=2, color=TEAL, spaceAfter=8))

        mitigation: dict = sd.get("mitigation_results", {})

        if not mitigation:
            story.append(Paragraph(
                "No mitigation data found. Run POST /api/mitigate to generate results.", small
            ))
        else:
            winner = mitigation.get("winner", "N/A")
            story.append(Paragraph(
                f"Recommended Technique: <b>{winner.capitalize()}</b>",
                bold,
            ))
            story.append(Spacer(1, 0.3 * cm))

            # ── Side-by-side metrics comparison ──────────────────────────────
            story.append(Paragraph("Fairness Metrics — Before vs. After", h2))

            rew = mitigation.get("reweigh", {})
            thr = mitigation.get("threshold", {})
            rew_before = rew.get("before", {})
            rew_after = rew.get("after", {})
            thr_after = thr.get("after", {})

            def _get_m(d, key):
                if d is None: return None
                return d.get(key.upper(), d.get(key.lower()))

            comparison_rows = [
                ["Metric", "Original", "After Reweighing", "After Threshold Adj."],
                ["SPD",
                 _fmt(_get_m(rew_before, "spd")),
                 _fmt(_get_m(rew_after, "spd")),
                 _fmt(_get_m(thr_after, "spd"))],
                ["DI",
                 _fmt(_get_m(rew_before, "di")),
                 _fmt(_get_m(rew_after, "di")),
                 _fmt(_get_m(thr_after, "di"))],
                ["EOD",
                 _fmt(_get_m(rew_before, "eod")),
                 _fmt(_get_m(rew_after, "eod")),
                 _fmt(_get_m(thr_after, "eod"))],
                ["AOD",
                 _fmt(_get_m(rew_before, "aod")),
                 _fmt(_get_m(rew_after, "aod")),
                 _fmt(_get_m(thr_after, "aod"))],
                ["Accuracy",
                 _fmt(_get_m(rew_before, "accuracy")),
                 _fmt(_get_m(rew_after, "accuracy")),
                 _fmt(_get_m(thr_after, "accuracy"))],
                ["F1",
                 _fmt(_get_m(rew_before, "f1")),
                 _fmt(_get_m(rew_after, "f1")),
                 _fmt(_get_m(thr_after, "f1"))],
            ]
            story.append(self._table(comparison_rows))
            story.append(Spacer(1, 0.4 * cm))

            # ── Effects table ─────────────────────────────────────────────────
            story.append(Paragraph("Mitigation Effects", h2))
            rew_eff = rew.get("effects", {})
            thr_eff = thr.get("effects", {})

            effects_rows = [
                ["Effect", "Reweighing", "Threshold Adj."],
                ["Accuracy Delta",
                 _delta_str(rew_eff.get("accuracy_delta")),
                 _delta_str(thr_eff.get("accuracy_delta"))],
                ["Precision Delta",
                 _delta_str(rew_eff.get("precision_delta")),
                 _delta_str(thr_eff.get("precision_delta"))],
                ["Recall Delta",
                 _delta_str(rew_eff.get("recall_delta")),
                 _delta_str(thr_eff.get("recall_delta"))],
                ["F1 Delta",
                 _delta_str(rew_eff.get("f1_delta")),
                 _delta_str(thr_eff.get("f1_delta"))],
                ["SPD Delta",
                 _delta_str(rew_eff.get("spd_delta")),
                 _delta_str(thr_eff.get("spd_delta"))],
                ["Bias Reduction %",
                 _fmt(rew_eff.get("bias_reduction_pct"), 1) + "%",
                 _fmt(thr_eff.get("bias_reduction_pct"), 1) + "%"],
                ["Accuracy Retained %",
                 _fmt(rew_eff.get("accuracy_retained_pct"), 1) + "%",
                 _fmt(thr_eff.get("accuracy_retained_pct"), 1) + "%"],
            ]
            story.append(self._table(effects_rows))
            story.append(Spacer(1, 0.4 * cm))

        # ── Gemini recommendations ────────────────────────────────────────────
        gemini_explanations: dict = sd.get("gemini_explanations", {})
        if gemini_explanations:
            story.append(Paragraph("Bias Narrative", h2))
            for attr, expl_text in gemini_explanations.items():
                story.append(Paragraph(f"<b>{attr}:</b>", body))
                story.append(Paragraph(expl_text.replace("\n\n", "<br/><br/>"), body))
                story.append(Spacer(1, 0.2 * cm))

        # ── AI Action Plan ────────────────────────────────────────────────────
        action_plan = sd.get("action_plan")
        if not action_plan:
            try:
                from services.gemini_service import GeminiService
                gemini = GeminiService()
                action_plan = gemini.get_action_plan(sd)
            except Exception:
                action_plan = "No action plan available."

        if action_plan:
            story.append(Paragraph("AI Action Plan", h2))
            story.append(Paragraph(action_plan.replace("\n\n", "<br/><br/>"), body))
            story.append(Spacer(1, 0.2 * cm))

        # ── Footer ────────────────────────────────────────────────────────────
        story.append(Spacer(1, 0.6 * cm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=TEAL, spaceAfter=6))
        story.append(Paragraph(
            "Generated by ByUs. For audit purposes only. "
            "This report does not constitute legal advice.",
            ParagraphStyle(
                "footer",
                parent=small,
                textColor=colors.HexColor("#94A3B8"),
                alignment=1,  # centre
            ),
        ))

        return story

    # ── Table builder ─────────────────────────────────────────────────────────

    @staticmethod
    def _table(
        data: list[list],
        severity_col: int | None = None,
    ) -> Table:
        t = Table(data, repeatRows=1)
        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), TEAL),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("TEXTCOLOR", (0, 1), (-1, -1), NAVY),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ]

        # Colour the severity column cells
        if severity_col is not None:
            for row_idx, row in enumerate(data[1:], start=1):
                cell_val = str(row[severity_col]).lower()
                sev_col = SEVERITY_COLORS.get(cell_val.split()[0], NAVY)
                style_cmds.append(
                    ("TEXTCOLOR", (severity_col, row_idx), (severity_col, row_idx), sev_col)
                )
                style_cmds.append(
                    ("FONTNAME", (severity_col, row_idx), (severity_col, row_idx), "Helvetica-Bold")
                )

        t.setStyle(TableStyle(style_cmds))
        return t
