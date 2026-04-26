"""
FairLens — Report Router

GET /api/report/{session_id}
  Generates and streams a multi-page PDF audit report using ReportLab.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    PageBreak,
)

router = APIRouter(tags=["Report"])

# ── Colour palette ────────────────────────────────────────────────────────────
BRAND_DARK = colors.HexColor("#0F172A")
BRAND_ACCENT = colors.HexColor("#6366F1")
SEVERITY_HIGH = colors.HexColor("#EF4444")
SEVERITY_MED = colors.HexColor("#F59E0B")
SEVERITY_LOW = colors.HexColor("#22C55E")
GRADE_COLORS = {
    "A": colors.HexColor("#22C55E"),
    "B": colors.HexColor("#84CC16"),
    "C": colors.HexColor("#F59E0B"),
    "F": colors.HexColor("#EF4444"),
}


@router.get("/report/{session_id}", status_code=status.HTTP_200_OK)
async def download_report(
    session_id: str,
    request: Request,
) -> StreamingResponse:
    """Generate and stream a PDF audit report for the given session."""
    sessions: dict = request.app.state.sessions

    if session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )

    session = sessions[session_id]

    if "bias_results" not in session:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No analysis results found. Run /api/analyze first.",
        )

    pdf_bytes = _build_pdf(session, session_id)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"fairlens_audit_{session_id[:8]}_{timestamp}.pdf"

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── PDF builder ───────────────────────────────────────────────────────────────

def _build_pdf(session: dict, session_id: str) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="FairLens Bias Audit Report",
        author="FairLens",
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], textColor=BRAND_DARK, fontSize=22, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=BRAND_ACCENT, fontSize=14, spaceBefore=12, spaceAfter=4)
    body = styles["BodyText"]
    bold_body = ParagraphStyle("bold_body", parent=body, fontName="Helvetica-Bold")

    bias_results: dict = session["bias_results"]
    validation: dict = session.get("validation", {})
    target_col: str = session.get("target_col", "N/A")
    sensitive_attrs: list = session.get("sensitive_attrs", [])
    filename: str = session.get("filename", "dataset.csv")
    row_count: int = session.get("row_count", 0)
    mitigation: dict = session.get("mitigation_results", {})

    audit_score = bias_results.get("audit_score", 0)
    grade = bias_results.get("grade", "F")
    overall_severity = bias_results.get("overall_severity", "N/A")
    metrics_per_attr: dict = bias_results.get("metrics_per_attr", {})

    story = []

    # ═══════════════════════════════════════════════════════════════
    # PAGE 1 — Cover + Dataset Summary
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("FairLens Bias Audit Report", h1))
    story.append(HRFlowable(width="100%", thickness=2, color=BRAND_ACCENT, spaceAfter=12))

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    story.append(Paragraph(f"Generated: {generated_at} | Session: {session_id[:8]}…", body))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("Dataset Summary", h2))
    summary_data = [
        ["Field", "Value"],
        ["File", filename],
        ["Rows", str(row_count)],
        ["Target Column", target_col],
        ["Sensitive Attributes", ", ".join(sensitive_attrs)],
        ["Target Type", validation.get("target_type", "N/A")],
        ["Analysis Engine", validation.get("engine", "N/A")],
    ]
    story.append(_table(summary_data))
    story.append(Spacer(1, 0.4 * cm))

    if validation.get("warnings"):
        story.append(Paragraph("Validation Warnings", h2))
        for w in validation["warnings"]:
            story.append(Paragraph(f"• {w}", body))
        story.append(Spacer(1, 0.3 * cm))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════
    # PAGE 2 — Bias Findings
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("Bias Findings", h1))
    story.append(HRFlowable(width="100%", thickness=2, color=BRAND_ACCENT, spaceAfter=12))

    # Audit Score banner
    grade_color = GRADE_COLORS.get(grade, SEVERITY_HIGH)
    story.append(
        Paragraph(
            f'<font color="{grade_color.hexval()}" size="36"><b>{grade}</b></font>'
            f'  <font size="18">Bias Audit Score: <b>{audit_score}/100</b></font>'
            f'  |  Overall Severity: <b>{overall_severity.upper()}</b>',
            body,
        )
    )
    story.append(Spacer(1, 0.5 * cm))

    # Metrics table per attribute
    for attr, metrics in metrics_per_attr.items():
        story.append(Paragraph(f"Attribute: {attr}", h2))
        if "error" in metrics:
            story.append(Paragraph(f"Error: {metrics['error']}", body))
            continue

        sev = metrics.get("severity", "N/A")
        sev_color = (
            SEVERITY_HIGH if sev == "high"
            else SEVERITY_MED if sev == "medium"
            else SEVERITY_LOW
        )

        metric_data = [
            ["Metric", "Value", "Notes"],
            [
                "SPD (Statistical Parity Diff.)",
                str(metrics.get("spd", "N/A")),
                f"Severity: {sev.upper()}",
            ],
            [
                "DI (Disparate Impact)",
                str(metrics.get("di", "N/A")),
                "Legal flag (DI<0.8): " + ("YES ⚠" if metrics.get("legal_flag") else "No"),
            ],
            [
                "EOD (Equal Opportunity Diff.)",
                str(metrics.get("eod", "N/A")),
                "",
            ],
            [
                "AOD (Average Odds Diff.)",
                str(metrics.get("aod", "N/A")),
                "",
            ],
            [
                "Stat. Significant",
                "Yes" if metrics.get("statistically_significant") else "No",
                f"95% CI: [{metrics.get('bootstrapped_ci', {}).get('low_95', 'N/A')}, "
                f"{metrics.get('bootstrapped_ci', {}).get('high_95', 'N/A')}]",
            ],
        ]
        story.append(_table(metric_data, highlight_col=1, highlight_color=sev_color))
        story.append(Spacer(1, 0.3 * cm))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════
    # PAGE 3 — Mitigation Before/After
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("Mitigation Results", h1))
    story.append(HRFlowable(width="100%", thickness=2, color=BRAND_ACCENT, spaceAfter=12))

    if not mitigation:
        story.append(Paragraph("No mitigation was run for this session. Use POST /api/mitigate to generate these results.", body))
    else:
        recommendation = mitigation.get("recommendation", "none")
        story.append(Paragraph(f"Recommended technique: <b>{recommendation.capitalize()}</b>", bold_body))
        story.append(Spacer(1, 0.3 * cm))

        # Performance comparison table
        orig_perf = mitigation.get("original", {}).get("performance", {})
        rew_perf = mitigation.get("reweighing", {}).get("performance", {}) if "reweighing" in mitigation else {}
        thr_perf = mitigation.get("threshold", {}).get("performance", {}) if "threshold" in mitigation else {}

        perf_data = [
            ["Metric", "Original", "Reweighing", "Threshold Adj."],
            ["Accuracy",
             _fmt(orig_perf.get("accuracy")),
             _fmt(rew_perf.get("accuracy")),
             _fmt(thr_perf.get("accuracy"))],
            ["Precision",
             _fmt(orig_perf.get("precision")),
             _fmt(rew_perf.get("precision")),
             _fmt(thr_perf.get("precision"))],
            ["Recall",
             _fmt(orig_perf.get("recall")),
             _fmt(rew_perf.get("recall")),
             _fmt(thr_perf.get("recall"))],
            ["F1 Score",
             _fmt(orig_perf.get("f1")),
             _fmt(rew_perf.get("f1")),
             _fmt(thr_perf.get("f1"))],
            ["Audit Score",
             _fmt(mitigation.get("original", {}).get("audit_score")),
             _fmt(mitigation.get("reweighing", {}).get("metrics", {}).get("audit_score") if "reweighing" in mitigation else None),
             _fmt(mitigation.get("threshold", {}).get("metrics", {}).get("audit_score") if "threshold" in mitigation else None)],
        ]
        story.append(_table(perf_data))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════
    # PAGE 4 — Recommendations
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("Recommendations", h1))
    story.append(HRFlowable(width="100%", thickness=2, color=BRAND_ACCENT, spaceAfter=12))

    story.append(Paragraph("Immediate Actions", h2))
    recommendations = validation.get("recommendations", [])
    if recommendations:
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", body))
    else:
        story.append(Paragraph("No critical recommendations from validation.", body))

    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("General Fairness Best Practices", h2))
    best_practices = [
        "Regularly audit model predictions for demographic parity before deployment.",
        "Document sensitive attributes and their treatment in your model card.",
        "Monitor production predictions for bias drift over time.",
        "Consult legal counsel when DI falls below 0.8 — this may indicate legally actionable discrimination.",
        "Prefer reweighing over threshold adjustment when accuracy degradation must be minimised.",
        "Consider collecting more diverse training data to address historical skew.",
    ]
    for bp in best_practices:
        story.append(Paragraph(f"• {bp}", body))

    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Regulatory References", h2))
    story.append(Paragraph("• EEOC 80% / Four-Fifths Rule: DI ≥ 0.8 required for employment decisions (USA)", body))
    story.append(Paragraph("• EU AI Act Article 10: High-risk AI systems must use balanced, representative training data.", body))
    story.append(Paragraph("• ECOA / Fair Housing Act: Prohibits discriminatory lending decisions (USA)", body))

    # ── Build PDF ──────────────────────────────────────────────────────────────
    doc.build(story)
    buf.seek(0)
    return buf.read()


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _fmt(val: Any) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _table(
    data: list[list[str]],
    highlight_col: int | None = None,
    highlight_color: colors.Color | None = None,
) -> Table:
    col_widths = None  # auto
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]
    t.setStyle(TableStyle(style_cmds))
    return t
