"""
ByUs — Report Router

GET /api/report/{session_id}
  Generates and streams a multi-page PDF audit report using ReportGenerator.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from services.reporter import ReportGenerator
from services.gemini_service import GeminiService

router = APIRouter(tags=["Report"])

reporter = ReportGenerator()
gemini = GeminiService()


@router.get("/report/{session_id}", status_code=status.HTTP_200_OK)
async def download_report(
    session_id: str,
    request: Request,
) -> StreamingResponse:
    """
    Generate and stream a PDF audit report for the given session.

    The report is generated fresh on every request from the latest
    session data (analysis + explanations + mitigation results).
    """
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
            detail="No analysis results found for this session. Run /api/analyze first.",
        )

    # ── Generate PDF ──────────────────────────────────────────────────────────
    try:
        pdf_bytes: bytes = reporter.generate(session_data=session)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF generation failed: {exc}",
        )

    # ── Stream as download ────────────────────────────────────────────────────
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"byus_audit_{session_id[:8]}_{timestamp}.pdf"

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(pdf_bytes)),
        },
    )

@router.get("/action-plan/{session_id}", status_code=status.HTTP_200_OK)
async def get_action_plan(
    session_id: str,
    request: Request,
) -> dict[str, str]:
    sessions: dict = request.app.state.sessions

    if session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )

    session = sessions[session_id]

    try:
        plan = gemini.get_action_plan(session)
        session["action_plan"] = plan
        return {"action_plan": plan}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate action plan: {exc}",
        )
