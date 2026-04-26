"""
FairLens — Gemini Chat Router

POST /api/gemini-chat
  Conversational Bias Copilot powered by Google Gemini 2.0 Flash.
  Injects session analysis context into every request as a system prompt.
"""

from __future__ import annotations

import os
from typing import Any

import google.generativeai as genai
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

router = APIRouter(tags=["Gemini Copilot"])


class ChatMessage(BaseModel):
    role: str   # "user" or "model"
    content: str


class GeminiChatRequest(BaseModel):
    session_id: str
    message: str
    conversation_history: list[ChatMessage] = []


@router.post("/gemini-chat", status_code=status.HTTP_200_OK)
async def gemini_chat(
    body: GeminiChatRequest,
    request: Request,
) -> dict[str, Any]:
    """
    Send a message to the Gemini Bias Copilot.

    The copilot is context-aware: it receives a system prompt containing
    the full analysis results from the session so it can answer questions
    like "Why is gender biased?" or "Which mitigation should I use?"
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GEMINI_API_KEY is not configured on the server.",
        )

    sessions: dict = request.app.state.sessions
    if body.session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{body.session_id}' not found.",
        )

    session = sessions[body.session_id]

    # ── Build system context ──────────────────────────────────────────────────
    system_prompt = _build_system_prompt(session)

    # ── Configure Gemini ──────────────────────────────────────────────────────
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        system_instruction=system_prompt,
    )

    # ── Build conversation history for multi-turn ─────────────────────────────
    history = [
        {"role": msg.role, "parts": [msg.content]}
        for msg in body.conversation_history
    ]

    chat = model.start_chat(history=history)

    # ── Send message ──────────────────────────────────────────────────────────
    try:
        response = chat.send_message(body.message)
        reply = response.text
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gemini API error: {exc}",
        )

    return {"reply": reply, "session_id": body.session_id}


# ── System prompt builder ─────────────────────────────────────────────────────

def _build_system_prompt(session: dict) -> str:
    lines = [
        "You are the FairLens Bias Copilot — an expert AI assistant specialising in "
        "algorithmic fairness, bias detection, and ML ethics.",
        "",
        "You have access to the full analysis results for the user's current dataset session.",
        "Use this context to answer questions accurately and specifically.",
        "Always explain technical terms (SPD, DI, EOD, AOD, Reweighing) in plain English.",
        "Be concise but thorough. Use bullet points where helpful.",
        "",
        "=== SESSION ANALYSIS CONTEXT ===",
    ]

    target_col = session.get("target_col", "N/A")
    sensitive_attrs = session.get("sensitive_attrs", [])
    filename = session.get("filename", "unknown")
    row_count = session.get("row_count", "unknown")

    lines += [
        f"Dataset file: {filename}",
        f"Rows: {row_count}",
        f"Target column: {target_col}",
        f"Sensitive attributes: {', '.join(sensitive_attrs) if sensitive_attrs else 'None selected'}",
    ]

    bias_results = session.get("bias_results")
    if bias_results:
        lines += [
            f"Bias Audit Score: {bias_results.get('audit_score', 'N/A')}/100 (Grade: {bias_results.get('grade', 'N/A')})",
            f"Overall Severity: {bias_results.get('overall_severity', 'N/A')}",
            "",
            "Metrics per sensitive attribute:",
        ]
        for attr, metrics in bias_results.get("metrics_per_attr", {}).items():
            if "error" in metrics:
                lines.append(f"  {attr}: ERROR — {metrics['error']}")
            else:
                lines.append(
                    f"  {attr}: SPD={metrics.get('spd', 'N/A')}, "
                    f"DI={metrics.get('di', 'N/A')}, "
                    f"EOD={metrics.get('eod', 'N/A')}, "
                    f"AOD={metrics.get('aod', 'N/A')}, "
                    f"Severity={metrics.get('severity', 'N/A').upper()}, "
                    f"Legal flag={metrics.get('legal_flag', False)}"
                )

    validation = session.get("validation")
    if validation:
        lines += [
            "",
            f"Validation engine: {validation.get('engine', 'N/A')}",
            f"Target type: {validation.get('target_type', 'N/A')}",
        ]
        warnings = validation.get("warnings", [])
        if warnings:
            lines.append("Validation warnings:")
            for w in warnings:
                lines.append(f"  - {w}")

    mitigation = session.get("mitigation_results")
    if mitigation:
        rec = mitigation.get("recommendation", "none")
        lines += [
            "",
            f"Mitigation run: Yes. Recommended technique: {rec}.",
        ]

    lines += [
        "=== END OF CONTEXT ===",
        "",
        "Answer the user's question based on the above context.",
    ]

    return "\n".join(lines)
