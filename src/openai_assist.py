from __future__ import annotations
import json
import os
from pathlib import Path


def explain_training(metrics_path: str) -> str:
    """
    Optional helper: summarizes a training run using the OpenAI API.
    Requires OPENAI_API_KEY and `openai` installed.
    """
    try:
        from openai import OpenAI
    except Exception:
        return "OpenAI SDK not available. Run `pip install openai` to enable."

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "Set OPENAI_API_KEY to use this feature."

    p = Path(metrics_path)
    if not p.exists():
        return f"Metrics file not found: {metrics_path}"

    data = json.loads(p.read_text())
    prompt = (
        "You are a friendly ML tutor. The user trained a small MLP"
        " image classifier with manual backprop. Given the training"
        " history JSON (train/val loss and accuracy over epochs),"
        " write a short, constructive summary (<= 180 words) with"
        " 3 actionable suggestions to improve generalization."
        "\n\nHistory JSON:\n" + json.dumps(data, indent=2)
    )

    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise, encouraging ML tutor."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI request failed: {e}"

