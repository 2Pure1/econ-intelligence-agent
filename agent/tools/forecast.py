"""
forecast.py
-----------
Tool 2: Call the econ-ml-platform /predict endpoints.
Returns a formatted summary the agent can cite in its response.
"""

from __future__ import annotations

import os
import httpx
from loguru import logger


ML_API = os.environ.get("ML_API_URL", "http://localhost:8000")

ENDPOINTS = {
    "gdp_growth":          f"{ML_API}/predict/gdp_growth",
    "unemployment_rate":   f"{ML_API}/predict/unemployment",
    "fed_funds_direction": f"{ML_API}/predict/fed_funds",
}


async def get_forecast(
    target:   str,
    features: dict | None = None,
    horizon:  int = 1,
) -> str:
    """
    Call the ML platform API for a live economic forecast.
    Returns a formatted string the agent can quote.
    """
    endpoint = ENDPOINTS.get(target)
    if not endpoint:
        return f"Unknown target '{target}'. Valid: {list(ENDPOINTS)}"

    payload = {"features": features or {}, "horizon": horizon}
    logger.info(f"forecast: target={target} horizon={horizon}")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(endpoint, json=payload)
            resp.raise_for_status()
            data = resp.json()

        return _format_forecast(target, data)

    except httpx.ConnectError:
        return (
            f"econ-ml-platform is not reachable at {ML_API}. "
            "Ensure the ML platform service is running."
        )
    except Exception as e:
        return f"Forecast error for {target}: {e}"


def _format_forecast(target: str, data: dict) -> str:
    lines = [f"**{target.upper()} FORECAST**"]

    if target == "gdp_growth":
        lines += [
            f"- Forecast (QoQ %): **{data['forecast_qoq_pct']:+.2f}%**",
            f"- 80% CI: [{data['confidence_lower']:.2f}%, {data['confidence_upper']:.2f}%]",
            f"- Macro regime: **{data['regime']}**",
            f"- Horizon: {data['horizon_quarters']} quarter(s) ahead",
        ]
    elif target == "unemployment_rate":
        chg = data.get("change_from_current")
        lines += [
            f"- Forecast rate: **{data['forecast_rate_pct']:.2f}%**",
            f"- Change from current: {chg:+.2f}pp" if chg is not None else "",
            f"- 80% CI: [{data['confidence_lower']:.2f}%, {data['confidence_upper']:.2f}%]",
            f"- Horizon: {data['horizon_months']} month(s) ahead",
        ]
    elif target == "fed_funds_direction":
        probs = data.get("probabilities", {})
        lines += [
            f"- Direction: **{data['direction']}**",
            f"- Implied next rate: {data.get('implied_next_rate', 'N/A')}%",
            f"- Probabilities: UP={probs.get('UP', 0):.0%} | FLAT={probs.get('FLAT', 0):.0%} | DOWN={probs.get('DOWN', 0):.0%}",
            f"- Horizon: {data['horizon_months']} month(s) ahead",
        ]

    lines += [
        f"- Model: {data.get('model_name', 'N/A')} v{data.get('model_version', '?')}",
        f"- Inference latency: {data.get('latency_ms', 0):.1f}ms",
        f"- Prediction ID: {data.get('prediction_id', 'N/A')}",
    ]

    return "\n".join(l for l in lines if l)
