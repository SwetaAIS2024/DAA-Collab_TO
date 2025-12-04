# -------------------------------
# Utility helpers
# -------------------------------

from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional


FOOTER_PATTERN = r"<!--output_json:(.+?)-->"

def make_seed_text(instruction: str, dataset_path: str) -> str:
    return f"Dataset: {dataset_path or 'unspecified'}\nInstructions: {instruction}"

def parse_footer(md: str) -> Dict[str, Any]:
    m = re.search(FOOTER_PATTERN, md, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}

def infer_plot_type(instr: str) -> str:
    low = instr.lower()
    if "histogram" in low: return "histogram"
    if "scatter" in low: return "scatterplot"
    if "box" in low: return "boxplot"
    if "pairplot" in low: return "pairplot"
    if "bar" in low or "chart" in low: return "bar_chart"
    # fallback
    return "histogram"

def infer_plot_columns(plot_type: str, numeric: List[str], categorical: List[str]) -> Optional[Dict[str, Any]]:
    if plot_type == "histogram" and numeric:
        return {"x_column": numeric[0]}
    if plot_type == "scatterplot" and len(numeric) >= 2:
        return {"x_column": numeric[0], "y_column": numeric[1]}
    if plot_type == "boxplot" and numeric and categorical:
        return {"x_column": numeric[0], "hue_column": categorical[0]}
    if plot_type == "bar_chart" and categorical:
        return {"x_column": categorical[0]}
    if plot_type == "pairplot" and len(numeric) >= 2:
        return {"columns_for_pairplot": numeric[:5]}
    return None










