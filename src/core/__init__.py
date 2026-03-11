"""Core pipeline logic — shared between pipeline entry points."""

from __future__ import annotations

# Canonical label list for the VTC model (FEM=female adult, MAL=male adult,
# KCHI=key child, OCH=other child).
VTC_LABELS: list[str] = ["FEM", "MAL", "KCHI", "OCH"]

# Consistent colours for plots keyed by VTC label.
LABEL_COLORS: dict[str, str] = {
    "FEM": "#4C72B0",
    "MAL": "#55A868",
    "KCHI": "#C44E52",
    "OCH": "#CCB974",
}
