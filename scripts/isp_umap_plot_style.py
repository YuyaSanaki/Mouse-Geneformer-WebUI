#!/usr/bin/env python3
"""Shared color / order helpers for ISP UMAP downstream figures.

Palette rules mirror ``plot_isp_umap_joint_overlays.py`` so L2 box/bar plots
stay aligned with ``umap_joint_l2_cluster_celltype.png``:

- ``cluster``: ``tab10`` indexed by sorted unique cluster ids (C0, C1, …)
- other group columns (coarse_type, pred_cell_type, sample_id, …): ``husl``
  indexed by value_counts order (same as the bottom-right UMAP panel)
"""

from __future__ import annotations

from typing import Any, Hashable, Iterable, Sequence

import pandas as pd
import seaborn as sns

# Preferred x-axis order for known coarse programs (does not affect colors).
DEFAULT_COARSE_ORDER = [
    "Vascular_SMC-like",
    "SMC-intermediate",
    "Microglia-like",
    "Other/Ambiguous",
]


def _sorted_cluster_ids(values: Iterable[Any]) -> list[Any]:
    """Match ``sorted(pd.unique(...))`` used in the joint-overlay cluster panel."""
    return list(sorted(pd.unique(pd.Series(list(values)))))


def palette_for_groups(labels: Sequence[Hashable], group_col: str) -> dict[str, Any]:
    """Return ``{str(label): color}`` using the same scheme as joint UMAP overlays."""
    series = pd.Series(list(labels))
    if group_col == "cluster":
        ids = _sorted_cluster_ids(series)
        colors = sns.color_palette("tab10", n_colors=max(len(ids), 1))
        return {str(c): colors[i % len(colors)] for i, c in enumerate(ids)}

    # Cell-type / sample_id panel: husl in value_counts order
    types = list(series.value_counts().index)
    colors = sns.color_palette("husl", n_colors=max(len(types), 1))
    return {str(t): colors[i % len(colors)] for i, t in enumerate(types)}


def order_for_groups(labels: Sequence[Hashable], group_col: str) -> list[str]:
    """X-axis order for L2 plots (display only; colors come from ``palette_for_groups``)."""
    series = pd.Series(list(labels)).astype(str)
    counts = series.value_counts()
    if group_col == "cluster":
        # Keep frequency order for the bar chart, but colors stay ID-keyed.
        return [str(x) for x in counts.index]

    order = [c for c in DEFAULT_COARSE_ORDER if c in set(series)]
    order += [c for c in counts.index if c not in order]
    return order
