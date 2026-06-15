"""Resolve mouse gene symbols to Ensembl IDs for ISP and pipeline configs."""
from __future__ import annotations

import pickle
from functools import lru_cache

from .in_silico_perturber_stats import GENE_NAME_ID_DICTIONARY_FILE


@lru_cache(maxsize=1)
def _load_gene_name_id_dict() -> dict[str, str]:
    if not GENE_NAME_ID_DICTIONARY_FILE.is_file():
        return {}
    with open(GENE_NAME_ID_DICTIONARY_FILE, "rb") as f:
        return pickle.load(f)


def _looks_like_ensembl_id(gene: str) -> bool:
    return str(gene).startswith("ENS")


def resolve_gene_identifier(raw: str) -> str:
    """Return an Ensembl ID for a symbol or pass through an existing Ensembl ID."""
    s = str(raw).strip()
    if not s:
        raise ValueError("Empty gene identifier in genes_to_perturb.")
    if _looks_like_ensembl_id(s):
        return s

    name_id = _load_gene_name_id_dict()
    if not name_id:
        raise ValueError(
            f"Gene {s!r} looks like a symbol but symbol dictionary is unavailable at "
            f"{GENE_NAME_ID_DICTIONARY_FILE}. Use an Ensembl ID (e.g. ENSMUSG00000057530)."
        )
    if s in name_id:
        return name_id[s]

    raise ValueError(
        f"Gene symbol {s!r} not found in symbol dictionary. "
        "Use an Ensembl ID (e.g. ENSMUSG00000057530) or a known mouse gene symbol (e.g. Ece1, Igfbp2)."
    )


def resolve_genes_to_perturb(genes: list) -> list[str]:
    if not genes:
        return []
    return [resolve_gene_identifier(g) for g in genes]
