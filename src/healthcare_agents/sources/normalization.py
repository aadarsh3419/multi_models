"""
Unicode and whitespace normalization utilities with original coordinate mapping.

Crucial for Hindi Devanagari, combining marks, CRLF line endings, and casefolded keys.
Zero provider or agent dependencies.
"""

import unicodedata


def canonicalize_label(label: str) -> str:
    """
    Produces a canonical label identity for deduplication and dictionary matching:
    NFC normalized -> casefolded -> collapsed whitespace.
    """
    if not label:
        return ""
    norm = unicodedata.normalize("NFC", label).casefold()
    return " ".join(norm.split())


def build_normalized_index_map(raw_text: str) -> tuple[str, list[int], list[int]]:
    """
    Normalizes raw text using NFC while collapsing whitespace sequences into a single space,
    returning:
      (normalized_text, orig_start_indices, orig_end_indices)
    where for normalized character i:
      orig_start_indices[i] is the starting code-point in raw_text
      orig_end_indices[i] is the ending code-point in raw_text

    Maintains coordinate integrity when combining marks (e.g. NFD accents or Hindi matras)
    are composed into NFC single code points, or when whitespace is collapsed.
    """
    # 1. Cluster characters into base + subsequent combining marks
    clusters: list[tuple[str, int, int]] = []
    i = 0
    total_len = len(raw_text)

    while i < total_len:
        start = i
        i += 1
        while i < total_len and unicodedata.category(raw_text[i]).startswith("M"):
            i += 1
        clusters.append((raw_text[start:i], start, i))

    # 2. Iterate clusters, apply whitespace collapse and NFC normalization
    norm_chars: list[str] = []
    orig_starts: list[int] = []
    orig_ends: list[int] = []
    in_whitespace = False

    for cluster_str, start_cp, end_cp in clusters:
        if cluster_str.isspace():
            if not in_whitespace:
                norm_chars.append(" ")
                orig_starts.append(start_cp)
                orig_ends.append(end_cp)
                in_whitespace = True
            else:
                # Update the ending offset of the active collapsed space
                if orig_ends:
                    orig_ends[-1] = end_cp
        else:
            in_whitespace = False
            nfc_cluster = unicodedata.normalize("NFC", cluster_str)
            for c in nfc_cluster:
                norm_chars.append(c)
                orig_starts.append(start_cp)
                orig_ends.append(end_cp)

    normalized_str = "".join(norm_chars)
    return normalized_str, orig_starts, orig_ends
