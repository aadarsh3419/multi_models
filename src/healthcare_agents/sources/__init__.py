"""
Sources package for document reading, normalization, and chunking.

Zero provider or agent dependencies.
"""

from healthcare_agents.sources.chunker import TextChunker
from healthcare_agents.sources.normalization import build_normalized_index_map, canonicalize_label
from healthcare_agents.sources.text_reader import TextReader

__all__ = [
    "TextChunker",
    "TextReader",
    "build_normalized_index_map",
    "canonicalize_label",
]
