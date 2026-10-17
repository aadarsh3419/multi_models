"""
Unit tests for Sources package: TextReader, normalization with index map, and TextChunker.
"""

import unicodedata
from uuid import uuid4

from healthcare_agents.sources.chunker import TextChunker
from healthcare_agents.sources.normalization import build_normalized_index_map, canonicalize_label
from healthcare_agents.sources.text_reader import TextReader

# ---------------------------------------------------------------------------
# 1. TextReader Invariant Tests (English, Devanagari, CRLF, Emoji)
# ---------------------------------------------------------------------------


def test_text_reader_english_exact_slice():
    text = (
        "Patient admitted on 2026-03-15.\n\n"
        "Vitals: BP 120/80 mmHg, HR 72 bpm.\n\n"
        "Assessment: Type 2 Diabetes Mellitus."
    )
    bundle = TextReader.read_bundle(text, uuid4())

    assert len(bundle.ordered_segments) == 3
    for seg in bundle.ordered_segments:
        loc = seg.source_location
        assert loc.kind == "text"
        assert text[loc.start : loc.end] == seg.text


def test_text_reader_crlf_windows_endings():
    text = "Line 1 in Windows format.\r\n\r\nLine 2 in Windows format.\r\n\r\nLine 3."
    bundle = TextReader.read_bundle(text, uuid4())

    assert len(bundle.ordered_segments) == 3
    for seg in bundle.ordered_segments:
        loc = seg.source_location
        assert text[loc.start : loc.end] == seg.text


def test_text_reader_hindi_unicode_and_emoji():
    text = (
        "जांच रिपोर्ट 🩺:\nरक्तचाप 120/80 mmHg दर्ज किया गया।\n\n"
        "मरीज को 3 दिन से बुखार है।\n\n"
        "दवा: Paracetamol 650 mg."
    )
    bundle = TextReader.read_bundle(text, uuid4())

    assert len(bundle.ordered_segments) == 3
    for seg in bundle.ordered_segments:
        loc = seg.source_location
        assert text[loc.start : loc.end] == seg.text


# ---------------------------------------------------------------------------
# 2. Normalization & Coordinate Index Map Tests
# ---------------------------------------------------------------------------


def test_canonicalize_label():
    assert canonicalize_label("  Blood   Pressure  ") == "blood pressure"
    assert canonicalize_label("BP") == "bp"
    assert canonicalize_label("रक्तचाप") == "रक्तचाप"


def test_build_normalized_index_map_collapsed_spaces():
    raw_text = "Heart    rate   is   72   bpm."
    norm_text, orig_starts, orig_ends = build_normalized_index_map(raw_text)

    assert norm_text == "Heart rate is 72 bpm."
    assert len(norm_text) == len(orig_starts)
    assert len(norm_text) == len(orig_ends)

    # Check that each character in norm_text corresponds to the correct position in raw_text
    sub_phrase = "72"
    norm_start = norm_text.index(sub_phrase)
    norm_end = norm_start + len(sub_phrase)

    orig_start = orig_starts[norm_start]
    orig_end = orig_ends[norm_end - 1]

    assert raw_text[orig_start:orig_end] == "72"


def test_build_normalized_index_map_decomposed_to_nfc():
    # 'é' as decomposed: 'e' + combining acute accent
    decomposed = "caf" + "e\u0301"
    norm_text, orig_starts, orig_ends = build_normalized_index_map(decomposed)

    assert norm_text == "café"
    assert unicodedata.is_normalized("NFC", norm_text)
    # Check that 'é' (last char) maps back to 'e\u0301' in decomposed string
    assert decomposed[orig_starts[-1] : orig_ends[-1]] == "e\u0301"


# ---------------------------------------------------------------------------
# 3. TextChunker Tests: Zero-Drop & Combining Mark Protection
# ---------------------------------------------------------------------------


def test_chunker_small_text_single_chunk():
    text = "Simple short clinical note."
    bundle = TextReader.read_bundle(text, uuid4())
    chunker = TextChunker(max_input_chars=1000)

    units = chunker.chunk_bundle(bundle)
    assert len(units) == 1
    assert units[0].text == text
    assert units[0].unit_id == "s0001"
    assert units[0].parent_segment_id == "s0001"
    assert units[0].parent_start == 0
    assert units[0].parent_end == len(text)


def test_chunker_large_text_multi_fragments_no_character_drops():
    sentence = "Patient exhibits signs of moderate hypertension during morning exam. "
    large_text = sentence * 120  # ~8400 chars
    bundle = TextReader.read_bundle(large_text, uuid4())

    chunker = TextChunker(max_input_chars=2000, overlap_chars=200)
    units = chunker.chunk_bundle(bundle)

    assert len(units) > 1

    # Invariant: Every character in the original segment is covered by at least one fragment
    covered = [False] * len(large_text)
    for unit in units:
        assert unit.parent_segment_id == "s0001"
        assert unit.unit_id.startswith("s0001#f")
        for i in range(unit.parent_start, unit.parent_end):
            covered[i] = True

    assert all(covered), "Chunker dropped characters during segmentation!"


def test_chunker_hindi_danda_sentence_boundary():
    seg1 = "मरीज को तेज बुखार और सिरदर्द है। " * 80  # ~2700 chars
    bundle = TextReader.read_bundle(seg1, uuid4())
    chunker = TextChunker(max_input_chars=1000, overlap_chars=100)

    units = chunker.chunk_bundle(bundle)
    assert len(units) > 1

    # First fragment should end on a danda boundary
    first_frag_text = units[0].text.strip()
    assert first_frag_text.endswith("।")


def test_chunker_combining_mark_protection():
    # Ensure chunker doesn't slice between a base char and a combining mark (Hindi matra)
    # Devanagari 'क' + virama + 'ष' = 'क्ष'
    base = "क"
    matra = "\u094d"  # virama combining mark
    text = (base + matra + "ष ") * 500
    bundle = TextReader.read_bundle(text, uuid4())

    chunker = TextChunker(max_input_chars=500, overlap_chars=50)
    units = chunker.chunk_bundle(bundle)

    for unit in units:
        # Unit text should not start with an isolated combining mark
        assert not unicodedata.category(unit.text[0]).startswith("M")
