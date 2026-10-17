"""
Architecture and dependency boundary enforcement tests using Python AST.

Strictly protects architectural boundaries:
- contracts/ has no provider or agent dependencies.
- sources/ has no provider or agent dependencies.
- extraction/ and agents/ never import providers or the groq SDK.
- Only factory.py (and tests) may import providers.*.
- No forbidden frameworks (fastapi, pymongo, motor, jwt, sqlalchemy, torch, sklearn, langchain).
- Held-out evaluation manifest SHA-256 integrity test.
"""

import ast
import hashlib
from pathlib import Path


def get_imports(file_path: Path) -> list[str]:
    """Parse a Python source file and extract all imported module names."""
    with open(file_path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(file_path))

    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.append(node.module)
    return imported_modules


def get_all_py_files(directory: Path) -> list[Path]:
    """Recursively retrieves all .py files in a directory."""
    return [
        p for p in directory.rglob("*.py") if p.is_file() and not p.name.startswith("__pycache__")
    ]


def test_contracts_has_no_provider_or_agent_dependencies():
    """Verify contracts/ package contains zero provider or agent dependencies."""
    src_dir = Path(__file__).resolve().parent.parent.parent / "src" / "healthcare_agents"
    contracts_dir = src_dir / "contracts"

    assert contracts_dir.exists(), f"Directory {contracts_dir} must exist."
    files = get_all_py_files(contracts_dir)

    forbidden = {
        "groq",
        "healthcare_agents.providers",
        "healthcare_agents.agents",
        "healthcare_agents.extraction",
        "healthcare_agents.factory",
    }

    for f in files:
        for imp in get_imports(f):
            for forb in forbidden:
                assert not imp.startswith(forb), f"{f.name} illegally imports: {imp}"


def test_sources_has_no_provider_or_agent_dependencies():
    """Verify sources/ package contains zero provider or agent dependencies."""
    src_dir = Path(__file__).resolve().parent.parent.parent / "src" / "healthcare_agents"
    sources_dir = src_dir / "sources"

    assert sources_dir.exists(), f"Directory {sources_dir} must exist."
    files = get_all_py_files(sources_dir)

    forbidden = {
        "groq",
        "healthcare_agents.providers",
        "healthcare_agents.agents",
        "healthcare_agents.extraction",
    }

    for f in files:
        for imp in get_imports(f):
            for forb in forbidden:
                assert not imp.startswith(forb), f"{f.name} illegally imports: {imp}"


def test_extraction_and_agents_never_import_providers():
    """Verify extraction/ and agents/ never import providers or the groq SDK."""
    src_dir = Path(__file__).resolve().parent.parent.parent / "src" / "healthcare_agents"
    extraction_dir = src_dir / "extraction"
    agents_dir = src_dir / "agents"

    forbidden = {
        "groq",
        "healthcare_agents.providers",
    }

    for d in (extraction_dir, agents_dir):
        for f in get_all_py_files(d):
            for imp in get_imports(f):
                for forb in forbidden:
                    assert not imp.startswith(forb), f"{f} illegally imports provider module: {imp}"


def test_no_forbidden_frameworks_in_src():
    """Verify no out-of-scope dependencies are imported anywhere in src."""
    src_dir = Path(__file__).resolve().parent.parent.parent / "src"
    forbidden = {
        "fastapi",
        "starlette",
        "pymongo",
        "motor",
        "jwt",
        "jose",
        "sqlalchemy",
        "torch",
        "sklearn",
        "transformers",
        "langchain",
        "langgraph",
    }

    for f in get_all_py_files(src_dir):
        for imp in get_imports(f):
            for forb in forbidden:
                assert not imp.startswith(forb), (
                    f"{f.name} illegally imports forbidden framework: {imp}"
                )


def test_heldout_manifest_integrity():
    """Verify heldout_manifest.json matches the actual SHA-256 of heldout_cases.json."""
    eval_dir = Path(__file__).resolve().parent.parent / "evaluation"
    heldout_file = eval_dir / "heldout_cases.json"
    manifest_file = eval_dir / "heldout_manifest.json"

    assert heldout_file.exists(), f"{heldout_file} must exist."
    assert manifest_file.exists(), f"{manifest_file} must exist."

    with open(heldout_file, "rb") as f:
        actual_sha = hashlib.sha256(f.read()).hexdigest()

    import json

    with open(manifest_file, encoding="utf-8") as f:
        manifest_data = json.load(f)

    expected_sha = manifest_data.get("sha256")
    assert actual_sha == expected_sha, (
        f"Held-out dataset modified without updating manifest! Actual: {actual_sha}, Expected: {expected_sha}"
    )


def test_only_ocr_service_imports_pytesseract():
    """Verify only healthcare_agents.ocr.service imports pytesseract across src/."""
    src_dir = Path(__file__).resolve().parent.parent.parent / "src"
    all_files = get_all_py_files(src_dir)

    for f in all_files:
        imports = get_imports(f)
        if "pytesseract" in imports or any(i.startswith("pytesseract.") for i in imports):
            assert f.name == "service.py" and f.parent.name == "ocr", (
                f"Illegal import of pytesseract in {f}. Only ocr/service.py may import pytesseract."
            )


def test_only_pdf_reader_imports_pymupdf():
    """Verify only healthcare_agents.sources.pdf_reader imports pymupdf across src/."""
    src_dir = Path(__file__).resolve().parent.parent.parent / "src"
    all_files = get_all_py_files(src_dir)

    for f in all_files:
        imports = get_imports(f)
        if any(i in ("pymupdf", "fitz") or i.startswith(("pymupdf.", "fitz.")) for i in imports):
            assert f.name == "pdf_reader.py" and f.parent.name == "sources", (
                f"Illegal import of pymupdf/fitz in {f}. Only sources/pdf_reader.py may import pymupdf."
            )


def test_all_four_agents_present_in_m4():
    """Verify milestone 4 completion: all four agents (text, pdf, image, voice) exist."""
    src_dir = Path(__file__).resolve().parent.parent.parent / "src" / "healthcare_agents"
    required_files = ["text_agent.py", "pdf_agent.py", "image_agent.py", "voice_agent.py"]
    found_files = {f.name for f in get_all_py_files(src_dir)}

    for req in required_files:
        assert req in found_files, f"Expected agent module {req} to exist in Milestone 4."


def test_shared_ocr_service_used_across_pdf_and_image():
    """Verify factory allows sharing the exact same OcrService instance between PDF and Image agents."""
    from healthcare_agents.factory import build_image_agent, build_pdf_agent
    from healthcare_agents.ocr.service import FakeOcrService

    shared_ocr = FakeOcrService()
    pdf_agent = build_pdf_agent(ocr_service=shared_ocr)
    image_agent = build_image_agent(ocr_service=shared_ocr)

    assert pdf_agent.reader.ocr_service is shared_ocr
    assert image_agent.reader.ocr_service is shared_ocr
