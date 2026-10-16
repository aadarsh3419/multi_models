"""Architecture enforcement tests using Python AST.

Protects critical dependency boundaries defined in Section 5:
- contracts.py has no provider, workflow, or agent dependencies.
- workflow.py has no direct Groq SDK imports.
- text_agent.py has no direct Groq SDK imports.
"""

import ast
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


def test_contracts_has_no_provider_or_agent_dependencies():
    """Verify contracts.py contains no Groq SDK, agent, or workflow dependencies."""
    src_dir = Path(__file__).resolve().parent.parent.parent / "src" / "healthcare_agents"
    contracts_file = src_dir / "contracts.py"

    assert contracts_file.exists(), f"File {contracts_file} must exist."
    imports = get_imports(contracts_file)

    forbidden = {
        "groq",
        "healthcare_agents.workflow",
        "healthcare_agents.text_agent",
        "healthcare_agents.groq_extractor",
    }
    for imp in imports:
        for f in forbidden:
            assert not imp.startswith(f), f"contracts.py illegally imports: {imp}"


def test_workflow_has_no_direct_groq_import():
    """Verify workflow.py does not directly import Groq SDK."""
    src_dir = Path(__file__).resolve().parent.parent.parent / "src" / "healthcare_agents"
    workflow_file = src_dir / "workflow.py"

    assert workflow_file.exists(), f"File {workflow_file} must exist."
    imports = get_imports(workflow_file)

    for imp in imports:
        assert not imp.startswith("groq"), f"workflow.py illegally imports Groq SDK: {imp}"


def test_text_agent_has_no_direct_groq_import():
    """Verify text_agent.py does not directly import Groq SDK."""
    src_dir = Path(__file__).resolve().parent.parent.parent / "src" / "healthcare_agents"
    agent_file = src_dir / "text_agent.py"

    assert agent_file.exists(), f"File {agent_file} must exist."
    imports = get_imports(agent_file)

    for imp in imports:
        assert not imp.startswith("groq"), f"text_agent.py illegally imports Groq SDK: {imp}"
