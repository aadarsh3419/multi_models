"""Command-Line Interface (CLI) and composition root.

Assembles configuration, provider adapter, workflow engine, and TextAgent.
Enforces:
- File size pre-check before reading (never loads unbounded input).
- PHI-safe stderr diagnostics.
- Serialized AgentResult JSON to stdout.
- Structured exit codes (0: success, 1: failure, 2: invalid input, 3: partial).
"""

import argparse
import logging
import os
import sys

from pydantic import ValidationError

from healthcare_agents.config import Settings, get_settings
from healthcare_agents.contracts import (
    AgentResult,
    AgentStatus,
    TextAgentOptions,
    TextAgentRequest,
)
from healthcare_agents.groq_extractor import GroqFieldExtractor
from healthcare_agents.text_agent import TextAgent
from healthcare_agents.workflow import ExtractionWorkflow


def setup_cli_logging() -> None:
    """Configure diagnostic logging to stderr only, preventing PHI leakage to stdout."""
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def build_text_agent(settings: Settings | None = None) -> TextAgent:
    """Composition root: instantiate components and assemble TextAgent."""
    cfg = settings or get_settings()

    # Instantiate Groq adapter only if remote processing is enabled and key is present
    extractor = None
    if cfg.allow_remote_llm_processing and cfg.groq_api_key:
        extractor = GroqFieldExtractor(settings=cfg)

    workflow = ExtractionWorkflow(
        extractor=extractor,
        date_order=cfg.date_order,
        max_llm_calls_per_request=cfg.max_llm_calls_per_request,
    )

    return TextAgent(workflow=workflow)


def run_extract_text(args: argparse.Namespace) -> int:
    """Handle extract-text command execution."""
    settings = get_settings()

    # Determine input source
    text_content = ""
    if args.stdin or args.input_file == "-":
        # Read from stdin up to max_text_chars + 1 to detect oversize safely
        text_content = sys.stdin.read(settings.max_text_chars + 1)
        if len(text_content) > settings.max_text_chars:
            sys.stderr.write(
                f"Error: Stdin input exceeds maximum allowed size ({settings.max_text_chars} characters).\n"
            )
            return 2
    elif args.input_file:
        file_path = args.input_file
        if not os.path.exists(file_path):
            sys.stderr.write(f"Error: Input file '{file_path}' does not exist.\n")
            return 2

        # Check file size in bytes before reading
        try:
            file_size = os.path.getsize(file_path)
            # UTF-8 characters are at least 1 byte each; if byte size > max_chars * 4, reject early
            if file_size > settings.max_text_chars * 4:
                sys.stderr.write(f"Error: File size ({file_size} bytes) exceeds safety limits.\n")
                return 2

            with open(file_path, encoding="utf-8") as f:
                text_content = f.read(settings.max_text_chars + 1)

            if len(text_content) > settings.max_text_chars:
                sys.stderr.write(
                    f"Error: File content exceeds maximum allowed size ({settings.max_text_chars} characters).\n"
                )
                return 2
        except Exception as err:
            sys.stderr.write(f"Error reading file '{file_path}': {type(err).__name__}\n")
            return 2
    else:
        sys.stderr.write("Error: Must provide either --input-file or --stdin.\n")
        return 2

    if not text_content.strip():
        sys.stderr.write("Error: Input text is empty.\n")
        return 2

    # Parse command-line overrides for options
    options = TextAgentOptions(
        max_text_chars=settings.max_text_chars,
        language_hint=args.language_hint,
        date_order=args.date_order or settings.date_order,
        enable_remote_llm=settings.allow_remote_llm_processing,
    )

    try:
        req = TextAgentRequest(
            text=text_content,
            language_hint=args.language_hint,
            options=options,
        )
    except ValidationError as val_err:
        sys.stderr.write(f"Validation Error: {val_err}\n")
        return 2

    # Execute TextAgent
    agent = build_text_agent(settings=settings)
    result: AgentResult = agent.process(req)

    # Output AgentResult JSON to stdout
    sys.stdout.write(result.model_dump_json(indent=2) + "\n")
    sys.stdout.flush()

    # Map AgentStatus to CLI exit codes
    if result.status == AgentStatus.success:
        return 0
    elif result.status == AgentStatus.partial:
        return 3
    else:
        return 1


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint parsing subcommands and flags."""
    setup_cli_logging()

    parser = argparse.ArgumentParser(
        prog="healthcare-text",
        description="Healthcare Multi-Agent Platform - Text Extraction Foundation CLI",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    extract_parser = subparsers.add_parser(
        "extract-text",
        help="Extract structured clinical facts from text input.",
    )
    extract_parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Path to clinical text file, or '-' for stdin.",
    )
    extract_parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read input text directly from standard input.",
    )
    extract_parser.add_argument(
        "--language-hint",
        type=str,
        default=None,
        help="Optional ISO language code hint (e.g. 'en', 'hi').",
    )
    extract_parser.add_argument(
        "--date-order",
        type=str,
        choices=["UNKNOWN", "DMY", "MDY"],
        default=None,
        help="Override default date order interpretation.",
    )

    args = parser.parse_args(argv)

    if args.subcommand == "extract-text":
        exit_code = run_extract_text(args)
        sys.exit(exit_code)
    else:
        parser.print_help(sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
