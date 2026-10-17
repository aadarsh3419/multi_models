"""
CLI interface for Healthcare Text Foundation.

Supports text and file inputs with strict size pre-checking and exit codes:
  0: Success (all stages executed successfully)
  2: Partial (stages skipped or units unprocessed, e.g. remote LLM disabled)
  1: Failed (invalid input, nonexistent file, or processing error)
"""

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

from healthcare_agents.agents.image_agent import ImageAgent
from healthcare_agents.agents.pdf_agent import PDFAgent
from healthcare_agents.agents.text_agent import TextAgent
from healthcare_agents.agents.voice_agent import VoiceAgent
from healthcare_agents.config import Settings
from healthcare_agents.contracts.requests import (
    ImageAgentOptions,
    ImageAgentRequest,
    PDFAgentOptions,
    PDFAgentRequest,
    TextAgentOptions,
    TextAgentRequest,
    VoiceAgentOptions,
    VoiceAgentRequest,
)
from healthcare_agents.factory import (
    build_image_agent,
    build_pdf_agent,
    build_text_agent,
    build_voice_agent,
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="healthcare-text",
        description="Healthcare Multi-Agent AI Platform — Extraction CLI (Milestones 1 & 2)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 'text' subcommand
    text_parser = subparsers.add_parser("text", help="Extract clinical facts from text or file")
    text_parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Raw clinical narrative passed as inline string.",
    )
    text_parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to UTF-8 text file containing clinical narrative.",
    )
    text_parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read clinical narrative from standard input.",
    )
    text_parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="Optional ISO-639-1 language hint (e.g. 'en', 'hi').",
    )
    text_parser.add_argument(
        "--date-order",
        type=str,
        choices=["UNKNOWN", "DMY", "MDY"],
        default=None,
        help="Optional date order rule override ('UNKNOWN', 'DMY', 'MDY').",
    )
    text_parser.add_argument(
        "--debug-include-source-text",
        action="store_true",
        help="Include source text in result JSON (local debugging only).",
    )

    # 'pdf' subcommand
    pdf_parser = subparsers.add_parser("pdf", help="Extract clinical facts from PDF document")
    pdf_parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to PDF document (native, scanned, or hybrid).",
    )
    pdf_parser.add_argument(
        "--ocr-mode",
        type=str,
        choices=["auto", "never", "always"],
        default="auto",
        help="OCR mode policy ('auto', 'never', 'always').",
    )
    pdf_parser.add_argument(
        "--ocr-languages",
        type=str,
        default="eng",
        help="OCR language codes joined by '+' (e.g. 'eng', 'eng+hin').",
    )
    pdf_parser.add_argument(
        "--date-order",
        type=str,
        choices=["UNKNOWN", "DMY", "MDY"],
        default=None,
        help="Optional date order rule override ('UNKNOWN', 'DMY', 'MDY').",
    )
    pdf_parser.add_argument(
        "--debug-include-source-text",
        action="store_true",
        help="Include extracted text in result JSON (local debugging only).",
    )

    # 'image' subcommand
    image_parser = subparsers.add_parser("image", help="Extract clinical facts from document image")
    image_parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to image document (scanned report, prescription photo, lab slip).",
    )
    image_parser.add_argument(
        "--ocr-languages",
        type=str,
        default="eng",
        help="OCR language codes joined by '+' (e.g. 'eng', 'eng+hin').",
    )
    image_parser.add_argument(
        "--date-order",
        type=str,
        choices=["UNKNOWN", "DMY", "MDY"],
        default=None,
        help="Optional date order rule override ('UNKNOWN', 'DMY', 'MDY').",
    )
    image_parser.add_argument(
        "--no-upscale",
        action="store_true",
        help="Disable automatic Lanczos upscaling for smaller images.",
    )
    image_parser.add_argument(
        "--debug-include-source-text",
        action="store_true",
        help="Include extracted text in result JSON (local debugging only).",
    )

    # 'voice' subcommand
    voice_parser = subparsers.add_parser(
        "voice", help="Extract clinical facts from audio recording"
    )
    voice_parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to audio recording (wav, mp3, ogg, flac, m4a, webm).",
    )
    voice_parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="Optional ISO-639-1 language hint (e.g. 'en', 'hi').",
    )
    voice_parser.add_argument(
        "--date-order",
        type=str,
        choices=["UNKNOWN", "DMY", "MDY"],
        default=None,
        help="Optional date order rule override ('UNKNOWN', 'DMY', 'MDY').",
    )
    voice_parser.add_argument(
        "--debug-include-source-text",
        action="store_true",
        help="Include transcribed text in result JSON (local debugging only).",
    )

    return parser


def run_text_subcommand(
    args: argparse.Namespace,
    settings: Settings,
    agent_factory: Callable[[Settings], TextAgent],
) -> int:
    text_content: str = ""

    if args.text is not None:
        text_content = args.text

    elif args.file is not None:
        file_path = Path(args.file)
        if not file_path.exists() or not file_path.is_file():
            sys.stderr.write(
                f"Error: Specified file does not exist or is not a file: {file_path}\n"
            )
            return 1

        # Pre-check file size before reading into memory
        file_size = os.path.getsize(file_path)
        if file_size > settings.max_text_chars:
            sys.stderr.write(
                f"Error: File size ({file_size} bytes) exceeds maximum configured limit ({settings.max_text_chars} chars).\n"
            )
            return 1

        try:
            with open(file_path, encoding="utf-8") as f:
                text_content = f.read()
        except Exception as err:
            sys.stderr.write(f"Error reading file {file_path}: {err}\n")
            return 1

    elif args.stdin or not sys.stdin.isatty():
        try:
            text_content = sys.stdin.read()
        except Exception as err:
            sys.stderr.write(f"Error reading from stdin: {err}\n")
            return 1

    else:
        sys.stderr.write("Error: Must provide clinical text via --text, --file, or --stdin.\n")
        return 1

    if not text_content or not text_content.strip():
        sys.stderr.write("Error: Clinical narrative is empty or contains only whitespace.\n")
        return 1

    agent = agent_factory(settings)

    options = TextAgentOptions(include_source_text_in_result=args.debug_include_source_text)
    request = TextAgentRequest(
        text=text_content,
        language_hint=args.lang,
        date_order_hint=args.date_order,
        options=options,
    )

    result = agent.process(request)

    # Output JSON to stdout
    print(result.model_dump_json(indent=2))

    # Diagnostics to stderr
    sys.stderr.write(
        f"TextAgent Execution: status={result.status}, candidates={len(result.candidates)}, "
        f"duration={result.duration_ms:.2f}ms\n"
    )

    if result.status == "success":
        return 0
    elif result.status == "partial":
        return 2
    else:
        return 1


def run_pdf_subcommand(
    args: argparse.Namespace,
    settings: Settings,
    agent_factory: Callable[[Settings], PDFAgent],
) -> int:
    file_path = Path(args.file)
    if not file_path.exists() or not file_path.is_file():
        sys.stderr.write(
            f"Error: Specified PDF file does not exist or is not a file: {file_path}\n"
        )
        return 1

    # Pre-check file size before reading into memory
    file_size = os.path.getsize(file_path)
    if file_size > settings.max_pdf_bytes:
        sys.stderr.write(
            f"Error: PDF file size ({file_size} bytes) exceeds maximum configured limit ({settings.max_pdf_bytes} bytes).\n"
        )
        return 1

    try:
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
    except Exception as err:
        sys.stderr.write(f"Error reading PDF file {file_path}: {err}\n")
        return 1

    if not pdf_bytes:
        sys.stderr.write("Error: PDF file is empty (0 bytes).\n")
        return 1

    try:
        options = PDFAgentOptions(
            ocr_mode=args.ocr_mode,
            ocr_languages=args.ocr_languages,
            date_order_hint=args.date_order,
            include_source_text_in_result=args.debug_include_source_text,
        )
    except Exception as opt_err:
        sys.stderr.write(f"Error in PDF options: {opt_err}\n")
        return 1

    agent = agent_factory(settings)
    request = PDFAgentRequest(
        content=pdf_bytes,
        source_name=file_path.name,
        options=options,
    )

    result = agent.process(request)

    # Output JSON to stdout
    print(result.model_dump_json(indent=2))

    # Diagnostics to stderr
    sys.stderr.write(
        f"PDFAgent Execution: status={result.status}, candidates={len(result.candidates)}, "
        f"duration={result.duration_ms:.2f}ms\n"
    )

    if result.status == "success":
        return 0
    elif result.status == "partial":
        return 2
    else:
        return 1


def run_image_subcommand(
    args: argparse.Namespace,
    settings: Settings,
    agent_factory: Callable[[Settings], ImageAgent],
) -> int:
    file_path = Path(args.file)
    if not file_path.exists() or not file_path.is_file():
        sys.stderr.write(
            f"Error: Specified image file does not exist or is not a file: {file_path}\n"
        )
        return 1

    # Pre-check file size before reading into memory
    file_size = os.path.getsize(file_path)
    if file_size > settings.max_image_bytes:
        sys.stderr.write(
            f"Error: Image file size ({file_size} bytes) exceeds maximum configured limit ({settings.max_image_bytes} bytes).\n"
        )
        return 1

    try:
        with open(file_path, "rb") as f:
            image_bytes = f.read()
    except Exception as err:
        sys.stderr.write(f"Error reading image file {file_path}: {err}\n")
        return 1

    if not image_bytes:
        sys.stderr.write("Error: Image file is empty (0 bytes).\n")
        return 1

    try:
        options = ImageAgentOptions(
            ocr_languages=args.ocr_languages,
            date_order_hint=args.date_order,
            include_source_text_in_result=args.debug_include_source_text,
            upscale_enabled=not getattr(args, "no_upscale", False),
        )
    except Exception as opt_err:
        sys.stderr.write(f"Error in image options: {opt_err}\n")
        return 1

    agent = agent_factory(settings)
    request = ImageAgentRequest(
        content=image_bytes,
        source_name=file_path.name,
        options=options,
    )

    result = agent.process(request)

    # Output JSON to stdout
    print(result.model_dump_json(indent=2))

    # Diagnostics to stderr
    sys.stderr.write(
        f"ImageAgent Execution: status={result.status}, candidates={len(result.candidates)}, "
        f"duration={result.duration_ms:.2f}ms\n"
    )

    if result.status == "success":
        return 0
    elif result.status == "partial":
        return 2
    else:
        return 1


def run_voice_subcommand(
    args: argparse.Namespace,
    settings: Settings,
    agent_factory: Callable[[Settings], VoiceAgent],
) -> int:
    file_path = Path(args.file)
    if not file_path.exists() or not file_path.is_file():
        sys.stderr.write(
            f"Error: Specified audio file does not exist or is not a file: {file_path}\n"
        )
        return 1

    # Pre-check file size before reading into memory
    file_size = os.path.getsize(file_path)
    if file_size > settings.max_audio_bytes:
        sys.stderr.write(
            f"Error: Audio file size ({file_size} bytes) exceeds maximum configured limit ({settings.max_audio_bytes} bytes).\n"
        )
        return 1

    try:
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
    except Exception as err:
        sys.stderr.write(f"Error reading audio file {file_path}: {err}\n")
        return 1

    if not audio_bytes:
        sys.stderr.write("Error: Audio file is empty (0 bytes).\n")
        return 1

    try:
        options = VoiceAgentOptions(
            date_order_hint=args.date_order,
            include_source_text_in_result=args.debug_include_source_text,
        )
    except Exception as opt_err:
        sys.stderr.write(f"Error in voice options: {opt_err}\n")
        return 1

    agent = agent_factory(settings)
    request = VoiceAgentRequest(
        content=audio_bytes,
        source_name=file_path.name,
        language_hint=args.lang,
        options=options,
    )

    result = agent.process(request)

    # Output JSON to stdout
    print(result.model_dump_json(indent=2))

    # Diagnostics to stderr
    sys.stderr.write(
        f"VoiceAgent Execution: status={result.status}, candidates={len(result.candidates)}, "
        f"duration={result.duration_ms:.2f}ms\n"
    )

    if result.status == "success":
        return 0
    elif result.status == "partial":
        return 2
    else:
        return 1


def main(
    argv: list[str] | None = None,
    agent_factory: Callable[[Settings], TextAgent] = build_text_agent,
    pdf_agent_factory: Callable[[Settings], PDFAgent] = build_pdf_agent,
    image_agent_factory: Callable[[Settings], ImageAgent] = build_image_agent,
    voice_agent_factory: Callable[[Settings], VoiceAgent] = build_voice_agent,
) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        # Default to 'text' if invoked with direct options or stdin
        if any(x in (argv or sys.argv[1:]) for x in ["--text", "--file", "--stdin"]):
            argv_adjusted = ["text"] + (argv if argv is not None else sys.argv[1:])
            args = parser.parse_args(argv_adjusted)
        else:
            parser.print_help(sys.stderr)
            return 1

    settings = Settings()

    if args.command == "text":
        return run_text_subcommand(args, settings, agent_factory)
    elif args.command == "pdf":
        return run_pdf_subcommand(args, settings, pdf_agent_factory)
    elif args.command == "image":
        return run_image_subcommand(args, settings, image_agent_factory)
    elif args.command == "voice":
        return run_voice_subcommand(args, settings, voice_agent_factory)

    return 1


if __name__ == "__main__":
    sys.exit(main())
