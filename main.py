"""
CLI entry point for testing the Sustainability SME system.
Supports: preprocessing, querying, and interactive chat mode.
"""

import argparse
import sys
import json

from utils.logging_config import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Sustainability SME — AI-Powered ESG Expert",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py ingest                          # Preprocess all PDFs
  python main.py query "What are Scope 1 emissions?"  # Single query
  python main.py chat                            # Interactive chat mode
  python main.py schema                          # Show database schema
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Preprocess PDF documents")
    ingest_parser.add_argument("--pdf-dir", default=None, help="Directory containing PDFs")
    ingest_parser.add_argument("--no-vision", action="store_true", help="Skip vision model captioning")
    ingest_parser.add_argument("--no-contextual", action="store_true", help="Skip contextual enrichment")
    ingest_parser.add_argument("--no-raptor", action="store_true", help="Skip RAPTOR tree building")
    ingest_parser.add_argument("--clear", action="store_true", help="Clear existing data first")

    # Query command
    query_parser = subparsers.add_parser("query", help="Ask a single question")
    query_parser.add_argument("question", help="The question to ask")
    query_parser.add_argument("--thread", default="cli-default", help="Thread ID")

    # Chat command
    subparsers.add_parser("chat", help="Interactive chat mode")

    # Schema command
    subparsers.add_parser("schema", help="Show database schema catalog")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    setup_logging(level="INFO")

    if args.command == "ingest":
        _run_ingest(args)
    elif args.command == "query":
        _run_query(args)
    elif args.command == "chat":
        _run_chat()
    elif args.command == "schema":
        _run_schema()


def _run_ingest(args):
    from ingestion.preprocessor import preprocess_all_pdfs

    print("\n=== PREPROCESSING PIPELINE ===\n")
    stats = preprocess_all_pdfs(
        pdf_dir=args.pdf_dir,
        use_vision=not args.no_vision,
        use_contextual=not args.no_contextual,
        use_raptor=not args.no_raptor,
        clear_existing=args.clear,
    )
    print("\n=== RESULTS ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def _run_query(args):
    from graph.workflow import invoke_query

    result = invoke_query(query=args.question, thread_id=args.thread)
    _print_result(result)


def _run_chat():
    from graph.workflow import invoke_query

    thread_id = f"chat-{__import__('uuid').uuid4().hex[:8]}"
    print("\n=== SUSTAINABILITY SME — INTERACTIVE CHAT ===")
    print(f"Thread: {thread_id}")
    print("Type 'quit' to exit, 'clear' for new session\n")

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if query.lower() == "clear":
            thread_id = f"chat-{__import__('uuid').uuid4().hex[:8]}"
            print(f"New session: {thread_id}")
            continue

        result = invoke_query(query=query, thread_id=thread_id)
        _print_result(result)


def _print_result(result: dict):
    print("\n" + "=" * 60)
    print(f"ANSWER (confidence: {result.get('confidence_score', 0):.0%})")
    print("=" * 60)
    print(result.get("answer", "No answer"))

    citations = result.get("citations", [])
    if citations:
        print(f"\nSOURCES ({len(citations)}):")
        for c in citations:
            print(f"  - {c.get('document', '?')}, Page {c.get('page', '?')}")

    sql = result.get("generated_sql")
    if sql:
        print(f"\nSQL: {sql}")

    gaps = result.get("information_gaps", [])
    if gaps:
        print(f"\nINFO GAPS: {', '.join(gaps)}")

    trace = result.get("execution_trace", [])
    if trace:
        print(f"\nTRACE ({len(trace)} steps):")
        for step in trace:
            agent = step.get("agent", "?")
            dur = step.get("duration_ms", 0)
            out = step.get("output_summary", "")
            print(f"  {agent:<20} {dur:>7.0f}ms  {out}")


def _run_schema():
    from storage.schema_manager import schema_manager
    print(schema_manager.generate_catalog())


if __name__ == "__main__":
    main()
