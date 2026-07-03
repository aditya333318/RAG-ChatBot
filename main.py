"""
CLI entry point for the RAG Chatbot.
Run: python main.py --pdf path/to/file.pdf
"""

import argparse
import sys
from pathlib import Path

from src.chatbot import RAGChatbot


def main():
    parser = argparse.ArgumentParser(description="RAG Chatbot CLI")
    parser.add_argument("--pdf", type=str, help="Path to a PDF file to ingest")
    parser.add_argument("--dir", type=str, help="Path to a directory of PDFs")
    parser.add_argument("--query", type=str, help="Single query mode")
    parser.add_argument("--index", type=str, help="Load a saved FAISS index")
    args = parser.parse_args()

    print("🤖 RAG Chatbot")
    print("=" * 50)

    bot = RAGChatbot()

    # Load or ingest
    if args.index:
        print(f"Loading index from {args.index}...")
        bot.load_index(args.index)
    elif args.pdf:
        print(f"Ingesting {args.pdf}...")
        result = bot.ingest_pdf(args.pdf)
        print(f"✅ Done. {result['chunk_stats']['total_chunks']} chunks indexed.")
    elif args.dir:
        print(f"Ingesting PDFs from {args.dir}...")
        result = bot.ingest_directory(args.dir)
        print(f"✅ Done. {result['total_chunks']} chunks indexed.")
    else:
        print("⚠️ Please provide --pdf, --dir, or --index")
        sys.exit(1)

    # Single query mode
    if args.query:
        print(f"\nQ: {args.query}\n")
        result = bot.ask(args.query)
        print(f"A: {result['answer']}\n")
        print("Sources:")
        for src in result["sources"]:
            print(f"  - {src['source']}, page {src['page']}")
        return

    # Interactive mode
    print("\nEnter your questions (type 'quit' to exit, 'save' to save index):\n")
    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() in ("quit", "exit", "q"):
            break
        if question.lower() == "save":
            bot.save_index()
            print("💾 Index saved.\n")
            continue
        if not question:
            continue

        result = bot.ask(question)
        print(f"A: {result['answer']}\n")


if __name__ == "__main__":
    main()
