import argparse

from .processor import PDFProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an academic PDF into section Markdown files")
    parser.add_argument("paper_id")
    parser.add_argument("pdf_url")
    parser.add_argument("--cache-dir", default="cache/papers")
    args = parser.parse_args()
    metadata = PDFProcessor(args.cache_dir).process_paper(args.paper_id, args.pdf_url)
    print(f"{metadata['paper_title']} -> {metadata['folder']}")


if __name__ == "__main__":
    main()
