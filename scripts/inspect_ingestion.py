"""Manually inspect the ingestion output for one Markdown file."""

from pathlib import Path
from pprint import pprint

from recipe_rag.ingestion.loader import load_markdown_file


def main() -> None:
    input_value = input("Markdown file path: ").strip()
    input_path = Path(input_value)

    print("\nINPUT:")
    print(input_path)

    output = load_markdown_file(input_path)

    print("\nOUTPUT:")
    pprint(output)


if __name__ == "__main__":
    main()
