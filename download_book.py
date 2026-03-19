# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

"""
Download books from Project Gutenberg into the corpus/ folder.
Run: uv run download_book.py

Books are saved as corpus/{id}_{slug}.txt
Re-run ingest.py after adding books to retrain.
"""

import urllib.request
import re
import os
import sys

# Curated selection of conversation-rich, public domain books
BOOKS = {
    "11":   "Alice's Adventures in Wonderland — Lewis Carroll",
    "1661": "The Adventures of Sherlock Holmes — Arthur Conan Doyle",
    "84":   "Frankenstein — Mary Shelley",
    "1342": "Pride and Prejudice — Jane Austen",
    "345":  "Dracula — Bram Stoker",
    "35":   "The Time Machine — H.G. Wells",
    "2701": "Moby Dick — Herman Melville",
    "76":   "The Adventures of Huckleberry Finn — Mark Twain",
    "98":   "A Tale of Two Cities — Charles Dickens",
    "844":  "The Importance of Being Earnest — Oscar Wilde (all dialogue)",
}

GUTENBERG_URLS = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://gutenberg.org/files/{id}/{id}-0.txt",
    "https://gutenberg.org/files/{id}/{id}.txt",
]


def strip_gutenberg_boilerplate(text):
    """Remove Project Gutenberg header and footer — keep only the actual book."""
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "*END*THE SMALL PRINT",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]

    start_pos = 0
    for marker in start_markers:
        idx = text.upper().find(marker.upper())
        if idx != -1:
            # Move past the marker line
            start_pos = text.find("\n", idx) + 1
            break

    end_pos = len(text)
    for marker in end_markers:
        idx = text.upper().find(marker.upper(), start_pos)
        if idx != -1:
            end_pos = idx
            break

    return text[start_pos:end_pos].strip()


def clean_text(text):
    """Normalize text for training: lowercase, collapse whitespace."""
    text = text.lower()
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lines that are clearly chapter headers (ALL CAPS or Roman numerals only)
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that are only Roman numerals, numbers, or very short caps headers
        if re.match(r"^[ivxlcdmIVXLCDM\d\s\.]+$", stripped) and len(stripped) < 20:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def download_book(book_id):
    print(f"\nDownloading book {book_id}: {BOOKS.get(book_id, '(unknown title)')}")

    text = None
    for url_template in GUTENBERG_URLS:
        url = url_template.format(id=book_id)
        try:
            print(f"  Trying {url}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            if len(raw) > 10000:  # sanity check — real books are large
                text = raw
                print(f"  Downloaded {len(raw):,} bytes.")
                break
        except Exception as e:
            print(f"  Failed: {e}")

    if text is None:
        print(f"  Could not download book {book_id}. Try manually saving it to corpus/.")
        return False

    text = strip_gutenberg_boilerplate(text)
    text = clean_text(text)

    word_count = len(re.findall(r"\b\w+\b", text))
    print(f"  {word_count:,} words after cleaning.")

    os.makedirs("corpus", exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", BOOKS.get(book_id, book_id).lower().split("—")[0].strip())
    slug = slug.strip("_")[:40]
    outpath = os.path.join("corpus", f"{book_id}_{slug}.txt")

    with open(outpath, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"  Saved to {outpath}")
    return True


def main():
    print("Project Gutenberg Book Downloader")
    print("=" * 40)
    print("\nAvailable books:")
    for book_id, title in BOOKS.items():
        fpath = None
        for f in (os.listdir("corpus") if os.path.exists("corpus") else []):
            if f.startswith(f"{book_id}_"):
                fpath = f
        status = f"[downloaded: {fpath}]" if fpath else ""
        print(f"  {book_id:>6} — {title} {status}")

    print("\nWhich books to download? Enter IDs separated by spaces.")
    print("  e.g.: 11 1661        (Alice + Sherlock Holmes)")
    print("  or:   all            (download everything — ~500k words total)")
    print("  or:   q              (quit)")

    choice = input("\n> ").strip().lower()

    if choice in ("q", "quit", ""):
        return

    if choice == "all":
        book_ids = list(BOOKS.keys())
    else:
        book_ids = choice.split()
        invalid = [b for b in book_ids if b not in BOOKS]
        if invalid:
            print(f"Unknown book IDs: {invalid}")
            print(f"Valid IDs: {list(BOOKS.keys())}")
            return

    succeeded = []
    for book_id in book_ids:
        if download_book(book_id):
            succeeded.append(book_id)

    if succeeded:
        print(f"\nDownloaded {len(succeeded)} book(s).")
        print("Run ingest.py to train on the new corpus.")
    else:
        print("\nNo books were downloaded successfully.")


if __name__ == "__main__":
    main()
