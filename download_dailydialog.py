"""Download DailyDialog dataset and convert to training_data.txt format."""
import urllib.request
import ssl
import os

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

os.makedirs("dailydialog_raw", exist_ok=True)

SOURCES = [
    ("https://raw.githubusercontent.com/li128/dailydialog/master/dialogues_train.txt", "dialogues_train.txt"),
    ("https://raw.githubusercontent.com/li128/dailydialog/master/dialogues_validation.txt", "dialogues_validation.txt"),
]

downloaded = []
for url, fname in SOURCES:
    out = f"dailydialog_raw/{fname}"
    print(f"Downloading {fname}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx) as r:
            data = r.read()
        open(out, "wb").write(data)
        lines = data.decode("utf-8", errors="replace").count("\n")
        print(f"  OK: {len(data):,} bytes, {lines:,} dialogues")
        downloaded.append(out)
    except Exception as e:
        print(f"  FAIL: {e}")

if not downloaded:
    print("No files downloaded — trying fallback zip source...")
    try:
        url = "http://yanran.li/files/ijcnlp_dailydialog.zip"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx) as r:
            data = r.read()
        open("dailydialog_raw/dailydialog.zip", "wb").write(data)
        print(f"  Zip downloaded: {len(data):,} bytes")
        import zipfile
        with zipfile.ZipFile("dailydialog_raw/dailydialog.zip") as z:
            z.extractall("dailydialog_raw/")
        print("  Extracted zip")
        downloaded = [f for f in [
            "dailydialog_raw/train/dialogues_train.txt",
            "dailydialog_raw/dialogues_train.txt",
        ] if os.path.exists(f)]
    except Exception as e:
        print(f"  Fallback also failed: {e}")

if not downloaded:
    print("ERROR: Could not download DailyDialog from any source.")
    exit(1)

# Convert to training_data.txt format
# DailyDialog format: each line = one dialogue, turns separated by " __eou__ "
print("\nConverting to training_data.txt format...")
out_lines = []
total_turns = 0
total_dialogues = 0

for fpath in downloaded:
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            turns = [t.strip() for t in line.split("__eou__") if t.strip()]
            if len(turns) < 2:
                continue
            total_dialogues += 1
            # Write as alternating USER/BOT pairs
            for i, turn in enumerate(turns):
                role = "User:" if i % 2 == 0 else "Bot:"
                out_lines.append(f"{role} {turn}")
                total_turns += 1
            out_lines.append("")  # blank line = paragraph boundary for ingest.py

output_path = "dailydialog_training.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))

size = os.path.getsize(output_path)
print(f"Written: {output_path}")
print(f"  {total_dialogues:,} dialogues | {total_turns:,} turns | {size:,} bytes")
print("Done. Add dailydialog_training.txt to corpus/ or use as training_data.txt.")
