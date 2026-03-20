"""Fix encoding issues in ingest.py."""
with open('ingest.py', 'rb') as f:
    content_bytes = f.read()

# Remove UTF-8 BOM if present
if content_bytes.startswith(b'\xef\xbb\xbf'):
    content_bytes = content_bytes[3:]

# Decode as UTF-8
content = content_bytes.decode('utf-8')

# Replace garbled multi-byte sequences (double-encoded unicode) with simple ASCII
REPLACEMENTS = [
    ('\xc3\xa2\xe2\x82\xac\xe2\x80\x9d', '--'),  # corrupted em-dash variant
    ('\xe2\x80\x94', '--'),   # em dash (correct UTF-8 sequence)
    ('\xe2\x86\x92', '->'),   # right arrow
    ('\xe2\x80\x98', "'"),    # left single quote
    ('\xe2\x80\x99', "'"),    # right single quote
    ('\xe2\x80\x9c', '"'),    # left double quote
    ('\xe2\x80\x9d', '"'),    # right double quote
]
for old, new in REPLACEMENTS:
    content = content.replace(old, new)

# Remove any remaining non-ASCII chars (replace with ?)
import re
non_ascii = re.findall(r'[^\x00-\x7F]', content)
if non_ascii:
    print(f'Removing {len(non_ascii)} non-ASCII chars: {set(non_ascii)}')
    content = re.sub(r'[^\x00-\x7F]', '?', content)
else:
    print('No non-ASCII chars remaining - clean!')

# Write clean file
with open('ingest.py', 'w', encoding='utf-8', newline='\r\n') as f:
    f.write(content)
print('ingest.py cleaned and saved!')
