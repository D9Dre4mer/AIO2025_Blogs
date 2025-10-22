#!/usr/bin/env python3
import os, re, time, pathlib, requests
import sys

# Set UTF-8 encoding for stdout
sys.stdout.reconfigure(encoding='utf-8')

MD_FILE = "genetic-algorithm-in-a-nutshell.md"
DRY_RUN = os.getenv("UPLOAD_DRY_RUN") is not None
IMGBB_KEY = os.getenv("IMGBB_KEY")

if not IMGBB_KEY:
    raise SystemExit("IMGBB_KEY not found. Set it as environment variable.")

def backoff(retry):
    time.sleep([0.5, 1.0, 2.0][min(retry, 2)])

def upload_imgbb(path):
    with open(path, "rb") as f:
        for r in range(3):
            resp = requests.post(
                "https://api.imgbb.com/1/upload",
                data={"key": IMGBB_KEY},
                files={"image": f}
            )
            if resp.ok:
                return resp.json()["data"]["url"]
            backoff(r)
    raise RuntimeError(f"Upload failed after retries: {path}")

pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
md_path = pathlib.Path(MD_FILE)
if not md_path.exists():
    raise SystemExit(f"File {MD_FILE} not found.")

content = md_path.read_text(encoding="utf-8")
replacements = []

def is_local(p):
    return not (p.startswith("http://") or p.startswith("https://") or p.startswith("data:"))

for m in pattern.finditer(content):
    alt, path = m.group(1), m.group(2).strip()
    if not is_local(path): continue
    local = pathlib.Path(path)
    if not local.exists():
        print(f"[skip] missing: {path}")
        continue
    if DRY_RUN:
        print(f"[dry-run] would upload: {path}")
        continue
    try:
        url = upload_imgbb(local)
        replacements.append((m.span(), f"![{alt}]({url})"))
        print(f"[ok] {path} -> {url}")
    except Exception as e:
        print(f"[error] {path}: {e}")

if DRY_RUN:
    print("[dry-run] no changes written.")
elif replacements:
    new_content = content
    for (a,b), rep in sorted(replacements, key=lambda x: x[0][0], reverse=True):
        new_content = new_content[:a] + rep + new_content[b:]
    md_path.write_text(new_content, encoding="utf-8")
    print(f"SUCCESS: Updated {MD_FILE}: {len(replacements)} image(s) replaced.")
else:
    print("No local images to replace.")
