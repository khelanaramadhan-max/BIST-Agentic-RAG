#!/usr/bin/env python3
"""Inject BIST_API_BASE_URL into ui/index.html meta tag (used by GitHub Pages deploy)."""
import os
import pathlib
import re
import sys

url = os.environ.get("BIST_API_BASE_URL", "").strip()
if not url:
    print("BIST_API_BASE_URL is empty; skipping inject.", file=sys.stderr)
    sys.exit(0)

url = url.replace("&", "&amp;").replace('"', "&quot;")
path = pathlib.Path(__file__).resolve().parent.parent / "ui" / "index.html"
text = path.read_text(encoding="utf-8")
text, n = re.subn(
    r'(<meta\s+name="bist-api-base"\s+content=")[^"]*(")',
    r"\1" + url + r"\2",
    text,
    count=1,
)
if n != 1:
    sys.exit("Could not find <meta name=\"bist-api-base\" ...> in ui/index.html")
path.write_text(text, encoding="utf-8")
print("Injected bist-api-base into ui/index.html")
