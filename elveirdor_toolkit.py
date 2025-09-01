
"""
Elveirdor Toolkit
-----------------
All-in-one Python utility that:
A) Implements encoder/decoder for English A1Z26 and the Elveirdor Ichthys cipher.
B) Parses a .docx file (no third-party dependencies required) into text + simple sections.
C) Extracts numeric sequences and attempts bi-cipher decoding.
Outputs JSON/CSV artifacts.

Usage (library-style):
    from elveirdor_toolkit import *
    text = docx_to_text(docx_path)
    sections = split_sections(text)
    results = decode_all_numeric_sequences(text)

Usage (CLI):
    python elveirdor_toolkit.py "/path/to/file.docx"

Outputs:
    - elveirdor_outputs/decoded_sequences.json
    - elveirdor_outputs/decoded_sequences.csv
    - elveirdor_outputs/sections.json
"""

from __future__ import annotations
import re
import json
import csv
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# --------------------
# Cipher definitions
# --------------------

# A1Z26 English mapping (A=1, ... Z=26)
ENG_L2N = {chr(ord('A') + i): i+1 for i in range(26)}
ENG_N2L = {v: k for k, v in ENG_L2N.items()}

# Elveirdor Ichthys mapping (as supplied)
# Supports digraph tokens: Ch, Ck, LL, Ph, St, Th, Wh
ELV_TOKENS = [
    ("CH", 4),
    ("CK", 5),
    ("LL", 16),
    ("PH", 21),
    ("ST", 25),
    ("TH", 27),
    ("WH", 31),
]

# Base single-letter mapping where letters that are not special digraphs
# follow the provided numbers. For English letters that differ (W=30, etc.), we set them here.
ELV_L2N = {
    "A": 1, "B": 2, "C": 3,
    # CH=4 as digraph
    # CK=5 as digraph
    "D": 6, "E": 7, "EE": 8,  # "Ee" in source; we add EE token for completeness but won't auto-emit it
    "F": 9, "G": 10, "H": 11, "I": 12, "J": 13, "K": 14, "L": 15,
    # LL=16 as digraph
    "M": 17, "N": 18, "O": 19, "P": 20,
    # PH=21 digraph
    "Q": 22, "R": 23, "S": 24,
    # ST=25 digraph
    "T": 26,
    # TH=27 digraph
    "U": 28, "V": 29, "W": 30,
    # WH=31 digraph
    "X": 32, "Y": 33, "Z": 34,
}

# We will prefer digraphs on encoding where applicable (e.g., "THIS" -> TH + I + S).
# Build a tokenization order (longest-first) for Elveirdor text.
ELV_TOKEN_ORDER = [t for t, _ in ELV_TOKENS] + list(ELV_L2N.keys())
# Sort by token length descending to prefer digraphs first
ELV_TOKEN_ORDER.sort(key=len, reverse=True)

ELV_N2SYM = {num: sym for sym, num in ([*ELV_TOKENS] + list(ELV_L2N.items()))}

# --------------------
# Encoding / Decoding core
# --------------------

def encode_english_a1z26(text: str) -> List[int]:
    """Encode letters to A1Z26; ignore non-letters."""
    out = []
    for ch in text.upper():
        if ch.isalpha():
            out.append(ENG_L2N[ch])
    return out

def decode_english_a1z26_concat(num_string: str) -> Optional[str]:
    """
    Decode a concatenated number string into letters using A1Z26.
    Example: "3118181520" -> "CARROT".
    Uses backtracking to resolve ambiguity.
    Returns None if no valid parse.
    """
    digits = num_string.strip()
    if not digits or not digits.isdigit():
        return None

    memo = {}

    def dfs(i: int) -> Optional[str]:
        if i == len(digits):
            return ""
        if i in memo:
            return memo[i]
        # Try 1 or 2 digits (since 1..26)
        for l in (2, 1):  # prefer two-digit first for common words
            if i + l <= len(digits):
                val = int(digits[i:i+l])
                if 1 <= val <= 26:
                    rest = dfs(i + l)
                    if rest is not None:
                        memo[i] = ENG_N2L[val] + rest
                        return memo[i]
        memo[i] = None
        return None

    return dfs(0)

def tokenize_elv_text_for_encoding(text: str) -> List[str]:
    """
    Tokenize text to Elveirdor tokens, preferring digraphs (CH, CK, LL, PH, ST, TH, WH).
    Non-letters are ignored.
    """
    s = re.sub(r'[^A-Za-z]', '', text.upper())
    tokens = []
    i = 0
    while i < len(s):
        matched = False
        for tok in ELV_TOKEN_ORDER:
            L = len(tok)
            if s[i:i+L] == tok:
                tokens.append(tok)
                i += L
                matched = True
                break
        if not matched:
            # Fallback: single letter if exists
            ch = s[i]
            if ch in ELV_L2N:
                tokens.append(ch)
            # else drop unknown
            i += 1
    return tokens

def encode_elveirdor(text: str) -> List[int]:
    tokens = tokenize_elv_text_for_encoding(text)
    out = []
    for tok in tokens:
        if tok in ELV_L2N:
            out.append(ELV_L2N[tok])
        else:
            # digraphs
            for sym, num in ELV_TOKENS:
                if tok == sym:
                    out.append(num)
                    break
    return out

def decode_elveirdor_concat(num_string: str) -> Optional[str]:
    """
    Decode concatenated number string using Elveirdor (1..34).
    Prefer 2-digit first, then 1-digit. Map back to symbols (digraph codes).
    """
    digits = num_string.strip()
    if not digits or not digits.isdigit():
        return None

    memo = {}

    def dfs(i: int) -> Optional[List[str]]:
        if i == len(digits):
            return []
        if i in memo:
            return memo[i]
        # Try 2 then 1 digits (since 1..34)
        for l in (2, 1):
            if i + l <= len(digits):
                val = int(digits[i:i+l])
                if 1 <= val <= 34 and val in ELV_N2SYM:
                    rest = dfs(i + l)
                    if rest is not None:
                        memo[i] = [ELV_N2SYM[val]] + rest
                        return memo[i]
        memo[i] = None
        return None

    syms = dfs(0)
    if syms is None:
        return None
    # Collapse digraph symbols by just concatenating them (e.g., TH + E -> "THE")
    # Note: We keep "EE" token literal if encountered.
    return "".join(syms)

# --------------------
# .docx text extraction (no third-party deps)
# --------------------

def docx_to_text(path: Path) -> str:
    """
    Extracts visible paragraph text from a .docx by reading word/document.xml.
    """
    with zipfile.ZipFile(path) as z:
        xml_content = z.read("word/document.xml")
    # Parse XML and extract text within w:t nodes
    # Namespaces in DOCX XML
    NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    root = ET.fromstring(xml_content)
    texts = []
    for t in root.findall(".//w:t", NS):
        texts.append(t.text or "")
    # Join with newlines between paragraphs (heuristic: w:p nodes)
    # For simplicity, just join with spaces and normalize later.
    raw = " ".join(texts)
    # Collapse multiple spaces and restore basic newlines on "double spaces after punctuation" heuristics
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw

# --------------------
# Simple section splitter & sequence extraction
# --------------------

SECTION_HEAD_RE = re.compile(r"(?im)^(?:[A-Z][A-Z0-9 \-\(\):]+)$")

def split_sections(text: str) -> List[Dict[str, str]]:
    """
    Naive splitter: identifies lines that look like headings (ALL CAPS-ish) and
    groups following lines until next heading.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    sections = []
    current = {"title": "ROOT", "body": []}
    for ln in lines:
        if ln.isupper() or SECTION_HEAD_RE.match(ln):
            # start new section
            if current["body"]:
                sections.append({"title": current["title"], "body": "\n".join(current["body"])})
            current = {"title": ln, "body": []}
        else:
            current["body"].append(ln)
    if current["body"]:
        sections.append({"title": current["title"], "body": "\n".join(current["body"])})
    return sections

NUMSEQ_RE = re.compile(r"(?:\d{1,3})+")

def extract_numeric_sequences(text: str) -> List[str]:
    """
    Pulls out long-ish digit-only runs (>= 4 chars) that look like concatenated codes.
    """
    seqs = []
    for m in NUMSEQ_RE.finditer(text):
        s = m.group(0)
        if len(s) >= 4:
            seqs.append(s)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in seqs:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique

# --------------------
# Decoding attempts
# --------------------

@dataclass
class DecodeAttempt:
    sequence: str
    english: Optional[str]
    elveirdor: Optional[str]

def decode_all_numeric_sequences(text: str) -> List[DecodeAttempt]:
    seqs = extract_numeric_sequences(text)
    results = []
    for s in seqs:
        eng = decode_english_a1z26_concat(s)
        elv = decode_elveirdor_concat(s)
        results.append(DecodeAttempt(sequence=s, english=eng, elveirdor=elv))
    return results

# --------------------
# Serialization helpers
# --------------------

def save_decodes_to_json_csv(decodes: List[DecodeAttempt], out_dir: Path) -> Tuple[Path, Path]:
    out_json = out_dir / "decoded_sequences.json"
    out_csv  = out_dir / "decoded_sequences.csv"
    data = [dict(sequence=d.sequence, english=d.english, elveirdor=d.elveirdor) for d in decodes]
    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "english", "elveirdor"])
        for d in decodes:
            w.writerow([d.sequence, d.english or "", d.elveirdor or ""])
    return out_json, out_csv

def save_sections(sections: List[Dict[str, str]], out_dir: Path) -> Path:
    out_path = out_dir / "sections.json"
    out_path.write_text(json.dumps(sections, indent=2), encoding="utf-8")
    return out_path

# --------------------
# CLI
# --------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Elveirdor Toolkit")
    parser.add_argument("docx", type=str, help="Path to .docx file")
    parser.add_argument("--out", type=str, default=str(Path("elveirdor_outputs")), help="Output directory")
    args = parser.parse_args()

    docx_path = Path(args.docx)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    text = docx_to_text(docx_path)
    sections = split_sections(text)
    decodes = decode_all_numeric_sequences(text)

    sec_path = save_sections(sections, out_dir)
    json_path, csv_path = save_decodes_to_json_csv(decodes, out_dir)

    print(f"[OK] Extracted text from: {docx_path}")
    print(f"[OK] Sections JSON: {sec_path}")
    print(f"[OK] Decoded sequences JSON: {json_path}")
    print(f"[OK] Decoded sequences CSV:  {csv_path}")
    print("[TIP] You can import encode/decode helpers from this module for custom use.")

if __name__ == "__main__":
    main()
