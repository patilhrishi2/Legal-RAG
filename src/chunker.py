# src/chunker.py  (new)
import re

TARGET_WORDS   = 350   # flush a chunk once it reaches this
MAX_WORDS      = 420   # hard ceiling (~600 tokens, safe for 768 limit)
OVERLAP_WORDS  = 50
MIN_TAIL_WORDS = 60    # a smaller leftover is merged into the previous chunk
MIN_SEG_WORDS  = 40   # a smaller opinion segment is folded into its neighbour

_KANOON  = re.compile(r"^Indian Kanoon\s*-\s*http\S+.*$")
# lone line such as "Mahajan, J."
_OPINION = re.compile(r"^([A-Z][A-Za-z'.\- ]{1,40}),\s*(?:C\.?\s?J|J\.?J|J)\.?\s*$")
# same-line start such as "FAZL ALI J.--The question ..."
_OPINION2 = re.compile(r"^([A-Z][A-Z.'\- ]{2,40}?)\s*,?\s*(?:C\.?\s?J|J\.?J?)\.?\s*[-—–]{1,3}\s*\S")
_PARA_START = re.compile(r"^\d{1,3}\.\s?[A-Z\"(]")
_HYPHEN  = re.compile(r"(?<=[a-z])-\s+(?=[a-z])")
_ABBREV  = {"mr","mrs","dr","no","nos","rs","s","ss","art","arts","sec","secs","vs","v",
            "j","cj","ors","anr","ltd","co","st","re","viz","ibid","p","pp","cl","sch",
            "ch","vol","all","cal","bom","mad","pat","lah"}


def get_title(text: str) -> str:
    for line in text.split("\n"):
        if line.strip():
            return line.strip()
    return ""


def _is_abbrev_end(s: str) -> bool:
    tok = s.split()[-1] if s.split() else ""
    if not tok.endswith("."):
        return False
    t = tok.rstrip(".").lower()
    return t in _ABBREV or len(t) == 1 or t.isdigit() or bool(re.fullmatch(r"(?:[a-z]\.)+[a-z]?", tok.lower()))


def _sentences(paragraph: str) -> list[str]:
    parts = re.split(r'(?<=[.?!])\s+(?=["“(\[]?[A-Z0-9])', paragraph)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if out and _is_abbrev_end(out[-1]):
            out[-1] += " " + p
        else:
            out.append(p)
    return out


def _segments(text: str):
    """Clean the text and split it at opinion markers. Yields (opinion_or_None, [paragraphs])."""
    norm  = lambda s: " ".join(s.split())
    title = norm(get_title(text))
    lines, seen_title, kept = [], False, 0
    for raw in text.split("\n"):
        line = raw.strip()
        if not line or _KANOON.match(line):
            continue
        nl = norm(line)
        if nl == title:                      # exact repeat (page headers, line 2)
            if seen_title:
                continue
            seen_title = True
        elif seen_title and kept < 4 and len(nl) >= 4 and nl in title:
            continue                         # wrapped title fragment, e.g. "1951"
        lines.append(line)
        kept += 1

    segs = [[None, []]]                      # [opinion, paragraphs]
    buf = []

    def flush():
        if buf:
            segs[-1][1].append(" ".join(buf))
            buf.clear()

    n = len(lines)
    for i, line in enumerate(lines):
        m = _OPINION.match(line)
        if m and i > 2:                      # lone "Mahajan, J." line                  # lone "Mahajan, J." line
            flush()
            segs.append([m.group(1).strip(), []])
            continue
        m2 = _OPINION2.match(line)
        if m2 and i > n * 0.15:              # "FAZL ALI J.--text", skipping the headnote
            flush()
            segs.append([m2.group(1).strip().title(), []])
        elif _PARA_START.match(line):
            flush()
        buf.append(line)
    flush()

    # fold tiny segments (headnote bits, stray matches) into a neighbour
    merged, carry = [], []
    for op, paras in segs:
        words = sum(len(p.split()) for p in paras)
        if words < MIN_SEG_WORDS:
            if merged:
                merged[-1][1].extend(paras)
            else:
                carry = carry + paras        # short header: attach to the next segment
        else:
            merged.append([op, carry + paras])
            carry = []
    if carry and not merged:
        merged.append([None, carry])

    for opinion, paras in merged:
        paras = [_HYPHEN.sub("", p) for p in paras]
        if any(p.strip() for p in paras):
            yield opinion, paras


def _pack(sentences: list[str]) -> list[str]:
    # break absurdly long "sentences" into word windows first
    flat = []
    for s in sentences:
        w = s.split()
        if len(w) > MAX_WORDS:
            flat += [" ".join(w[i:i + TARGET_WORDS]) for i in range(0, len(w), TARGET_WORDS)]
        else:
            flat.append(s)

    chunks, cur, new, n = [], [], [], 0
    for s in flat:
        w = len(s.split())
        if cur and not new and n + w > MAX_WORDS:
            cur, n = [], 0                   # drop the overlap rather than exceed the ceiling
        if cur and new and (n >= TARGET_WORDS or n + w > MAX_WORDS):
            chunks.append(" ".join(cur))
            ov, k = [], 0
            for t in reversed(cur):
                if k >= OVERLAP_WORDS:
                    break
                ov.insert(0, t); k += len(t.split())
            if k >= TARGET_WORDS or len(ov) == len(cur):
                ov, k = [], 0
            cur, new, n = ov, [], k
        cur.append(s); new.append(s); n += w

    if new:
        tail_words = sum(len(t.split()) for t in new)
        if (chunks and tail_words < MIN_TAIL_WORDS
                and len(chunks[-1].split()) + tail_words <= MAX_WORDS):
            chunks[-1] += " " + " ".join(new)
        else:
            chunks.append(" ".join(cur))
    return chunks


def _hard_split(text: str, limit: int = 430) -> list[str]:
    w = text.split()
    if len(w) <= limit:
        return [text]
    n = -(-len(w) // limit)            # ceil
    size = -(-len(w) // n)
    return [" ".join(w[i:i + size]) for i in range(0, len(w), size)]


def chunk_judgment(text: str) -> list[dict]:
    """Returns [{"text": clean chunk text, "opinion": 'Mahajan' or None}, ...]"""
    result = []
    for opinion, paras in _segments(text):
        sents = [s for p in paras for s in _sentences(p)]
        for c in _pack(sents):
            for part in _hard_split(c):
                result.append({"text": part, "opinion": opinion})
    return result


def embed_text(chunk: dict, title: str) -> str:
    """What gets embedded (stored text stays clean): title + opinion author + chunk."""
    head = title + (f". Opinion of {chunk['opinion']}." if chunk["opinion"] else ".")
    return head + "\n" + chunk["text"]


if __name__ == "__main__":
    import sys, pdfplumber
    with pdfplumber.open(sys.argv[1]) as pdf:
        text = "\n".join((p.extract_text() or "") for p in pdf.pages)
    chunks = chunk_judgment(text)
    wc = [len(c["text"].split()) for c in chunks]
    print(f"{len(chunks)} chunks | words min {min(wc)} / avg {sum(wc)//len(wc)} / max {max(wc)}")
    print("opinions:", sorted({c['opinion'] for c in chunks if c['opinion']}) or "none found")
    print("noise left:", sum("Indian Kanoon" in c["text"] for c in chunks), "chunks")
    print("title copies in chunk 0:", chunks[0]["text"].count(get_title(text)[:30]))
    for i in (0, len(chunks) - 1):
        print(f"\n--- chunk {i} (opinion={chunks[i]['opinion']}) ---\n{chunks[i]['text'][:500]}...")