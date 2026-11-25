import re, difflib

def extract(text):
    """text = first chunks of a case joined. Returns (dissenters, held_line)."""
    names = set()
    for inner in re.findall(r"\(([^()]*dissent[a-z]*[^()]*)\)", text, re.I):
        inner = re.split(r"\bJJ?\b", inner)[0]
        for n in re.split(r"\band\b|,", inner):
            n = re.sub(r"[^A-Za-z ]", "", n).strip()
            if len(n) > 2:
                names.add(n.title())
    m = re.search(r"Held,?\s+(?:per|by)[^.]{0,300}", text)
    return sorted(names), (m.group(0).strip() if m else "")

def judge_dissented(opinion, dissenters):
    return bool(opinion) and any(
        difflib.SequenceMatcher(None, opinion.lower(), d.lower()).ratio() > 0.75
        for d in dissenters)

def load_notes(conn):
    try:
        rows = conn.execute("SELECT source_file, dissenters, held_line FROM case_notes").fetchall()
    except Exception:
        return {}
    return {r[0]: ([x for x in r[1].split("|") if x], r[2]) for r in rows}