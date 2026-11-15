# v12/guardrails.py  (run from project root, venv311)
# python v12/guardrails.py        -> offline self-test, no API calls
import sys, re, json
sys.path.insert(0, "src")
import contradiction as C
import strategist as S

# ---- Fix 1: skeptical strategist (patches the prompt in memory only) ----
EXTRA_RULES = """

ADDITIONAL RULES (override anything above):
7. A "CONTRADICTIONS IN PRECEDENTS" section is an UNVERIFIED claim by another tool. Never say a case was overruled, followed or distinguished unless an excerpt's own text says so.
8. If a contradiction claim is not supported by the excerpts, ignore it in your strategy and list it under RISK FACTORS as "unverified claim".
9. If you ignore a claim, do not let it raise STRENGTH ASSESSMENT or CONFIDENCE."""
if EXTRA_RULES not in S.SYSTEM_PROMPT:
    S.SYSTEM_PROMPT += EXTRA_RULES


# ---- Fix 2: detector must prove its claims with quotes ----
EVIDENCE_RULES = """

EVIDENCE RULES:
Add these string fields to every contradiction:
  "case_1_quote": exact words copied from the excerpts showing case 1's position (max 25 words)
  "case_2_quote": exact words copied from the excerpts showing case 2's position (max 25 words)
  "overruling_quote": exact words copied from the excerpts saying a case was overruled/overturned, or "" if none
Set "overruled" to true ONLY if "overruling_quote" is not empty. Copy quotes word for word. Never invent a quote."""


def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())).strip()


def quote_ok(quote, corpus_norm):
    """True if the quote (split at '...') appears word for word in the text."""
    parts = [p for p in re.split(r"\.\.\.|…", quote or "") if len(_norm(p).split()) >= 3]
    return bool(parts) and all(_norm(p) in corpus_norm for p in parts)


def verify_report(report, chunks, excerpts=""):
    corpus = _norm(excerpts + " " + " ".join(c.get("text", "") for c in chunks))
    kept, dropped = [], 0
    for c in report.get("contradictions", []):
        if not (quote_ok(c.get("case_1_quote"), corpus) and quote_ok(c.get("case_2_quote"), corpus)):
            dropped += 1
            continue
        if c.get("overruled") and not quote_ok(c.get("overruling_quote"), corpus):
            c["overruled"] = False              # claim without proof is removed
            c["note"] = "overruling claim removed: no supporting quote"
        kept.append(c)
    out = dict(report)
    out.update(contradictions=kept, contradiction_count=len(kept),
               contradictions_found=bool(kept),
               analysis_note=(report.get("analysis_note", "") +
                              f" [guardrail: {dropped} unverified dropped]").strip())
    return out


def detect_contradictions_verified(query, chunks):
    if not chunks:
        return C.no_contradiction("No chunks provided.")
    if not C.has_multiple_cases(chunks):
        return C.detect_contradictions(query, chunks)       # short-circuit as before
    excerpts = C.build_excerpts(chunks)
    prompt = C.USER_PROMPT_TEMPLATE.format(excerpts=excerpts) + EVIDENCE_RULES
    try:
        r = C.groq_client.chat.completions.create(
            model=C.MODEL, temperature=0.0, max_tokens=1500, reasoning_effort="none",
            messages=[{"role": "system", "content": C.SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}])
        parsed = C.parse_contradiction_response(r.choices[0].message.content)
        if not parsed:
            return C.no_contradiction("Could not parse contradiction analysis.")
        rep = {"contradictions_found": bool(parsed.get("contradictions_found")),
               "contradiction_count": 0,
               "contradictions": parsed.get("contradictions", []),
               "analysis_note": parsed.get("analysis_note", ""), "error": None}
        return verify_report(rep, chunks, excerpts)
    except Exception as e:
        return {**C.no_contradiction(""), "error": str(e)}


if __name__ == "__main__":
    chunks = [{"text": "The Court held that Article 22 is a self-contained code on preventive detention."}]
    fake = {"contradictions_found": True, "contradiction_count": 1, "analysis_note": "x", "contradictions": [{
        "case_1_quote": "Article 22 is a self-contained code on preventive detention",
        "case_2_quote": "Article 21 applies and the procedure must be fair just and reasonable",
        "overruled": True, "overruling_quote": "Gopalan has been overruled"}]}
    print("fake report  ->", verify_report(fake, chunks)["contradictions_found"], "(expect False)")
    good = {"contradictions_found": True, "analysis_note": "", "contradictions": [{
        "case_1_quote": "Article 22 is a self-contained code on preventive detention",
        "case_2_quote": "Article 22 is a self-contained code", "overruled": True,
        "overruling_quote": "Gopalan has been overruled"}]}
    r = verify_report(good, chunks)
    print("good quotes, fake overrule ->", r["contradictions_found"], r["contradictions"][0]["overruled"],
          "(expect True False)")
    print("strategist rules active:", "UNVERIFIED" in S.SYSTEM_PROMPT)