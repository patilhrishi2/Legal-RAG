import sys, pdfplumber
from chunker import _segments, chunk_judgment
with pdfplumber.open(sys.argv[1]) as pdf:
    text = "\n".join((p.extract_text() or "") for p in pdf.pages)
print("SEGMENTS (opinion, words, start):")
for op, paras in _segments(text):
    print(f"  {op!s:<18} {sum(len(p.split()) for p in paras):>6}  {paras[0][:70]!r}")
print("\nCHUNKS over 420 words:")
for i, c in enumerate(chunk_judgment(text)):
    w = len(c["text"].split())
    if w > 420:
        print(f"  chunk {i}: {w} words, opinion={c['opinion']}, ends {c['text'][-80:]!r}")