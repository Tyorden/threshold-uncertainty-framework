import json, sys, os, glob, math, hashlib
O, N = sys.argv[1], sys.argv[2]
def cmp(a, b, path, diffs):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b: diffs.append((path + "/" + k, "missing", None)); continue
            cmp(a[k], b[k], path + "/" + k, diffs)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): diffs.append((path, "len", None)); return
        for i, (x, y) in enumerate(zip(a, b)): cmp(x, y, f"{path}[{i}]", diffs)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if a != b:
            rel = abs(a - b) / max(abs(a), abs(b), 1e-300); diffs.append((path, rel, (a, b)))
    elif a != b: diffs.append((path, "value", (str(a)[:40], str(b)[:40])))
for f in sorted(glob.glob(os.path.join(O, "results", "**", "*.json"), recursive=True)):
    r = os.path.relpath(f, O); g = os.path.join(N, r)
    d = []; cmp(json.load(open(f)), json.load(open(g)), "", d)
    d = [x for x in d if not x[0].endswith("seconds") and "sec_per_draw" not in x[0] and "/seconds" not in x[0]]
    num = [x for x in d if isinstance(x[1], float)]
    big = [x for x in num if x[1] > 1e-9]
    print(f"{r}: differing leaves {len(d)}; numeric with rel diff > 1e-9: {len(big)}; max rel {max([x[1] for x in num], default=0):.2e}; non-numeric {len(d)-len(num)}")
    for x in (big + [y for y in d if not isinstance(y[1], float)])[:6]: print("    ", x)
for f in sorted(glob.glob(os.path.join(O, "results", "**", "*.*"), recursive=True)):
    if f.endswith(".json") or f.endswith(".txt"): continue
    r = os.path.relpath(f, O); g = os.path.join(N, r)
    h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    print(r, "byte-identical" if h(f) == h(g) else "DIFFERENT bytes")
