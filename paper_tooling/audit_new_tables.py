from pathlib import Path
import re

REPO = Path(__file__).resolve().parent.parent
p = REPO / "paper_reference/scripts/new_tables.tex"
t = p.read_text(encoding="utf-8")

minv, maxv = 999.0, -999.0
bad = []
all_vals = []
for line in t.splitlines():
    if "& GART" in line or "& MST Ratio" in line:
        cells = line.split("&")
        if len(cells) >= 9:
            raw = cells[7].strip()
            # strip \textbf{...}, $...$, backslashes
            s = raw
            s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
            s = s.replace("$", "").replace("\\", "").strip()
            if s == "---":
                continue
            try:
                v = float(s)
            except ValueError:
                continue
            all_vals.append((cells[1].strip(), v, raw))
            minv = min(minv, v)
            maxv = max(maxv, v)
            if v < -1 or v > 1:
                bad.append((cells[1].strip(), raw, v))

print(f"total r_alpha values checked: {len(all_vals)}")
print(f"min = {minv:.4f}")
print(f"max = {maxv:.4f}")
print(f"values outside [-1, 1]: {len(bad)}")
for b in bad:
    print(" ", b)
print()
print("sample of all values:")
for model, v, raw in all_vals[:20]:
    print(f"  {model:20s}  {v:+.4f}   ({raw})")
