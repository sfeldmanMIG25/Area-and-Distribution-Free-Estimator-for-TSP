"""Replace the TSPLIB by-size table block with the new 4-bin version."""
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parent.parent
tex_path = REPO / "paper_reference/Area_Free_Main.tex"

# Generate the new table block
result = subprocess.run(
    ["python", str(REPO / "paper_reference/scripts/gen_tsplib_4bin.py")],
    capture_output=True, text=True, check=True
)
new_block = result.stdout.rstrip() + "\n"

text = tex_path.read_text(encoding="utf-8")

# Find the old table: starts with `\begin{table}` before `\label{tab:tsplib_by_size}`, ends at `\end{table}`
start_marker = r"\begin{table}[!ht]"
label_marker = r"\label{tab:tsplib_by_size}"
end_marker = r"\end{table}"

# Find label position
label_idx = text.find(label_marker)
if label_idx == -1:
    raise SystemExit("label not found")
# Find the \begin{table} before it (last occurrence before label_idx)
start_idx = text.rfind(start_marker, 0, label_idx)
if start_idx == -1:
    raise SystemExit("start marker not found")
# Find \end{table} after label
end_idx = text.find(end_marker, label_idx)
if end_idx == -1:
    raise SystemExit("end marker not found")
end_idx += len(end_marker)

print(f"Replacing text from char {start_idx} to {end_idx} (length {end_idx-start_idx})")
new_text = text[:start_idx] + new_block + text[end_idx:]
tex_path.write_text(new_text, encoding="utf-8")
print(f"Wrote {tex_path} ({len(new_text)} chars)")
