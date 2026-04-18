"""Replace all five table blocks in the tex file with the new-Pearson versions."""
from pathlib import Path

tex_path = Path(r"D:/Area-and-Distribution-Free-Estimator-for-TSP/paper_reference/Area_Free_Main.tex")
new_tables = Path(r"D:/Area-and-Distribution-Free-Estimator-for-TSP/paper_reference/scripts/new_tables.tex").read_text(encoding="utf-8")

# Parse new_tables into per-label blocks
blocks = {}  # label -> full \begin{table}...\end{table}
current_label = None
current_buf = []
in_table = False
for line in new_tables.splitlines(keepends=True):
    if line.startswith(r"\begin{table}"):
        in_table = True
        current_buf = [line]
        current_label = None
    elif in_table:
        current_buf.append(line)
        if r"\label{" in line and current_label is None:
            s = line.find(r"\label{") + len(r"\label{")
            e = line.find("}", s)
            current_label = line[s:e]
        if line.rstrip().endswith(r"\end{table}"):
            if current_label:
                blocks[current_label] = "".join(current_buf)
            in_table = False

print("Found new blocks:", list(blocks.keys()))

text = tex_path.read_text(encoding="utf-8")

for label, new_block in blocks.items():
    marker = f"\\label{{{label}}}"
    idx = text.find(marker)
    if idx == -1:
        print(f"SKIP {label}: not in tex")
        continue
    start = text.rfind(r"\begin{table}", 0, idx)
    end = text.find(r"\end{table}", idx) + len(r"\end{table}")
    if start == -1 or end < len(r"\end{table}"):
        print(f"SKIP {label}: bounds not found")
        continue
    old_len = end - start
    text = text[:start] + new_block.rstrip() + text[end:]
    print(f"Replaced {label}: old={old_len} new={len(new_block)}")

tex_path.write_text(text, encoding="utf-8")
print(f"Wrote {tex_path} ({len(text)} chars)")
