from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
tex_path = REPO / "paper_reference/Area_Free_Main.tex"
new_block_path = REPO / "paper_reference/scripts/new_sections.tex"

lines = tex_path.read_text(encoding="utf-8").splitlines(keepends=True)

# Sanity check: find exact anchors
start_marker = "\\subsection{Adversarial 2D Dataset Construction}"
end_section_marker = "\\section{Conclusion}"

start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if start_idx is None and line.strip().startswith(start_marker):
        start_idx = i
    if line.strip().startswith(end_section_marker):
        end_idx = i
        break

if start_idx is None or end_idx is None:
    raise SystemExit(f"Anchors not found. start={start_idx}, end={end_idx}")

print(f"Splicing out lines {start_idx+1} to {end_idx} (inclusive start, exclusive end)")
print(f"Old line at start: {lines[start_idx].rstrip()}")
print(f"Old line at end: {lines[end_idx].rstrip()}")

new_block = new_block_path.read_text(encoding="utf-8")
# Ensure trailing blank line before Conclusion
if not new_block.endswith("\n\n"):
    new_block = new_block.rstrip("\n") + "\n\n"

new_lines = lines[:start_idx] + [new_block] + lines[end_idx:]
out = "".join(new_lines)
tex_path.write_text(out, encoding="utf-8")
print(f"Wrote {tex_path} ({len(out)} chars, {out.count(chr(10))} lines)")
