"""Write pipeline.svg (light) and pipeline-dark.svg.   python docs/figures/src/make_pipeline.py"""
# viewBox units; the README shows the SVG at about 920 px (77 %), so 19-unit titles render near 15 px.
from pathlib import Path

OUT = Path(__file__).parent.parent
THEMES = {  # tuned for GitHub's #ffffff and #0d1117 page grounds
    "pipeline.svg": dict(fill="#eef1f5", stroke="#57606a", title="#1f2328", sub="#424a53", note="#57606a",
                         arrow="#57606a", hi_fill="#e3eefb", hi_stroke="#1c5cab"),
    "pipeline-dark.svg": dict(fill="#21262d", stroke="#8b949e", title="#f0f6fc", sub="#c9d1d9", note="#9198a1",
                              arrow="#8b949e", hi_fill="#132a45", hi_stroke="#6da7ec")}
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"
W, H = 1200, 326
PAR = [("Speech detection", "TenVAD"), ("Speaker type", "VTC 2.0 (BabyHuBERT)"),
       ("Noise & reverb", "Brouhaha SNR, C50"), ("Environmental sound", "PANNs, 16 categories")]
PX, PW, PH, PG, PY0 = 238, 250, 58, 14, 44          # parallel column
SEQ = [(4, 176, ["Daylong", "recording"], "10–16 h, one child"),
       (560, 176, ["Cut at", "silences"], "clips ≤ 10 min"),
       (786, 176, ["Shards +", "metadata"], "~40 fields a clip"),
       (1012, 184, ["Filtered", "training batches"], "streamed to GPUs")]
SH = 92
MID = PY0 + (4 * PH + 3 * PG) / 2
LABEL = ("DL++ pipeline: a daylong recording goes through four detectors in parallel (speech detection with TenVAD, "
         "speaker type with VTC 2.0, noise and reverberation with Brouhaha, environmental sound with PANNs), is cut into "
         "clips at silences, written as shards with per-clip metadata, and streamed as filtered training batches.")


def t(x, y, s, size, fill, weight=400):
    return (f'<text x="{x}" y="{y}" text-anchor="middle" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}">{s.replace("&", "&amp;")}</text>')


def arrow(x1, y1, x2, y2, c):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c["arrow"]}" stroke-width="2" '
            f'marker-end="url(#ah)"/>')


for name, c in THEMES.items():
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{LABEL}">',
         f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
         f'<path d="M0,0 L10,5 L0,10 z" fill="{c["arrow"]}"/></marker></defs>']
    o.append(t(PX + PW / 2, 26, "in parallel, one cluster job each", 16, c["note"]))
    for x, w, lines, sub in SEQ:
        y = MID - SH / 2
        o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{SH}" rx="10" fill="{c["fill"]}" '
                 f'stroke="{c["stroke"]}" stroke-width="1.5"/>')
        for j, s in enumerate(lines):
            o.append(t(x + w / 2, y + 30 + j * 23, s, 19, c["title"], 600))
        o.append(t(x + w / 2, y + SH - 13, sub, 15, c["sub"]))
    for i, (title, sub) in enumerate(PAR):
        y = PY0 + i * (PH + PG)
        o.append(f'<rect x="{PX}" y="{y}" width="{PW}" height="{PH}" rx="10" fill="{c["hi_fill"]}" '
                 f'stroke="{c["hi_stroke"]}" stroke-width="1.5"/>')
        o.append(t(PX + PW / 2, y + 25, title, 19, c["title"], 600))
        o.append(t(PX + PW / 2, y + 47, sub, 15, c["sub"]))
        cy = y + PH / 2
        o.append(arrow(SEQ[0][0] + SEQ[0][1] + 4, MID, PX - 4, cy, c))            # fan out
        o.append(arrow(PX + PW + 4, cy, SEQ[1][0] - 4, MID + (i - 1.5) * 20, c))                   # fan in
    for (x, w, *_), (nx, *_) in zip(SEQ[1:], SEQ[2:]):
        o.append(arrow(x + w + 4, MID, nx - 4, MID, c))
    o.append("</svg>")
    (OUT / name).write_text("\n".join(o) + "\n")
