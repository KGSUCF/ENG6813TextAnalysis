"""
Export each panel from the corpus analysis as a standalone figure.
"""

import re, os, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE    = "/root/.claude/uploads/009355d3-afe3-4522-8444-12129a13d27e"
FOX_PATH = os.path.join(BASE, "92e14d39-Fox_Entertainment_Files200_text.txt")
NYT_PATH = os.path.join(BASE, "165ec9af-NYT_Files200_text.txt")
OUT_DIR  = "/home/user/ENG6813TextAnalysis/panels"
os.makedirs(OUT_DIR, exist_ok=True)

C_FOX, C_NYT = "#D42B28", "#1A4BA0"

# ── parser ───────────────────────────────────────────────────────────────────
MONTHS = (r"January|February|March|April|May|June|July|"
          r"August|September|October|November|December")
DATE_WEEKDAY = re.compile(
    rf"^({MONTHS})\s+(\d{{1,2}}),\s+(\d{{4}})\s+"
    r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)", re.I)

def _find_headline(lines, di):
    non_blank = []
    for i in range(di - 1, max(di - 8, -1), -1):
        s = lines[i].strip()
        if s:
            non_blank.append(s)
            if len(non_blank) >= 2: break
    if not non_blank:               return ""
    if len(non_blank) == 1:         return non_blank[0]
    if len(non_blank[0].split()) <= 3: return non_blank[1]
    return non_blank[0]

def _extract_body(lines, start, end):
    body, skip, past_hl = [], 0, False
    hl_re  = re.compile(r"^Highlight[\s:»\xa0]", re.I)
    end_re = re.compile(
        r"^(Classification|Subject[\s:»\xa0]|Industry[\s:»\xa0]|"
        r"Person[\s:»\xa0]|Geographic[\s:»\xa0]|Load-Date:|"
        r"Organization[\s:»\xa0]|Notes\s*$|"
        r"Link to the original|Watch the clip|"
        r"The post .{5,80} first appeared on)", re.I)
    noise_re = re.compile(
        r"^\s*(Delivered by|All Rights Reserved|\d{3,4} words?|"
        r"(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th),\s+\d{4})",
        re.I)
    for ln in lines[start:end]:
        s = ln.strip()
        if end_re.match(s):   break
        if noise_re.match(s): continue
        if skip < 7 and not past_hl:
            skip += 1; continue
        if hl_re.match(s):
            past_hl = True; continue
        if s: body.append(s)
    return " ".join(body)

def parse_file(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    date_idx = [(i, int(m.group(3)))
                for i, ln in enumerate(lines)
                if (m := DATE_WEEKDAY.match(ln))]
    arts = []
    for k, (di, year) in enumerate(date_idx):
        hl  = _find_headline(lines, di)
        end = date_idx[k+1][0] - 4 if k+1 < len(date_idx) else len(lines)
        body = _extract_body(lines, di + 1, end)
        if len(body) >= 80 and hl:
            arts.append({"headline": hl, "year": year, "body": body})
    return arts

print("Parsing …")
fox = parse_file(FOX_PATH)
nyt = parse_file(NYT_PATH)
print(f"Fox {len(fox)}  NYT {len(nyt)}")

# ── lexicons ─────────────────────────────────────────────────────────────────
TERMS = {
    "climate change": re.compile(r"climate\s+change",                      re.I),
    "global warming": re.compile(r"global\s+warming",                      re.I),
    "climate crisis": re.compile(r"climate\s+crisis",                      re.I),
    "hoax (any)":     re.compile(r"\bhoax\b",                               re.I),
}
ALARM_RE = re.compile(
    r"\b(catastroph|crisis|disast|doom|deadly|emergency|existential|extreme|"
    r"devastating|dangerous|threat|alarm|warning|worsen|irreversible|"
    r"unprecedented|urgent|severe|accelerat|escalat|dire|peril)\w*\b", re.I)
DOUBT_RE = re.compile(
    r"\b(hoax|fake|scam|con\b|myth|fraud|hysteria|alarmis[mt]|scare|junk|"
    r"fiction|narrative|propaganda|mislead|politicize|politicization|"
    r"radical|woke|agenda|lie|lies|misinform|debunk|false)\w*\b", re.I)

def cnt(text, pat): return len(pat.findall(text))
def tone(text):
    a, d = cnt(text, ALARM_RE), cnt(text, DOUBT_RE)
    return (a - d) / (a + d) if (a + d) else 0.0
def wds(arts, f="body"):   return sum(len(a[f].split()) for a in arts) or 1
def trate(arts, pat, f="body"):
    return sum(cnt(a[f], pat) for a in arts) / wds(arts, f) * 1000
def all_rates(arts, f="body"):
    return {t: trate(arts, p, f) for t, p in TERMS.items()}
def pyr(arts, pat, f="body"):
    by_y = collections.defaultdict(list)
    for a in arts: by_y[a["year"]].append(a)
    ys = sorted(by_y)
    return ys, [trate(by_y[y], pat, f) for y in ys]
def era(arts, y0, y1): return [a for a in arts if y0 <= a["year"] <= y1]
def ad(arts, f="body"):
    return (sum(cnt(a[f], ALARM_RE) for a in arts),
            sum(cnt(a[f], DOUBT_RE) for a in arts))

fox_h = np.array([tone(a["headline"]) for a in fox])
fox_b = np.array([tone(a["body"])     for a in fox])
nyt_h = np.array([tone(a["headline"]) for a in nyt])
nyt_b = np.array([tone(a["body"])     for a in nyt])
fox_gap = fox_b - fox_h
nyt_gap = nyt_b - nyt_h

fox_by_year = collections.Counter(a["year"] for a in fox)
nyt_by_year = collections.Counter(a["year"] for a in nyt)
all_years   = sorted(set(fox_by_year) | set(nyt_by_year))

fox_C = era(fox, 2021, 2023);  fox_D = era(fox, 2024, 2026)
nyt_C = era(nyt, 2021, 2023);  nyt_D = era(nyt, 2024, 2026)
fox_A = era(fox, 2017, 2020)

TK = dict(fontsize=13, fontweight="bold", pad=10, color="#111")
LK = dict(fontsize=10, color="#444")
BG = "#F8F6F1"

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved → {path}")
    return path

saved = []

# ─── Panel A: year distribution ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor("#FDFDFD")
ys = all_years
w = 0.38; x = np.arange(len(ys))
ax.bar(x - w/2, [fox_by_year.get(y,0) for y in ys], w, color=C_FOX, alpha=0.85, label="Fox")
ax.bar(x + w/2, [nyt_by_year.get(y,0) for y in ys], w, color=C_NYT, alpha=0.85, label="NYT")
ax.set_xticks(x)
ax.set_xticklabels([str(y) for y in ys], fontsize=9, rotation=45, ha="right")
ax.set_ylabel("Articles in corpus", **LK)
ax.set_title("Panel A — Corpus Year Distribution\n"
             "NYT sample covers 2021–2026 only; Fox spans 2009–2026", **TK)
ax.legend(fontsize=10, framealpha=0.8)
ax.axvspan(-0.5, ys.index(2020)+0.5, alpha=0.07, color="#FF9900")
ax.axvspan(ys.index(2021)-0.5, ys.index(2023)+0.5, alpha=0.07, color="#77BB44")
ax.axvspan(ys.index(2024)-0.5, len(ys)-0.5, alpha=0.07, color="#4488FF")
for label, xi, col in [("2017–20\n(Fox only)", ys.index(2017), "#FF9900"),
                        ("2021–23\n(both)", ys.index(2021), "#77BB44"),
                        ("2024–26\n(both)", ys.index(2024), "#4488FF")]:
    ax.text(xi+0.1, ax.get_ylim()[1]*0.82, label, fontsize=8, color=col, style="italic")
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
saved.append(save(fig, "panel_A_year_distribution.png"))

# ─── Panel B: headline vs body tone scatter ───────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9))
fig.patch.set_facecolor(BG); ax.set_facecolor("#FDFDFD")
rng = np.random.default_rng(42)
for h, b, c, lbl in [(fox_h, fox_b, C_FOX, "Fox"), (nyt_h, nyt_b, C_NYT, "NYT")]:
    jh = h + rng.uniform(-0.015, 0.015, len(h))
    jb = b + rng.uniform(-0.015, 0.015, len(b))
    ax.scatter(jh, jb, alpha=0.28, s=22, color=c, linewidths=0)
    ax.scatter([h.mean()], [b.mean()], s=150, color=c, zorder=6,
               marker="D", edgecolors="white", linewidths=1.5,
               label=f"{lbl}  (mean headline {h.mean():+.2f}, body {b.mean():+.2f})")
ax.axline((0,0), slope=1, color="#999", lw=1.3, ls="--", label="headline = body (no gap)")
ax.axhline(0, color="#ccc", lw=0.7); ax.axvline(0, color="#ccc", lw=0.7)
ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
ax.set_xlabel("Headline tone  (−1 = purely skeptical/dismissive · +1 = purely alarming)", **LK)
ax.set_ylabel("Body-text tone  (−1 = skeptical · +1 = alarming)", **LK)
ax.set_title("Panel B — Headline Tone vs. Body-Text Tone per Article\n"
             "Above dashed line = body more alarming than headline; below = opposite", **TK)
ax.legend(fontsize=10, framealpha=0.78, loc="upper left")
for tx, ty, txt in [(0.78,-0.85,"skeptical body"), (-0.78,0.85,"alarming body"),
                     (0.78,0.85,"alarming\nhead + body"), (-0.78,-0.85,"skeptical\nhead + body")]:
    ax.text(tx, ty, txt, ha="center", va="center", fontsize=8, color="#bbb", style="italic")
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
saved.append(save(fig, "panel_B_headline_body_scatter.png"))

# ─── Panel C: gap distribution ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor("#FDFDFD")
bins = np.linspace(-1.1, 1.1, 32)
ax.hist(fox_gap, bins=bins, color=C_FOX, alpha=0.58, density=True, label="Fox")
ax.hist(nyt_gap, bins=bins, color=C_NYT, alpha=0.58, density=True, label="NYT")
for gap, c, lbl in [(fox_gap, C_FOX, "Fox"), (nyt_gap, C_NYT, "NYT")]:
    m = float(np.mean(gap))
    ax.axvline(m, color=c, lw=2.2, ls="--", label=f"{lbl} mean {m:+.2f}")
ax.axvline(0, color="#666", lw=1.2)
ax.set_xlabel("Body tone − Headline tone\n(positive = body more alarming than headline)", **LK)
ax.set_ylabel("Density", **LK)
ax.set_title("Panel C — Headline ↔ Body Divergence Distribution\n"
             "How often and by how much do headlines misrepresent body tone?", **TK)
ax.legend(fontsize=10, framealpha=0.75)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
saved.append(save(fig, "panel_C_divergence_distribution.png"))

# ─── Panel D: alarming vs doubt balance ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor("#FDFDFD")
era_sets = [
    ("Fox\n2021–23", fox_C, C_FOX, 0.52),
    ("Fox\n2024–26", fox_D, C_FOX, 0.90),
    ("NYT\n2021–23", nyt_C, C_NYT, 0.52),
    ("NYT\n2024–26", nyt_D, C_NYT, 0.90),
]
x_d = np.arange(4); w_d = 0.30
for xi, (lbl, arts, col, alph) in enumerate(era_sets):
    a, d = ad(arts); tot = a + d or 1
    ax.bar(xi-w_d/2, a/tot, w_d, color=col, alpha=alph)
    ax.bar(xi+w_d/2, d/tot, w_d, color=col, alpha=alph*0.38, hatch="////", edgecolor=col)
    ax.text(xi-w_d/2, a/tot+0.01, f"{a/tot*100:.0f}%", ha="center", fontsize=9, color=col)
    ax.text(xi+w_d/2, d/tot+0.01, f"{d/tot*100:.0f}%", ha="center", fontsize=9, color=col)
ax.set_xticks(x_d)
ax.set_xticklabels([s[0] for s in era_sets], fontsize=10)
ax.set_ylabel("Share of tone vocabulary", **LK)
ax.set_ylim(0, 1.15)
ax.set_title("Panel D — Alarming vs. Skeptical Language by Outlet & Era\n"
             "Solid = alarming  ·  hatched = skeptical/dismissive  ·  shared era 2021–26", **TK)
legend_els = [mpatches.Patch(color="#555", alpha=0.80, label="Alarming language"),
              mpatches.Patch(color="#555", alpha=0.30, hatch="////", label="Skeptical/dismissive language")]
ax.legend(handles=legend_els, fontsize=10, framealpha=0.75)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
saved.append(save(fig, "panel_D_alarming_vs_skeptical.png"))

# ─── Panel E: Fox terminology drift ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor("#FDFDFD")
ax.axvspan(2016.5, 2020.5, alpha=0.09, color="#FF9900", label="2017–20")
ax.axvspan(2020.5, 2023.5, alpha=0.09, color="#77BB44", label="2021–23")
ax.axvspan(2023.5, 2026.5, alpha=0.09, color="#4488FF", label="2024–26")
styles = [("-", 2.3), ("--", 2.1), (":", 2.1), ("-.", 1.9)]
for (term, pat), (ls, lw) in zip(TERMS.items(), styles):
    ys_, rs_ = pyr(fox, pat)
    if ys_: ax.plot(ys_, rs_, ls=ls, lw=lw, marker="o", ms=5, label=f'"{term}"')
ax.set_xlabel("Year", **LK)
ax.set_ylabel("Mentions per 1 000 body words", **LK)
ax.set_title("Panel E — Fox: Terminology Drift Over Time\n"
             "How term choice shifts across eras", **TK)
ax.legend(fontsize=10, framealpha=0.75)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
saved.append(save(fig, "panel_E_fox_terminology.png"))

# ─── Panel F: NYT terminology drift ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor("#FDFDFD")
ax.axvspan(2020.5, 2023.5, alpha=0.09, color="#77BB44", label="2021–23")
ax.axvspan(2023.5, 2026.5, alpha=0.09, color="#4488FF", label="2024–26")
for (term, pat), (ls, lw) in zip(TERMS.items(), styles):
    ys_, rs_ = pyr(nyt, pat)
    if ys_: ax.plot(ys_, rs_, ls=ls, lw=lw, marker="o", ms=5, label=f'"{term}"')
ax.set_xlabel("Year", **LK)
ax.set_ylabel("Mentions per 1 000 body words", **LK)
ax.set_title("Panel F — NYT: Terminology Drift Over Time\n"
             "How term choice shifts across eras (NYT sample begins 2021)", **TK)
ax.legend(fontsize=10, framealpha=0.75)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
saved.append(save(fig, "panel_F_nyt_terminology.png"))

# ─── Panel G: term-rate bar chart ────────────────────────────────────────────
show_t = ["climate change", "global warming", "climate crisis",
          "hoax (any)", "con job", "denial/denier", "alarmist",
          "existential", "fake"]

TERMS_ALL = {
    "climate change": re.compile(r"climate\s+change",                         re.I),
    "global warming": re.compile(r"global\s+warming",                         re.I),
    "climate crisis": re.compile(r"climate\s+crisis",                         re.I),
    "hoax (any)":     re.compile(r"\bhoax\b",                                  re.I),
    "con job":        re.compile(r"\bcon\s+job\b",                             re.I),
    "denial/denier":  re.compile(r"\bdeni(?:al|er|ers|ed)\b",                 re.I),
    "alarmist":       re.compile(r"\balarmis[mt]\b",                          re.I),
    "existential":    re.compile(r"\bexistential\b",                           re.I),
    "fake":           re.compile(r"\bfake\b",                                  re.I),
}
def all_rates_ext(arts):
    return {t: trate(arts, p) for t, p in TERMS_ALL.items()}

group_rates = {
    "Fox 2021–23": all_rates_ext(fox_C),
    "Fox 2024–26": all_rates_ext(fox_D),
    "NYT 2021–23": all_rates_ext(nyt_C),
    "NYT 2024–26": all_rates_ext(nyt_D),
}
colours_g = [C_FOX, C_FOX, C_NYT, C_NYT]
alphas_g  = [0.50, 0.90, 0.50, 0.90]
n_t = len(show_t); x_g = np.arange(n_t)
offsets_g = np.array([-1.5, -0.5, 0.5, 1.5]) * 0.19

fig, ax = plt.subplots(figsize=(16, 7))
fig.patch.set_facecolor(BG); ax.set_facecolor("#FDFDFD")
for ki, (key, col, alph) in enumerate(zip(group_rates, colours_g, alphas_g)):
    vals = [group_rates[key].get(t, 0) for t in show_t]
    ax.bar(x_g + offsets_g[ki], vals, 0.18, color=col, alpha=alph,
           label=key, edgecolor="white", linewidth=0.4)
ax.set_xticks(x_g)
ax.set_xticklabels(show_t, fontsize=10, rotation=20, ha="right")
ax.set_ylabel("Mentions per 1 000 body words", **LK)
ax.set_title("Panel G — Term Frequency by Outlet & Era (Body Text)\n"
             "Shared comparison: 2021–23 vs 2024–26", **TK)
ax.legend(fontsize=10, ncol=4, framealpha=0.75)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
saved.append(save(fig, "panel_G_term_frequency.png"))

print(f"\nAll {len(saved)} panels saved to {OUT_DIR}/")
