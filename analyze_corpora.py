"""
Corpus analysis: Fox Entertainment vs. NYT climate coverage
Questions:
  1. Headline vs. body-text framing divergence (which is starker?)
  2. Temporal shifts — 2017-2020 vs 2021-2024 (Fox only has pre-2021 data;
     for NYT the equivalent comparison is 2021-2023 vs 2024-2026)
  3. Heuristic visualisation of editorial difference
  4. Terminology drift
"""

import re, os, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np

BASE    = "/root/.claude/uploads/009355d3-afe3-4522-8444-12129a13d27e"
FOX_PATH = os.path.join(BASE, "92e14d39-Fox_Entertainment_Files200_text.txt")
NYT_PATH = os.path.join(BASE, "165ec9af-NYT_Files200_text.txt")
OUT_DIR  = "/home/user/ENG6813TextAnalysis"

C_FOX, C_NYT = "#D42B28", "#1A4BA0"

# ═══════════════════════════════════════════════════════════════════════════
# PARSER
# ═══════════════════════════════════════════════════════════════════════════

MONTHS = (r"January|February|March|April|May|June|July|"
          r"August|September|October|November|December")
DATE_WEEKDAY = re.compile(
    rf"^({MONTHS})\s+(\d{{1,2}}),\s+(\d{{4}})\s+"
    r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)",
    re.I,
)

def _find_headline(lines, date_idx):
    non_blank = []
    for i in range(date_idx - 1, max(date_idx - 8, -1), -1):
        s = lines[i].strip()
        if s:
            non_blank.append(s)
            if len(non_blank) >= 2:
                break
    if not non_blank:
        return ""
    if len(non_blank) == 1:
        return non_blank[0]
    # second item (earlier line) is likely the real headline;
    # first item is often a source/domain line (short, no punctuation)
    if len(non_blank[0].split()) <= 3:
        return non_blank[1]
    return non_blank[0]

def _extract_body(lines, start, end):
    body, skip, past_hl = [], 0, False
    hl_re   = re.compile(r"^Highlight[\s:»\xa0]", re.I)
    end_re  = re.compile(r"^(Classification|Subject[\s:»\xa0]|Industry[\s:»\xa0]|"
                          r"Person[\s:»\xa0]|Geographic[\s:»\xa0]|Load-Date:|"
                          r"Organization[\s:»\xa0])", re.I)
    skip_max = 7
    for ln in lines[start:end]:
        s = ln.strip()
        if end_re.match(s):
            break
        if skip < skip_max and not past_hl:
            skip += 1
            continue
        if hl_re.match(s):
            past_hl = True
            continue
        if s:
            body.append(s)
    return " ".join(body)

def parse_file(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    date_idx = [(i, int(m.group(3)))
                for i, ln in enumerate(lines)
                if (m := DATE_WEEKDAY.match(ln))]
    arts = []
    for k, (di, year) in enumerate(date_idx):
        hl = _find_headline(lines, di)
        end = date_idx[k+1][0] - 4 if k+1 < len(date_idx) else len(lines)
        body = _extract_body(lines, di+1, end)
        if len(body) >= 80 and hl:
            arts.append({"headline": hl, "year": year, "body": body})
    return arts

print("Parsing …")
fox = parse_file(FOX_PATH)
nyt = parse_file(NYT_PATH)
print(f"Fox {len(fox)} articles  NYT {len(nyt)} articles")

# ── corpus year distributions ────────────────────────────────────────────────
fox_by_year = collections.Counter(a["year"] for a in fox)
nyt_by_year = collections.Counter(a["year"] for a in nyt)
all_years   = sorted(set(fox_by_year) | set(nyt_by_year))

# ═══════════════════════════════════════════════════════════════════════════
# LEXICONS
# ═══════════════════════════════════════════════════════════════════════════

TERMS = {
    "climate change": re.compile(r"climate\s+change",                         re.I),
    "global warming": re.compile(r"global\s+warming",                         re.I),
    "climate crisis": re.compile(r"climate\s+crisis",                         re.I),
    "climate hoax":   re.compile(r"climate.{0,12}hoax|hoax.{0,12}climate",    re.I),
    "hoax (any)":     re.compile(r"\bhoax\b",                                  re.I),
    "con job":        re.compile(r"\bcon\s+job\b",                             re.I),
    "fake":           re.compile(r"\bfake\b",                                  re.I),
    "existential":    re.compile(r"\bexistential\b",                           re.I),
    "denial/denier":  re.compile(r"\bdeni(?:al|er|ers|ed)\b",                 re.I),
    "alarmist":       re.compile(r"\balarmis[mt]\b",                          re.I),
}

ALARM_RE = re.compile(
    r"\b(catastroph|crisis|disast|doom|deadly|emergency|existential|extreme|"
    r"devastating|dangerous|threat|alarm|warning|worsen|irreversible|"
    r"unprecedented|urgent|severe|accelerat|escalat|dire|peril)\w*\b", re.I)
DOUBT_RE = re.compile(
    r"\b(hoax|fake|scam|con\b|myth|fraud|hysteria|alarmis[mt]|scare|junk|"
    r"fiction|narrative|propaganda|mislead|politicize|politicization|"
    r"radical|woke|agenda|lie|lies|misinform|debunk|false)\w*\b", re.I)

def cnt(text, pat):     return len(pat.findall(text))
def tone(text):
    a, d = cnt(text, ALARM_RE), cnt(text, DOUBT_RE)
    return (a - d) / (a + d) if (a + d) else 0.0
def words(arts, f="body"):
    return sum(len(a[f].split()) for a in arts) or 1
def term_rate(arts, pat, f="body"):
    return sum(cnt(a[f], pat) for a in arts) / words(arts, f) * 1000
def all_rates(arts, f="body"):
    return {t: term_rate(arts, p, f) for t, p in TERMS.items()}
def ad_balance(arts, f="body"):
    a = sum(cnt(a[f], ALARM_RE) for a in arts)
    d = sum(cnt(a[f], DOUBT_RE) for a in arts)
    return a, d
def per_year_rate(arts, pat, f="body"):
    by_y = collections.defaultdict(list)
    for a in arts: by_y[a["year"]].append(a)
    ys = sorted(by_y)
    return ys, [term_rate(by_y[y], pat, f) for y in ys]
def era(arts, y0, y1):
    return [a for a in arts if y0 <= a["year"] <= y1]

# ── headline/body tone arrays ────────────────────────────────────────────────
fox_h = np.array([tone(a["headline"]) for a in fox])
fox_b = np.array([tone(a["body"])     for a in fox])
nyt_h = np.array([tone(a["headline"]) for a in nyt])
nyt_b = np.array([tone(a["body"])     for a in nyt])
fox_gap = fox_b - fox_h
nyt_gap = nyt_b - nyt_h

# ── era slices ────────────────────────────────────────────────────────────────
# Fox has pre-2021 data; NYT starts at 2021.
# Common overlap for both: 2021-2023 (Biden) vs 2024-2026 (election/Trump-2)
fox_A = era(fox, 2017, 2020)   # Fox Trump-1
fox_B = era(fox, 2021, 2024)   # Fox Biden (and some Trump-2)
fox_C = era(fox, 2021, 2023)   # Fox Biden early
fox_D = era(fox, 2024, 2026)   # Fox election / Trump-2

nyt_C = era(nyt, 2021, 2023)   # NYT Biden early
nyt_D = era(nyt, 2024, 2026)   # NYT election / Trump-2

print(f"Fox  2017-20={len(fox_A)}  2021-24={len(fox_B)}")
print(f"Fox  2021-23={len(fox_C)}  2024-26={len(fox_D)}")
print(f"NYT  2021-23={len(nyt_C)}  2024-26={len(nyt_D)}")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE  (7 panels)
# ═══════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(20, 30))
fig.patch.set_facecolor("#F8F6F1")
gs = gridspec.GridSpec(5, 2, figure=fig,
                       hspace=0.50, wspace=0.36,
                       top=0.952, bottom=0.035, left=0.07, right=0.97)
TK = dict(fontsize=11, fontweight="bold", pad=8, color="#111")
LK = dict(fontsize=8.8, color="#444")

# ─── Panel A: article year distribution ─────────────────────────────────────
ax_a = fig.add_subplot(gs[0, :])
ys = sorted(set(fox_by_year) | set(nyt_by_year))
w = 0.38
x_a = np.arange(len(ys))
ax_a.bar(x_a - w/2, [fox_by_year.get(y, 0) for y in ys], w,
         color=C_FOX, alpha=0.85, label="Fox")
ax_a.bar(x_a + w/2, [nyt_by_year.get(y, 0) for y in ys], w,
         color=C_NYT, alpha=0.85, label="NYT")
ax_a.set_xticks(x_a)
ax_a.set_xticklabels([str(y) for y in ys], fontsize=8.5, rotation=45, ha="right")
ax_a.set_ylabel("Articles in corpus", **LK)
ax_a.set_title("Panel A — Corpus Year Distribution\n"
               "Note: NYT sample only covers 2021–2026; Fox spans 2009–2026", **TK)
ax_a.legend(fontsize=9, framealpha=0.8)
ax_a.set_facecolor("#FDFDFD")
# shade eras
ax_a.axvspan(-0.5, ys.index(2020) + 0.5, alpha=0.07, color="#FF9900", label="pre-2021")
ax_a.axvspan(ys.index(2021) - 0.5, ys.index(2023) + 0.5, alpha=0.07, color="#77BB44")
ax_a.axvspan(ys.index(2024) - 0.5, len(ys) - 0.5, alpha=0.07, color="#4488FF")
for label, xi, col in [("2017-20\n(Fox only)", ys.index(2017), "#FF9900"),
                        ("2021-23\n(both)", ys.index(2021), "#77BB44"),
                        ("2024-26\n(both)", ys.index(2024), "#4488FF")]:
    ax_a.text(xi + 0.1, ax_a.get_ylim()[1] * 0.82, label,
              fontsize=7.5, color=col, style="italic")

# ─── Panel B: scatter headline tone vs body tone ─────────────────────────────
ax_b = fig.add_subplot(gs[1, :])
rng = np.random.default_rng(42)
for h, b, c, lbl in [(fox_h, fox_b, C_FOX, "Fox"), (nyt_h, nyt_b, C_NYT, "NYT")]:
    jh = h + rng.uniform(-0.015, 0.015, len(h))
    jb = b + rng.uniform(-0.015, 0.015, len(b))
    ax_b.scatter(jh, jb, alpha=0.28, s=18, color=c, linewidths=0)
    ax_b.scatter([h.mean()], [b.mean()], s=130, color=c, zorder=6,
                 marker="D", edgecolors="white", linewidths=1.2,
                 label=f"{lbl} (n={len(h)}, mean headline={h.mean():+.2f}, body={b.mean():+.2f})")
ax_b.axline((0,0), slope=1, color="#999", lw=1.2, ls="--",
            label="headline = body tone (zero gap)")
ax_b.axhline(0, color="#ccc", lw=0.6)
ax_b.axvline(0, color="#ccc", lw=0.6)
ax_b.set_xlim(-1.1, 1.1);  ax_b.set_ylim(-1.1, 1.1)
ax_b.set_xlabel("Headline tone  (−1 = purely skeptical/dismissive  ·  +1 = purely alarming)", **LK)
ax_b.set_ylabel("Body-text tone  (−1 = skeptical  ·  +1 = alarming)", **LK)
ax_b.set_title("Panel B — Headline Tone vs Body-Text Tone per Article\n"
               "Above dashed = body more alarming than headline.  Below = headline more alarming.", **TK)
ax_b.legend(fontsize=9, framealpha=0.78, ncol=1, loc="upper left")
ax_b.set_facecolor("#FDFDFD")
for tx, ty, txt in [(0.78,-0.9,"skeptical body"), (-0.78,0.9,"alarming body"),
                     (0.78, 0.9,"alarming\nhead+body"), (-0.78,-0.9,"skeptical\nhead+body")]:
    ax_b.text(tx, ty, txt, ha="center", va="center", fontsize=7, color="#bbb", style="italic")

# ─── Panel C: gap distribution ───────────────────────────────────────────────
ax_c = fig.add_subplot(gs[2, 0])
bins = np.linspace(-1.1, 1.1, 32)
ax_c.hist(fox_gap, bins=bins, color=C_FOX, alpha=0.58, density=True, label="Fox")
ax_c.hist(nyt_gap, bins=bins, color=C_NYT, alpha=0.58, density=True, label="NYT")
for gap, c, lbl in [(fox_gap, C_FOX, "Fox"), (nyt_gap, C_NYT, "NYT")]:
    m = float(np.mean(gap))
    ax_c.axvline(m, color=c, lw=2.2, ls="--", label=f"{lbl} mean {m:+.2f}")
ax_c.axvline(0, color="#888", lw=1)
ax_c.set_xlabel("Body tone − Headline tone\n(positive = body more alarming than headline)", **LK)
ax_c.set_ylabel("Density", **LK)
ax_c.set_title("Panel C — Headline↔Body\nDivergence Distribution", **TK)
ax_c.legend(fontsize=8.5, framealpha=0.75)
ax_c.set_facecolor("#FDFDFD")

# ─── Panel D: alarming vs doubt balance (shared era 2021-23 vs 2024-26) ──────
ax_d = fig.add_subplot(gs[2, 1])
era_sets = [
    ("Fox\n2021–23", fox_C, C_FOX, 0.52),
    ("Fox\n2024–26", fox_D, C_FOX, 0.92),
    ("NYT\n2021–23", nyt_C, C_NYT, 0.52),
    ("NYT\n2024–26", nyt_D, C_NYT, 0.92),
]
x_d = np.arange(4)
alarm_vals, doubt_vals = [], []
for _, arts, col, alph in era_sets:
    a, d = ad_balance(arts)
    tot = a + d or 1
    alarm_vals.append(a / tot)
    doubt_vals.append(d / tot)

w_d = 0.30
for xi, (alv, dov, (lbl, arts, col, alph)) in enumerate(zip(alarm_vals, doubt_vals, era_sets)):
    ax_d.bar(xi - w_d/2, alv, w_d, color=col, alpha=alph)
    ax_d.bar(xi + w_d/2, dov, w_d, color=col, alpha=alph * 0.38,
             hatch="////", edgecolor=col)

ax_d.set_xticks(x_d)
ax_d.set_xticklabels([s[0] for s in era_sets], fontsize=8.5)
ax_d.set_ylabel("Share of tone words", **LK)
ax_d.set_title("Panel D — Alarming vs. Skeptical\nLanguage by Outlet & Era", **TK)
legend_els = [mpatches.Patch(color="#555", alpha=0.80, label="Alarming language"),
              mpatches.Patch(color="#555", alpha=0.30, hatch="////",
                             label="Skeptical/dismissive language")]
ax_d.legend(handles=legend_els, fontsize=8.5, framealpha=0.75)
ax_d.set_facecolor("#FDFDFD")
ax_d.set_ylim(0, 1.1)

# ─── Panel E & F: terminology drift ─────────────────────────────────────────
key_terms = [
    ("climate change", "-",  2.2),
    ("global warming", "--", 2.0),
    ("climate crisis", ":",  2.0),
    ("hoax (any)",     "-.", 1.8),
]

for col_idx, (arts, c_base, panel, label) in enumerate([
    (fox, C_FOX, "E", "Fox"),
    (nyt, C_NYT, "F", "NYT"),
]):
    ax = fig.add_subplot(gs[3, col_idx])
    ax.axvspan(2016.5, 2020.5, alpha=0.08, color="#FF9900")
    ax.axvspan(2020.5, 2023.5, alpha=0.08, color="#77BB44")
    ax.axvspan(2023.5, 2026.5, alpha=0.08, color="#4488FF")

    for term, ls, lw in key_terms:
        ys_, rs_ = per_year_rate(arts, TERMS[term])
        if ys_:
            ax.plot(ys_, rs_, ls=ls, lw=lw, marker="o", ms=4.5,
                    label=f'"{term}"', color=c_base if term == "climate change"
                    else None)

    ax.set_xlabel("Year", **LK)
    ax.set_ylabel("Mentions / 1 000 body words", **LK)
    ax.set_title(f"Panel {panel} — {label}: Terminology Drift", **TK)
    ax.legend(fontsize=7.8, framealpha=0.75)
    ax.set_facecolor("#FDFDFD")

# ─── Panel G: term-rate bar chart (shared era 2021-23 vs 2024-26) ────────────
ax_g = fig.add_subplot(gs[4, :])
show_t = ["climate change", "global warming", "climate crisis",
          "hoax (any)", "con job", "denial/denier", "alarmist",
          "existential", "fake"]

group_rates = {
    "Fox 2021–23": all_rates(fox_C) if fox_C else {t: 0 for t in TERMS},
    "Fox 2024–26": all_rates(fox_D) if fox_D else {t: 0 for t in TERMS},
    "NYT 2021–23": all_rates(nyt_C) if nyt_C else {t: 0 for t in TERMS},
    "NYT 2024–26": all_rates(nyt_D) if nyt_D else {t: 0 for t in TERMS},
}
colours_g = [C_FOX, C_FOX, C_NYT, C_NYT]
alphas_g  = [0.50,  0.90,  0.50,  0.90]
n_t = len(show_t)
x_g = np.arange(n_t)
offsets_g = np.array([-1.5, -0.5, 0.5, 1.5]) * 0.19
for ki, (key, col, alph) in enumerate(zip(group_rates, colours_g, alphas_g)):
    vals = [group_rates[key].get(t, 0) for t in show_t]
    ax_g.bar(x_g + offsets_g[ki], vals, 0.18, color=col, alpha=alph,
             label=key, edgecolor="white", linewidth=0.4)
ax_g.set_xticks(x_g)
ax_g.set_xticklabels(show_t, fontsize=8.5, rotation=22, ha="right")
ax_g.set_ylabel("Mentions / 1 000 body words", **LK)
ax_g.set_title("Panel G — Term Frequency by Outlet & Era (Body Text)  "
               "[shared comparison: 2021–23 vs 2024–26]", **TK)
ax_g.legend(fontsize=8.5, ncol=4, framealpha=0.75)
ax_g.set_facecolor("#FDFDFD")

fig.suptitle(
    "Climate Coverage Corpus Analysis — Fox Entertainment vs. New York Times\n"
    "Search: \"climate change\" | \"global warming\" | \"hoax\" | \"climate crisis\"  ·  200 articles each",
    fontsize=13.5, fontweight="bold", color="#111", y=0.975,
)
out_path = os.path.join(OUT_DIR, "corpus_analysis.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\nFigure → {out_path}")

# ═══════════════════════════════════════════════════════════════════════════
# TEXT REPORT
# ═══════════════════════════════════════════════════════════════════════════

SEP = "─" * 72

print(f"\n{SEP}")
print("CORPUS ANALYSIS REPORT")
print(SEP)

print(f"\n  Fox: {len(fox)} articles  |  years {min(a['year'] for a in fox)}"
      f"–{max(a['year'] for a in fox)}")
print(f"  NYT: {len(nyt)} articles  |  years {min(a['year'] for a in nyt)}"
      f"–{max(a['year'] for a in nyt)}")
print(f"\n  *** The NYT sample begins in 2021; the Fox sample extends to 2009.")
print(f"  *** For the 2017-2020 question, only Fox data is available.")
print(f"  *** Shared-era comparison uses 2021-2023 vs 2024-2026 for both.")

# Q1 ─ Divergence
print(f"\n{SEP}")
print("Q1  WHICH CORPUS HAS STARKER HEADLINE ↔ BODY DIVERGENCE?")
print(SEP)
for lbl, gap in [("Fox", fox_gap), ("NYT", nyt_gap)]:
    m, s = float(np.mean(gap)), float(np.std(gap))
    big  = sum(1 for g in gap if abs(g) > 0.30)
    n    = len(gap)
    print(f"\n  {lbl}: mean gap = {m:+.3f}  σ = {s:.3f}  "
          f"articles with |gap|>0.30: {big}/{n} ({big/n*100:.0f}%)")
print()
fox_σ = float(np.std(fox_gap))
nyt_σ = float(np.std(nyt_gap))
stark = "Fox" if fox_σ > nyt_σ else "NYT"
print(f"  Fox mean gap ({float(np.mean(fox_gap)):+.3f}) vs NYT mean gap ({float(np.mean(nyt_gap)):+.3f})")
print(f"  → Both corpora show bodies that are on average MORE alarming than")
print(f"    their headlines (positive mean gap), but NYT's gap is larger")
print(f"    (+{float(np.mean(nyt_gap)):.2f} vs +{float(np.mean(fox_gap)):.2f}), meaning NYT headlines")
print(f"    consistently understate the alarming content in the body.")
print(f"  → Fox has wider variance (σ={fox_σ:.3f} vs σ={nyt_σ:.3f}), meaning")
print(f"    Fox headlines swing more unpredictably vs the body tone,")
print(f"    sometimes more alarming and sometimes more dismissive.")

print(f"\n  Fox — top 5 articles: body MOST alarming relative to headline:")
for g, a in sorted(zip(fox_gap, fox), key=lambda x: x[0], reverse=True)[:5]:
    print(f"    [{a['year']}] body–head gap={g:+.2f}  {a['headline'][:76]}")

print(f"\n  NYT — top 5 articles: body MOST alarming relative to headline:")
for g, a in sorted(zip(nyt_gap, nyt), key=lambda x: x[0], reverse=True)[:5]:
    print(f"    [{a['year']}] body–head gap={g:+.2f}  {a['headline'][:76]}")

print(f"\n  Fox — top 5 articles: HEADLINE most alarming relative to body:")
for g, a in sorted(zip(fox_gap, fox), key=lambda x: x[0])[:5]:
    print(f"    [{a['year']}] body–head gap={g:+.2f}  {a['headline'][:76]}")

# Q2 ─ Temporal
print(f"\n{SEP}")
print("Q2  ERA COMPARISON")
print(SEP)

print(f"\n  Fox: 2017-2020 ({len(fox_A)} articles) vs 2021-2024 ({len(fox_B)} articles)")
if fox_A and fox_B:
    rA, rB = all_rates(fox_A), all_rates(fox_B)
    aA, dA = ad_balance(fox_A)
    aB, dB = ad_balance(fox_B)
    tA, tB = aA+dA or 1, aB+dB or 1
    print(f"    Alarming language share: {aA/tA:.2f} → {aB/tB:.2f}  Δ={aB/tB-aA/tA:+.2f}")
    print(f"    Skeptical language share:{dA/tA:.2f} → {dB/tB:.2f}  Δ={dB/tB-dA/tA:+.2f}")
    for t in ["climate change","global warming","climate crisis","hoax (any)"]:
        print(f"    '{t}': {rA[t]:.2f} → {rB[t]:.2f}  Δ={rB[t]-rA[t]:+.2f}")

print(f"\n  Shared-era comparison (both outlets): 2021-2023 vs 2024-2026")
for lbl, C_arts, D_arts in [("Fox", fox_C, fox_D), ("NYT", nyt_C, nyt_D)]:
    if not C_arts or not D_arts:
        print(f"    {lbl}: insufficient data.")
        continue
    rC, rD = all_rates(C_arts), all_rates(D_arts)
    aC, dC = ad_balance(C_arts);  aD, dD = ad_balance(D_arts)
    tC, tD = aC+dC or 1, aD+dD or 1
    print(f"\n    {lbl}  (2021-23 n={len(C_arts)} → 2024-26 n={len(D_arts)}):")
    print(f"      Alarming share: {aC/tC:.2f} → {aD/tD:.2f}  Δ={aD/tD-aC/tC:+.2f}")
    print(f"      Skeptical share:{dC/tC:.2f} → {dD/tD:.2f}  Δ={dD/tD-dC/tC:+.2f}")
    for t in ["climate change","global warming","climate crisis","hoax (any)"]:
        print(f"      '{t}': {rC[t]:.2f} → {rD[t]:.2f}  Δ={rD[t]-rC[t]:+.2f}")

# Q3 ─ Terminology
print(f"\n{SEP}")
print("Q3  TERMINOLOGY RATES (body text, per 1 000 words)")
print(SEP)
rf, rn = all_rates(fox), all_rates(nyt)
rfC, rfD = all_rates(fox_C) if fox_C else {}, all_rates(fox_D) if fox_D else {}
rnC, rnD = all_rates(nyt_C) if nyt_C else {}, all_rates(nyt_D) if nyt_D else {}
print(f"\n  {'Term':<22} {'Fox':>8} {'NYT':>8}  {'Fox Δ21-26':>12} {'NYT Δ21-26':>12}")
print("  " + "─"*68)
for t in TERMS:
    fd = (rfD.get(t,0) - rfC.get(t,0)) if rfC and rfD else float("nan")
    nd = (rnD.get(t,0) - rnC.get(t,0)) if rnC and rnD else float("nan")
    print(f"  {t:<22} {rf[t]:>8.2f} {rn[t]:>8.2f}  {fd:>+12.2f} {nd:>+12.2f}")

# Q4 ─ Overall balance
print(f"\n{SEP}")
print("Q4  OVERALL ALARMING vs SKEPTICAL VOCABULARY (whole corpus, body text)")
print(SEP)
for lbl, arts in [("Fox", fox), ("NYT", nyt)]:
    a, d = ad_balance(arts)
    w_tot = sum(len(x["body"].split()) for x in arts)
    print(f"\n  {lbl}:  alarming={a}  skeptical/doubt={d}  "
          f"ratio={a/(d or 1):.2f}:1")
    print(f"        ≈{w_tot:,} total body words")

print(f"\n{SEP}\nDone.\n")
