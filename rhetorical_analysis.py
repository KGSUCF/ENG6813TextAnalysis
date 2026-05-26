"""
Rhetorical analysis: Fox Entertainment vs. NYT — climate coverage corpora
Two analyses:
  1. Keyness  — log-likelihood (G²) with %DIFF effect size
  2. Attribution-verb framing — "say verbs" and their epistemic register
"""

import re, os, math, collections
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
# PARSER  (same as analyze_corpora.py)
# ═══════════════════════════════════════════════════════════════════════════

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
            if len(non_blank) >= 2:
                break
    if not non_blank:          return ""
    if len(non_blank) == 1:    return non_blank[0]
    if len(non_blank[0].split()) <= 3: return non_blank[1]
    return non_blank[0]

def _extract_body(lines, start, end):
    body, skip, past_hl = [], 0, False
    hl_re  = re.compile(r"^Highlight[\s:»\xa0]", re.I)
    end_re = re.compile(
        r"^(Classification|Subject[\s:»\xa0]|Industry[\s:»\xa0]|"
        r"Person[\s:»\xa0]|Geographic[\s:»\xa0]|Load-Date:|"
        r"Organization[\s:»\xa0]|"
        # Fox legal boilerplate block
        r"Notes\s*$|"
        r"Link to the original|"
        r"Watch the clip|"
        r"The post .{5,80} first appeared on)",
        re.I,
    )
    # Fox classification metadata fragments in body
    noise_re = re.compile(
        r"^\s*(Delivered by|All Rights Reserved|"
        r"\d{3,4} words?|"
        r"(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th),\s+\d{4})",
        re.I,
    )
    for ln in lines[start:end]:
        s = ln.strip()
        if end_re.match(s): break
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
        hl = _find_headline(lines, di)
        end = date_idx[k+1][0] - 4 if k+1 < len(date_idx) else len(lines)
        body = _extract_body(lines, di + 1, end)
        if len(body) >= 80 and hl:
            arts.append({"headline": hl, "year": year, "body": body})
    return arts

print("Parsing …")
fox = parse_file(FOX_PATH)
nyt = parse_file(NYT_PATH)
print(f"Fox {len(fox)}  NYT {len(nyt)}")

# Combine all body text per corpus
fox_full = " ".join(a["body"] for a in fox)
nyt_full = " ".join(a["body"] for a in nyt)

# ═══════════════════════════════════════════════════════════════════════════
# STOPWORD LIST  (English, extended for journalism)
# ═══════════════════════════════════════════════════════════════════════════

from nltk.corpus import stopwords as _sw
BASE_STOPS = set(_sw.words("english"))

# journalism / corpus noise additions
EXTRA_STOPS = {
    # generic journalism function words
    "said", "say", "says", "also", "would", "could", "one", "two",
    "according", "told", "like", "year", "years", "new", "first",
    "last", "us", "u.s", "mr", "ms", "dr", "s", "n", "t", "re",
    "ve", "ll", "d", "m", "time", "way", "back", "get", "well",
    "many", "much", "even", "still", "going", "go", "come",
    "including", "however", "added", "noted", "states", "state",
    "percent", "per", "make", "made", "take", "put", "use", "used",
    "using", "want", "need", "people", "thing", "things", "week",
    "day", "days", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "may", "must", "also",
    # Fox legal-boilerplate leakage
    "content", "authoritative", "accordingly", "warranties",
    "guarantees", "re-distributors", "commentary", "provided",
    "delivered", "contained", "information", "stories", "story",
    "com", "www", "org", "net",
    # NYT print/database format leakage
    "print", "photograph", "photo", "page", "section", "desk",
    "edition", "final", "late",
    # named entities that inflate frequency noise
    "trump", "biden", "fox", "nyt", "times", "cnn", "abc", "news",
    "media", "kerry", "zeldin",  # persons better tracked by NER
}
# NOTE: we keep politically meaningful content words:
# "climate", "change", "global", "warming", "hoax", "crisis" etc.

STOPS = BASE_STOPS | EXTRA_STOPS

# ═══════════════════════════════════════════════════════════════════════════
# ──────────────────────────────────────────────────────────────────────────
# ANALYSIS 1: KEYNESS
# Log-likelihood G² (Dunning 1993) + %DIFF effect size
# ──────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

TOKEN_RE = re.compile(r"\b[a-z][a-z'\-]{2,}\b")   # min 3-char alpha tokens

def tokenize(text):
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPS]

fox_tokens = tokenize(fox_full)
nyt_tokens = tokenize(nyt_full)

fox_freq = collections.Counter(fox_tokens)
nyt_freq = collections.Counter(nyt_tokens)

N1 = len(fox_tokens)
N2 = len(nyt_tokens)
print(f"Tokens — Fox: {N1:,}  NYT: {N2:,}")

all_types = set(fox_freq) | set(nyt_freq)

def log_likelihood(o1, o2, n1, n2):
    """Dunning log-likelihood G²."""
    e1 = n1 * (o1 + o2) / (n1 + n2)
    e2 = n2 * (o1 + o2) / (n1 + n2)
    g2 = 0.0
    if o1 > 0: g2 += 2 * o1 * math.log(o1 / e1)
    if o2 > 0: g2 += 2 * o2 * math.log(o2 / e2)
    return g2

def pct_diff(o1, o2, n1, n2):
    """
    %DIFF effect size (Gabrielatos & Marchi 2012):
    ((norm1 - norm2) / norm2) * 100  where norms are per-million.
    Positive = over-represented in corpus 1 (Fox).
    """
    norm1 = (o1 / n1) * 1_000_000
    norm2 = (o2 / n2) * 1_000_000
    denom = norm2 if norm2 > 0 else 0.5   # avoid /0
    return ((norm1 - norm2) / denom) * 100

results = []
for w in all_types:
    o1 = fox_freq.get(w, 0)
    o2 = nyt_freq.get(w, 0)
    # require at least 5 total occurrences to reduce noise
    if o1 + o2 < 5:
        continue
    g2  = log_likelihood(o1, o2, N1, N2)
    eff = pct_diff(o1, o2, N1, N2)
    results.append((w, o1, o2, g2, eff))

# Sort by G² descending
results.sort(key=lambda x: x[3], reverse=True)

# Split into Fox-key (positive %DIFF = over in Fox) and NYT-key
fox_key = [(w, o1, o2, g2, eff) for w, o1, o2, g2, eff in results if eff > 0][:35]
nyt_key = [(w, o1, o2, g2, eff) for w, o1, o2, g2, eff in results if eff < 0][:35]

# ═══════════════════════════════════════════════════════════════════════════
# ──────────────────────────────────────────────────────────────────────────
# ANALYSIS 2: ATTRIBUTION VERB FRAMING
# ──────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

# Attribution verbs organised into five epistemic registers
ATTR_VERBS = {
    # ── Neutral / transparent ──────────────────────────────────────────────
    "neutral": [
        "say", "says", "said", "note", "notes", "noted",
        "state", "states", "stated", "tell", "tells", "told",
        "write", "writes", "wrote", "report", "reports", "reported",
        "describe", "describes", "described",
        "explain", "explains", "explained",
        "indicate", "indicates", "indicated",
    ],
    # ── Validating / authoritative ──────────────────────────────────────────
    "validating": [
        "confirm", "confirms", "confirmed",
        "show", "shows", "showed", "shown",
        "find", "finds", "found",
        "prove", "proves", "proved", "proven",
        "demonstrate", "demonstrates", "demonstrated",
        "establish", "establishes", "established",
        "conclude", "concludes", "concluded",
        "document", "documents", "documented",
        "reveal", "reveals", "revealed",
    ],
    # ── Epistemic distance / hedged ────────────────────────────────────────
    "hedged": [
        "suggest", "suggests", "suggested",
        "argue", "argues", "argued",
        "contend", "contends", "contended",
        "believe", "believes", "believed",
        "think", "thinks", "thought",
        "appear", "appears", "appeared",
        "seem", "seems", "seemed",
        "estimate", "estimates", "estimated",
    ],
    # ── Warning / urgency ──────────────────────────────────────────────────
    "warning": [
        "warn", "warns", "warned",
        "caution", "cautions", "cautioned",
        "urge", "urges", "urged",
        "fear", "fears", "feared",
        "predict", "predicts", "predicted",
        "project", "projects", "projected",
        "forecast", "forecasts", "forecasted",
        "threaten", "threatens", "threatened",
    ],
    # ── Distancing / adversarial ───────────────────────────────────────────
    "distancing": [
        "claim", "claims", "claimed",
        "allege", "alleges", "alleged",
        "insist", "insists", "insisted",
        "assert", "asserts", "asserted",
        "accuse", "accuses", "accused",
        "blame", "blames", "blamed",
        "reject", "rejects", "rejected",
        "deny", "denies", "denied",
        "dismiss", "dismisses", "dismissed",
        "attack", "attacks", "attacked",
        "slam", "slams", "slammed",
        "mock", "mocks", "mocked",
        "blast", "blasts", "blasted",
        "rip", "rips", "ripped",
        "hit", "hits",
        "call", "calls", "called",   # "called it a hoax"
        "accuse", "accuses", "accused",
    ],
}

# Build reverse lookup: verb → category
VERB_TO_CAT = {}
for cat, verbs in ATTR_VERBS.items():
    for v in verbs:
        VERB_TO_CAT[v] = cat

# Build a single regex that matches any attribution verb as a whole word
# We need to be careful to match the verb form, not substrings
ALL_VERB_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(v) for v in sorted(VERB_TO_CAT, key=len, reverse=True)) + r")\b",
    re.I,
)

def count_attr_verbs(text):
    """Return Counter of (verb_lemma, category) → count."""
    cat_counts = collections.Counter()
    verb_counts = collections.Counter()
    for m in ALL_VERB_PATTERN.finditer(text):
        v = m.group(1).lower()
        cat = VERB_TO_CAT.get(v)
        if cat:
            cat_counts[cat] += 1
            verb_counts[v] += 1
    return cat_counts, verb_counts

fox_cat, fox_verbs = count_attr_verbs(fox_full)
nyt_cat, nyt_verbs = count_attr_verbs(nyt_full)

# Normalise per 1000 words (use raw token counts as proxy for words)
fox_words = len(fox_full.split())
nyt_words = len(nyt_full.split())

def norm(count, total):
    return count / total * 1000

# Top individual verbs per corpus (normalised)
fox_verb_norm = {v: norm(c, fox_words) for v, c in fox_verbs.most_common(40)}
nyt_verb_norm = {v: norm(c, nyt_words) for v, c in nyt_verbs.most_common(40)}

all_top_verbs = set(list(fox_verbs.keys())[:25]) | set(list(nyt_verbs.keys())[:25])

# ── Keyness of attribution verbs (which verbs are over-represented?) ─────────
verb_keyness = []
for v in all_top_verbs:
    o1, o2 = fox_verbs.get(v, 0), nyt_verbs.get(v, 0)
    if o1 + o2 < 3: continue
    g2  = log_likelihood(o1, o2, fox_words, nyt_words)
    eff = pct_diff(o1, o2, fox_words, nyt_words)
    cat = VERB_TO_CAT.get(v, "?")
    verb_keyness.append((v, cat, o1, o2, g2, eff))
verb_keyness.sort(key=lambda x: x[4], reverse=True)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════

CAT_COLOURS = {
    "neutral":     "#888888",
    "validating":  "#27AE60",
    "hedged":      "#F39C12",
    "warning":     "#E74C3C",
    "distancing":  "#8E44AD",
}
CAT_LABELS = {
    "neutral":    "Neutral / transparent",
    "validating": "Validating / authoritative",
    "hedged":     "Hedged / epistemic distance",
    "warning":    "Warning / urgency",
    "distancing": "Distancing / adversarial",
}

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — KEYNESS
# ─────────────────────────────────────────────────────────────────────────────

TOP_N = 25   # words per side

fig1, axes = plt.subplots(1, 2, figsize=(20, 13))
fig1.patch.set_facecolor("#F8F6F1")

TK = dict(fontsize=11.5, fontweight="bold", color="#111")
LK = dict(fontsize=9,    color="#444")

for ax, key_list, corpus_col, corpus_lbl, side in [
    (axes[0], fox_key[:TOP_N], C_FOX, "Fox", "Fox-distinctive"),
    (axes[1], nyt_key[:TOP_N], C_NYT, "NYT", "NYT-distinctive"),
]:
    words_plot = [r[0] for r in key_list]
    g2_vals    = [r[3] for r in key_list]
    eff_vals   = [abs(r[4]) for r in key_list]  # absolute %DIFF for width

    # Sort ascending so highest G² is at top
    order   = sorted(range(len(words_plot)), key=lambda i: g2_vals[i])
    words_o = [words_plot[i] for i in order]
    g2_o    = [g2_vals[i]    for i in order]
    eff_o   = [eff_vals[i]   for i in order]

    y = np.arange(len(words_o))
    bars = ax.barh(y, g2_o, color=corpus_col, alpha=0.75, edgecolor="white",
                   linewidth=0.4)

    # Colour bars by effect size magnitude
    max_eff = max(eff_o) if eff_o else 1
    for bar, eff in zip(bars, eff_o):
        bar.set_alpha(0.35 + 0.60 * (eff / max_eff))

    ax.set_yticks(y)
    ax.set_yticklabels(words_o, fontsize=8.8)
    ax.set_xlabel("Log-likelihood G²  (higher = more statistically distinctive)", **LK)
    ax.set_title(f"Key words in {corpus_lbl} corpus\n(over-represented vs {('NYT' if corpus_lbl=='Fox' else 'Fox')})",
                 **TK)
    ax.set_facecolor("#FDFDFD")

    # Annotate %DIFF on bars
    for i, (g2v, effv) in enumerate(zip(g2_o, eff_o)):
        ax.text(g2v + max(g2_o)*0.01, i, f"+{effv:.0f}%",
                va="center", fontsize=7, color="#555")

    ax.spines[["top","right"]].set_visible(False)

fig1.suptitle(
    "Keyness Analysis — Statistically Distinctive Vocabulary\n"
    "Fox Entertainment vs. New York Times  ·  Body text  ·  "
    "Log-likelihood G² (Dunning 1993)  ·  Shade intensity = %DIFF effect size",
    fontsize=13, fontweight="bold", color="#111", y=1.01,
)
fig1.tight_layout()
path1 = os.path.join(OUT_DIR, "keyness_analysis.png")
fig1.savefig(path1, dpi=150, bbox_inches="tight", facecolor=fig1.get_facecolor())
plt.close(fig1)
print(f"Keyness figure → {path1}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — ATTRIBUTION VERB FRAMING  (3 panels)
# ─────────────────────────────────────────────────────────────────────────────

fig2 = plt.figure(figsize=(20, 18))
fig2.patch.set_facecolor("#F8F6F1")
gs2 = gridspec.GridSpec(2, 2, figure=fig2,
                        hspace=0.48, wspace=0.36,
                        top=0.92, bottom=0.05, left=0.07, right=0.97)

# ── Panel A: category totals (normalised) ────────────────────────────────────
ax2a = fig2.add_subplot(gs2[0, 0])
cats = list(CAT_COLOURS)
fox_cat_norm = [norm(fox_cat.get(c, 0), fox_words) for c in cats]
nyt_cat_norm = [norm(nyt_cat.get(c, 0), nyt_words) for c in cats]
x2a = np.arange(len(cats))
w2a = 0.35
ax2a.bar(x2a - w2a/2, fox_cat_norm, w2a, color=C_FOX, alpha=0.82, label="Fox")
ax2a.bar(x2a + w2a/2, nyt_cat_norm, w2a, color=C_NYT, alpha=0.82, label="NYT")
ax2a.set_xticks(x2a)
ax2a.set_xticklabels([CAT_LABELS[c].replace(" / ", "\n") for c in cats],
                     fontsize=8, rotation=15, ha="right")
ax2a.set_ylabel("Occurrences per 1 000 words", **LK)
ax2a.set_title("Panel A — Attribution Verb Register\n(all verbs, normalised)", **TK)
ax2a.legend(fontsize=9, framealpha=0.78)
ax2a.set_facecolor("#FDFDFD")
ax2a.spines[["top","right"]].set_visible(False)

# ── Panel B: register SHARE (stacked proportion) ─────────────────────────────
ax2b = fig2.add_subplot(gs2[0, 1])
fox_total_verbs = sum(fox_cat.values()) or 1
nyt_total_verbs = sum(nyt_cat.values()) or 1
fox_shares = [fox_cat.get(c, 0) / fox_total_verbs for c in cats]
nyt_shares = [nyt_cat.get(c, 0) / nyt_total_verbs for c in cats]

outlets  = ["Fox", "NYT"]
bottoms  = [0.0, 0.0]
for ci, cat in enumerate(cats):
    vals = [fox_shares[ci], nyt_shares[ci]]
    ax2b.bar(outlets, vals, bottom=bottoms,
             color=CAT_COLOURS[cat], alpha=0.85, label=CAT_LABELS[cat],
             edgecolor="white", linewidth=0.8)
    # label segments > 5%
    for j, (v, bot) in enumerate(zip(vals, bottoms)):
        if v > 0.04:
            ax2b.text(j, bot + v/2, f"{v*100:.1f}%",
                      ha="center", va="center", fontsize=8.5,
                      color="white", fontweight="bold")
    bottoms = [b + v for b, v in zip(bottoms, vals)]

ax2b.set_ylabel("Proportion of attribution verbs", **LK)
ax2b.set_title("Panel B — Attribution Register Share\n(proportion of all attribution verbs)", **TK)
ax2b.legend(fontsize=8.5, bbox_to_anchor=(1.01, 1), loc="upper left", framealpha=0.78)
ax2b.set_facecolor("#FDFDFD")
ax2b.spines[["top","right"]].set_visible(False)

# ── Panel C: diverging bar — top individual verbs by keyness ─────────────────
ax2c = fig2.add_subplot(gs2[1, :])

# Show top ~30 verbs that appear in both corpora, ranked by |%DIFF|
show_verbs = [(v, cat, o1, o2, g2, eff)
              for v, cat, o1, o2, g2, eff in verb_keyness
              if o1 + o2 >= 5][:30]

# Sort by %DIFF (Fox over-rep positive, NYT negative)
show_verbs.sort(key=lambda x: x[5])

verb_labels = [f"{v}  [{cat[:4]}]" for v, cat, *_ in show_verbs]
eff_vals2   = [eff for v, cat, o1, o2, g2, eff in show_verbs]
cat_cols    = [CAT_COLOURS[cat] for v, cat, *_ in show_verbs]

y2c = np.arange(len(verb_labels))
colors_bar = [C_NYT if eff < 0 else C_FOX for eff in eff_vals2]

for yi, (eff, cat_c) in enumerate(zip(eff_vals2, cat_cols)):
    ax2c.barh(yi, eff, color=cat_c, alpha=0.80, edgecolor="white", linewidth=0.4)
    # corpus label
    lbl = "← NYT" if eff < 0 else "Fox →"
    x_pos = eff - 8 if eff < 0 else eff + 4
    ax2c.text(x_pos, yi, lbl, va="center", fontsize=6.5, color="#777")

ax2c.axvline(0, color="#555", lw=1.2)
ax2c.set_yticks(y2c)
ax2c.set_yticklabels(verb_labels, fontsize=8.5)
ax2c.set_xlabel("%DIFF effect size  (positive = over-represented in Fox; "
                "negative = over-represented in NYT)", **LK)
ax2c.set_title("Panel C — Attribution Verb Keyness\n"
               "Which attribution verbs are statistically distinctive to each outlet?  "
               "Colour = verb register  (neut=grey  valid=green  hedge=amber  warn=red  dist=purple)",
               **TK)
ax2c.set_facecolor("#FDFDFD")
ax2c.spines[["top","right"]].set_visible(False)

# legend for verb categories
legend_patches = [mpatches.Patch(color=CAT_COLOURS[c], alpha=0.82,
                                  label=CAT_LABELS[c]) for c in cats]
ax2c.legend(handles=legend_patches, fontsize=8.5, loc="lower right",
            framealpha=0.78, ncol=2)

fig2.suptitle(
    "Attribution Verb Framing Analysis — Fox Entertainment vs. New York Times\n"
    "How each outlet frames the credibility and register of its sources",
    fontsize=13, fontweight="bold", color="#111", y=0.975,
)
path2 = os.path.join(OUT_DIR, "attribution_verb_framing.png")
fig2.savefig(path2, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
plt.close(fig2)
print(f"Attribution figure → {path2}")

# ═══════════════════════════════════════════════════════════════════════════
# TEXT REPORT
# ═══════════════════════════════════════════════════════════════════════════

SEP = "─" * 72

print(f"\n{SEP}")
print("KEYNESS ANALYSIS — TOP 20 DISTINCTIVE WORDS PER CORPUS")
print("(G² log-likelihood, min 5 occurrences, stopwords removed)")
print(SEP)

print(f"\n  {'WORD':<20} {'Fox':>7} {'NYT':>7}  {'G²':>9}  {'%DIFF':>9}")
print(f"  {'─'*18} {'─'*7} {'─'*7}  {'─'*9}  {'─'*9}")
print(f"\n  ── FOX-DISTINCTIVE ──")
for w, o1, o2, g2, eff in fox_key[:20]:
    print(f"  {w:<20} {o1:>7} {o2:>7}  {g2:>9.1f}  {eff:>+9.0f}%")

print(f"\n  ── NYT-DISTINCTIVE ──")
for w, o1, o2, g2, eff in nyt_key[:20]:
    print(f"  {w:<20} {o1:>7} {o2:>7}  {g2:>9.1f}  {eff:>+9.0f}%")

print(f"\n{SEP}")
print("ATTRIBUTION VERB ANALYSIS")
print(SEP)

print(f"\n  Register totals (raw counts / per-1000-words):")
print(f"  {'Register':<28} {'Fox raw':>9} {'Fox /1k':>9}  {'NYT raw':>9} {'NYT /1k':>9}")
print(f"  {'─'*26} {'─'*9} {'─'*9}  {'─'*9} {'─'*9}")
for cat in cats:
    fc, nc = fox_cat.get(cat, 0), nyt_cat.get(cat, 0)
    print(f"  {CAT_LABELS[cat]:<28} {fc:>9}  {norm(fc,fox_words):>8.2f}  "
          f"{nc:>9}  {norm(nc,nyt_words):>8.2f}")

print(f"\n  Attribution verb register share:")
for cat in cats:
    fs = fox_cat.get(cat, 0) / fox_total_verbs
    ns = nyt_cat.get(cat, 0) / nyt_total_verbs
    print(f"  {CAT_LABELS[cat]:<28}  Fox={fs*100:5.1f}%   NYT={ns*100:5.1f}%   "
          f"Δ={ns*100-fs*100:+.1f}pp")

print(f"\n  Top 15 individual verbs by keyness (|%DIFF|):")
print(f"  {'Verb':<16} {'Cat':<12} {'Fox /1k':>9} {'NYT /1k':>9}  {'G²':>8}  {'%DIFF':>9}")
print(f"  {'─'*14} {'─'*12} {'─'*9} {'─'*9}  {'─'*8}  {'─'*9}")
for v, cat, o1, o2, g2, eff in verb_keyness[:15]:
    f_n = norm(o1, fox_words)
    n_n = norm(o2, nyt_words)
    print(f"  {v:<16} {cat:<12} {f_n:>9.3f} {n_n:>9.3f}  {g2:>8.1f}  {eff:>+9.0f}%")

print(f"\n{SEP}\nDone.\n")
