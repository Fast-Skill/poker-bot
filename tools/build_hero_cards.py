"""Build rank and suit templates at the size the client draws the hero's cards.

The hero's own pair is drawn about a sixth larger than the board's - its rank
glyph stands 34 pixels tall against the board's 29 - so the board templates are
the wrong shape for it. Matching them anyway produced distances of 37 to 43
where a sound match scores under 20, and on one frame an eight scored nearer
the six template than its own.

As everywhere else here, glyphs are stored at their own natural size and never
resized, keyed by that size, and an unlabelled class produces no template so
that anything shaped like it is refused rather than guessed at.
"""
import collections, glob, os, struct, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harvest_hero_cards as H

OUT = r"C:\poker\data\hero_cards.bin"
MAGIC, VERSION = b"PKHC", 1
SKIP = "_"

# Read off captures/_hero/ranks-labelled.png and pips-labelled.png, in the same
# order those sheets were drawn: by descending sample count within each size.
RANKS = "5J89" + "10" + "26AQJ"          # one entry per class, "10" is one class
RANK_CLASSES = ["5", "J", "8", "9", "10", "2", "6", "A", "Q", "J"]
# Classes 6..8 are pips clipped by the overlapping card into shapes that no
# longer say which suit they are, and they carry two to four samples each.
SUIT_CLASSES = ["s", "d", "s", "c", "h", "h", SKIP, SKIP, SKIP]


def cluster(patches, tol=0.10):
    reps = []
    for p in patches:
        v = p.astype(np.float64)
        for r in reps:
            if np.abs(v - r["sum"] / r["n"]).mean() < tol:
                r["sum"] += v
                r["n"] += 1
                break
        else:
            reps.append({"sum": v.copy(), "n": 1})
    reps.sort(key=lambda r: -r["n"])
    return [r for r in reps if r["n"] >= 2]


def collect():
    ranks, pips = collections.defaultdict(list), collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(H.CAPTURES, "*.png"))):
        for card in H.hero_cards(path):
            ranks[card["rank"][0]].append(card["rank"][1])
            pips[(card["pip"][0], card["pip"][2])].append(card["pip"][1])
    return ranks, pips


def classes(bank):
    out = []
    for key in sorted(bank, key=lambda k: -len(bank[k])):
        for rep in cluster(bank[key]):
            out.append((key, rep))
    return out


if __name__ == "__main__":
    rank_bank, pip_bank = collect()
    rank_cells, pip_cells = classes(rank_bank), classes(pip_bank)
    if len(rank_cells) != len(RANK_CLASSES) or len(pip_cells) != len(SUIT_CLASSES):
        raise SystemExit(f"clustered {len(rank_cells)} ranks and {len(pip_cells)} suits, "
                         f"but {len(RANK_CLASSES)} and {len(SUIT_CLASSES)} labels are given; "
                         f"re-check the sheets in captures/_hero")

    ranks, suits = [], []
    for (size, rep), label in zip(rank_cells, RANK_CLASSES):
        if label == SKIP:
            continue
        mask = ((rep["sum"] / rep["n"]) > 0.5).astype(np.uint8) * 255
        ranks.append((label, size[0], size[1], mask, rep["n"]))
    for (key, rep), label in zip(pip_cells, SUIT_CLASSES):
        if label == SKIP:
            continue
        mask = ((rep["sum"] / rep["n"]) > 0.5).astype(np.uint8) * 255
        suits.append((label, key[1], key[0][0], key[0][1], mask, rep["n"]))

    with open(OUT, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", VERSION, len(ranks)))
        for label, h, w, mask, _ in ranks:
            f.write(struct.pack("<B", len(label))); f.write(label.encode())
            f.write(struct.pack("<II", h, w)); f.write(mask.tobytes())
        f.write(struct.pack("<I", len(suits)))
        for label, red, h, w, mask, _ in suits:
            f.write(struct.pack("<B", len(label))); f.write(label.encode())
            f.write(struct.pack("<BII", 1 if red else 0, h, w)); f.write(mask.tobytes())

    covered = sorted({l for l, *_ in ranks}, key="23456789 10JQKA".find)
    missing = [r for r in ["2","3","4","5","6","7","8","9","10","J","Q","K","A"]
               if r not in covered]
    print(f"wrote {OUT}: {len(ranks)} rank templates, {len(suits)} suit templates, "
          f"{os.path.getsize(OUT)} bytes")
    for label, h, w, _, n in ranks:
        print(f"   rank {label:>2}  {h}x{w}  from {n} samples")
    for label, red, h, w, _, n in suits:
        print(f"   suit {label}   {h}x{w}  red={red}  from {n} samples")
    print(f"\nranks covered: {' '.join(covered)}")
    print(f"ranks MISSING: {' '.join(missing) if missing else 'none'}")
