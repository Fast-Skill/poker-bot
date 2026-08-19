"""Find the hero's own cards across the captures, ready for templating.

The hero's pair is drawn overlapping, larger than board cards and at a slight
tilt, so there is no card rectangle to measure glyph offsets from. The corner
suit pip is the anchor instead: one solid shape, on a card face, with its rank
directly above it.

Rank and suit are collected at their own natural sizes and never resized, the
same rule the board and digit templates follow.
"""
import glob, os
import numpy as np
from PIL import Image
from scipy import ndimage

CAPTURES = r"C:\poker\captures"
# The hero's seat is always the bottom middle of the window, so unlike the
# other seats this one region holds for every table size.
REGION = (560, 730, 880, 900)
PIP = (22, 40)
INK = 110.0


def card_face(a):
    lo = a.min(axis=2); hi = a.max(axis=2)
    return (lo > 110) & (hi - lo < 60)


def hero_cards(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    luma = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    face = card_face(a)
    x0, y0, x1, y1 = REGION
    ink = np.zeros(luma.shape, bool)
    ink[y0:y1, x0:x1] = luma[y0:y1, x0:x1] < INK

    labels, _ = ndimage.label(ink)
    found = []
    for sl in ndimage.find_objects(labels):
        ys, xs = sl
        h, w = ys.stop - ys.start, xs.stop - xs.start
        if not (PIP[0] <= w <= PIP[1] and PIP[0] <= h <= PIP[1]):
            continue
        # A countdown badge is pip-shaped; a card face behind it is what a pip
        # actually has.
        if face[max(0, ys.start - 8):ys.stop + 8, max(0, xs.start - 8):xs.stop + 8].mean() <= 0.40:
            continue

        pip = ink[ys, xs]
        red = (a[ys, xs, 0] - a[ys, xs, 2])[pip].mean() > 20

        # The rank sits directly above, within the same card.
        ry0, ry1 = ys.start - 45, ys.start - 3
        rx0, rx1 = xs.start - 20, xs.start + 35
        zone = ink[ry0:ry1, rx0:rx1]
        zone_face = face[ry0:ry1, rx0:rx1]
        marks, _ = ndimage.label(zone)
        parts = []
        for s in ndimage.find_objects(marks):
            ph, pw = s[0].stop - s[0].start, s[1].stop - s[1].start
            # A rank glyph is tall, narrow, and printed on the card. Ink that
            # runs to the edge of the search zone left the card entirely.
            if 25 <= ph <= 38 and 4 <= pw <= 30 and zone_face[s].size and                zone_face[max(0, s[0].start - 3):s[0].stop + 3,
                         max(0, s[1].start - 3):s[1].stop + 3].mean() > 0.35:
                parts.append(s)
        if not parts:
            continue
        # A ten is drawn as two glyphs; take them together as one rank.
        top = min(s[0].start for s in parts)
        bottom = max(s[0].stop for s in parts)
        left = min(s[1].start for s in parts)
        right = max(s[1].stop for s in parts)
        if bottom - top > 38 or right - left > 34:
            continue
        rank = zone[top:bottom, left:right]
        found.append({
            "pip": (pip.shape, pip.copy(), bool(red)),
            "rank": (rank.shape, rank.copy()),
            "at": (xs.start, ys.start),
        })
    return sorted(found, key=lambda f: f["at"][0])


if __name__ == "__main__":
    import collections
    pips = collections.Counter()
    ranks = collections.Counter()
    per_frame = collections.Counter()
    frames = sorted(glob.glob(os.path.join(CAPTURES, "*.png")))
    for path in frames:
        cards = hero_cards(path)
        per_frame[len(cards)] += 1
        for card in cards:
            pips[(card["pip"][0], card["pip"][2])] += 1
            ranks[card["rank"][0]] += 1
    print(f"{len(frames)} frames; cards found per frame: {dict(sorted(per_frame.items()))}")
    print("\npip sizes (height, width), red:")
    for key, n in sorted(pips.items(), key=lambda kv: -kv[1])[:14]:
        print(f"   {key[0]}  red={key[1]}  {n}")
    print("\nrank sizes (height, width):")
    for key, n in sorted(ranks.items(), key=lambda kv: -kv[1])[:14]:
        print(f"   {key}  {n}")
