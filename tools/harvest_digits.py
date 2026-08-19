"""Harvest digit glyphs from captured frames, grouped by colour and native height.

Scaling glyphs to a common box destroys them: resampling a 9px glyph up to 22px
produces interpolation artifacts that a natively-22px glyph does not share, and
clustering then splits one character into dozens of classes. Cards never hit
this because they are always exactly 103x156. So nothing here is ever resized -
glyphs are grouped by exact pixel height and compared only within a group.
"""
import sys, glob, os, json
import numpy as np
from PIL import Image
from scipy import ndimage

CAPTURES = r"C:\poker\captures"
OUT = os.path.join(CAPTURES, "_glyphs")

# Measured on the captures: the client colour-codes its readouts, which
# separates them before any shape work - the same trick that makes red-vs-black
# suit detection exact.
def masks(im):
    r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    return {
        "cyan": (b > 150) & (g > 130) & (r < 120) & (b - r > 70),
        "gold": (r > 170) & (g > 140) & (b < 120) & (r - b > 80),
        "white": (r > 200) & (g > 200) & (b > 200),
    }

def harvest(paths):
    out = {}
    for p in paths:
        im = np.asarray(Image.open(p).convert("RGB"), dtype=np.int16)
        for colour, mask in masks(im).items():
            labels, _ = ndimage.label(mask)
            for sl in ndimage.find_objects(labels):
                ys, xs = sl
                h, w = ys.stop - ys.start, xs.stop - xs.start
                if 4 <= h <= 32 and 2 <= w <= 32:
                    patch = mask[ys, xs].astype(np.uint8) * 255
                    out.setdefault((colour, h), []).append((w, patch, os.path.basename(p), xs.start, ys.start))
    return out

def cluster(items, tol=18.0):
    """Group identical glyphs. Widths differ per character, so a candidate only
    compares against templates of its own width."""
    reps = []          # (w, mean-accumulator, count, members)
    for w, patch, *_ in items:
        a = patch.astype(np.float64)
        for rep in reps:
            if rep["w"] == w and np.abs(a - rep["sum"] / rep["n"]).mean() < tol:
                rep["sum"] += a
                rep["n"] += 1
                break
        else:
            reps.append({"w": w, "sum": a.copy(), "n": 1})
    reps.sort(key=lambda r: -r["n"])
    return reps

if __name__ == "__main__":
    paths = sorted(glob.glob(os.path.join(CAPTURES, "*.png")))
    print(f"{len(paths)} frames")
    harvested = harvest(paths)
    for (colour, h), items in sorted(harvested.items(), key=lambda kv: (kv[0][0], -len(kv[1]))):
        if len(items) < 40:
            continue
        reps = cluster(items)
        big = [r for r in reps if r["n"] >= max(5, len(items) // 100)]
        print(f"{colour:6s} h={h:2d}  {len(items):5d} glyphs  {len(reps):3d} classes  "
              f"{len(big):2d} significant  widths={sorted({r['w'] for r in big})}  counts={[r['n'] for r in big][:16]}")
        # Contact sheet: one averaged representative per class, index-labelled.
        if big:
            pad = 4
            cw = max(r["w"] for r in big) + pad
            sheet = Image.new("L", (cw * len(big), h + pad), 20)
            for i, r in enumerate(big):
                avg = (r["sum"] / r["n"]).clip(0, 255).astype(np.uint8)
                sheet.paste(Image.fromarray(avg), (i * cw + pad // 2, pad // 2))
            sheet.resize((sheet.width * 5, sheet.height * 5), Image.NEAREST).save(
                os.path.join(OUT, f"{colour}-{h:02d}.png"))
