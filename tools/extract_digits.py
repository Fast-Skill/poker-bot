"""Build the glyph template file the number reader matches against.

Templates are keyed by exact (height, width). A glyph is never resized: the
client draws its readouts at a handful of fixed sizes, and resampling one size
into another produces interpolation artifacts that split a single character
into dozens of apparent classes. Several templates may share a label - the same
digit has a few anti-aliasing variants - which costs nothing but a comparison.

Three inks are harvested, each from where that readout actually lives:

  cyan   seat stacks, taken from the seven seat plates
  gold   the total pot, taken from the centre of the table
  white  bet amounts on the felt - and, unavoidably, player names, since the
         client draws both in the same white. Names are separated by harvesting
         only glyphs sitting on green felt, and the few name fragments that
         still slip through are left unlabelled below.

Labels were assigned by eye from the contact sheets in captures/_glyphs, and
this script re-renders a labelled sheet so the assignment can be re-checked.
An underscore means "this class is not a character we read" - noise, an icon,
or a letter - and produces no template, so anything shaped like it is refused.
"""
import glob, os, struct
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

CAPTURES = r"C:\poker\captures"
OUT_BIN = r"C:\poker\data\digit_templates.bin"
MAGIC, VERSION = b"PKGT", 1
UNREAD = os.path.join(CAPTURES, "unread")
# Frames kept by `poker typetest`, where the bot chose what to type and so
# could cover digits the client never happened to display on its own.
TYPED = os.path.join(CAPTURES, "typetest")
# Where the client draws the raise amount, at a 1430x1040 table.
AMOUNT_BOX = (1184, 902, 1299, 935)
# The one window size the templates are exact at. A frame captured at any
# other size reflows rather than scaling, so these coordinates mean nothing
# there and reading them would index whatever happens to sit nearby.
TABLE = (1430, 1040)

# Readout locations, found by the cross-frame variance map: static chrome is
# drawn identically in every frame, so anything that changes is a readout.
SEATS = [(715, 936, 972), (278, 846, 878), (149, 434, 466), (529, 241, 273),
         (892, 241, 273), (1264, 434, 466), (1146, 846, 878)]
POT = (640, 820, 370, 402)
HALF = 115
SKIP = "_"

# Read off the contact sheets, ordered by descending sample count.
LABELS = {
    ("gold", 15): "5123648790",
    ("gold", 11): "B",
    ("gold", 3):  ".",
    ("cyan", 22): "3251748906",
    ("cyan", 18): "B682359207620150493178994" + "88",
    ("cyan", 17): "15147495166",
    ("cyan", 14): "BBB",
    ("cyan", 4):  ".",
    ("cyan", 3):  "...",
    # The three trailing underscores are the letters a, c and m, from player
    # names drawn where the plate overlaps felt; the second is an arrow icon.
    ("white", 13): "B_B15BBB015364270B554803232268586781997___",
    # Both 2x2 classes are the decimal point, one a pixel thinner than the
    # other. The w=3 classes are felt speckle.
    ("white", 2):  "..____",
    # The raise amount box, which the client draws a pixel smaller than the
    # labels on the felt. Same ink, its own size, its own templates: the
    # thirteen-pixel set matched every twelve-pixel glyph about equally
    # badly, so nothing won by a margin and every reading was refused.
    # The second class is the text caret, not a character - which is itself
    # worth knowing, since a caret means the field takes keyboard focus.
    ("white", 12): "B_1274659",
    ("white", 1):  ".",
}
MIN_SAMPLES = 3
# The smallest character the client draws is the decimal point, at three to four
# lit pixels. Anything smaller is anti-aliasing speckle on the felt, not a
# glyph, and letting it through would poison the readouts it sits beside.
MIN_INK = 3


def on_felt(im, ys, xs, pad=4):
    """Whether a glyph's surroundings are table felt rather than a name plate."""
    y0, y1 = max(0, ys.start - pad), min(im.shape[0], ys.stop + pad)
    x0, x1 = max(0, xs.start - pad), min(im.shape[1], xs.stop + pad)
    ring = im[y0:y1, x0:x1].reshape(-1, 3)
    r, g, b = ring[:, 0], ring[:, 1], ring[:, 2]
    return ((g > r + 15) & (g > b + 15)).mean() > 0.35


def raw_frames():
    """The frames the bot kept, which include raises in progress."""
    import struct

    sources = sorted(glob.glob(os.path.join(UNREAD, "*.rgb")))
    sources += sorted(glob.glob(os.path.join(TYPED, "*.rgb")))
    for path in sources:
        with open(path, "rb") as f:
            w, h = struct.unpack("<II", f.read(8))
            if (w, h) != TABLE:
                continue
            yield np.frombuffer(f.read(w * h * 3), np.uint8).reshape(h, w, 3).astype(np.int16)


def harvest_amount_box(out):
    """Glyphs from the raise amount field, which sits on black rather than felt."""
    x0, y0, x1, y1 = AMOUNT_BOX
    for im in raw_frames():
        box = im[y0:y1, x0:x1]
        r, g, b = box[:, :, 0], box[:, :, 1], box[:, :, 2]
        white = (r > 180) & (g > 180) & (b > 180) & (abs(r - b) < 25) & (abs(g - b) < 25)
        if white.sum() < 20:
            continue
        labels, _ = ndimage.label(white)
        for sl in ndimage.find_objects(labels):
            ys, xs = sl
            h, w = ys.stop - ys.start, xs.stop - xs.start
            patch = white[ys, xs]
            if 1 <= h <= 16 and 1 <= w <= 16 and patch.sum() >= 2:
                out.setdefault(("white", h), []).append((w, patch.astype(np.uint8) * 255))


def harvest_frames():
    """Collect every glyph candidate, keyed by (ink, height)."""
    out = {}
    for path in sorted(glob.glob(os.path.join(CAPTURES, "*.png"))):
        with Image.open(path) as probe:
            if probe.size != TABLE:
                continue
        im = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
        r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
        boxed = [
            ("cyan", (b > 150) & (g > 130) & (r < 120) & (b - r > 70),
             [(cx - HALF, cx + HALF, y0, y1) for cx, y0, y1 in SEATS]),
            ("gold", (r > 170) & (g > 140) & (b < 120) & (r - b > 80), [POT]),
        ]
        for ink, mask, boxes in boxed:
            for x0, x1, y0, y1 in boxes:
                window = mask[y0:y1, x0:x1]
                labels, _ = ndimage.label(window)
                for sl in ndimage.find_objects(labels):
                    ys, xs = sl
                    h, w = ys.stop - ys.start, xs.stop - xs.start
                    patch = window[ys, xs]
                    if 3 <= h <= 30 and 1 <= w <= 30 and patch.sum() >= MIN_INK:
                        out.setdefault((ink, h), []).append(
                            (w, patch.astype(np.uint8) * 255))

        # White is not confined to a box: bet amounts move with the chips. The
        # felt test is what keeps player names out.
        white = (r > 180) & (g > 180) & (b > 180) & (abs(r - b) < 25) & (abs(g - b) < 25)
        labels, _ = ndimage.label(white)
        for sl in ndimage.find_objects(labels):
            ys, xs = sl
            h, w = ys.stop - ys.start, xs.stop - xs.start
            patch = white[ys, xs]
            if 2 <= h <= 20 and 1 <= w <= 20 and patch.sum() >= MIN_INK and on_felt(im, ys, xs):
                out.setdefault(("white", h), []).append((w, patch.astype(np.uint8) * 255))
    harvest_amount_box(out)
    return out


def cluster(items, tol=20.0):
    """Group identical glyphs. Only same-width candidates ever compare."""
    reps = []
    for w, patch in items:
        v = patch.astype(np.float64)
        for rep in reps:
            if rep["w"] == w and np.abs(v - rep["sum"] / rep["n"]).mean() < tol:
                rep["sum"] += v
                rep["n"] += 1
                break
        else:
            reps.append({"w": w, "sum": v.copy(), "n": 1})
    reps.sort(key=lambda rep: (-rep["n"], rep["w"]))
    return [rep for rep in reps if rep["n"] >= MIN_SAMPLES]


if __name__ == "__main__":
    harvested = harvest_frames()
    templates, sheets = [], []
    for key, expected in sorted(LABELS.items()):
        items = harvested.get(key, [])
        keep = cluster(items)
        if len(keep) != len(expected):
            raise SystemExit(f"{key}: clustered {len(keep)} classes but {len(expected)} labels "
                             f"({expected!r}); re-check the contact sheet")
        for rep, label in zip(keep, expected):
            if label == SKIP:
                continue
            avg = (rep["sum"] / rep["n"]).clip(0, 255).astype(np.uint8)
            # Binarise: each frame's mask is already binary, and an averaged
            # edge would match no single frame crisply.
            templates.append((key[0], label, key[1], rep["w"], (avg > 127).astype(np.uint8) * 255))
        sheets.append((key, keep, expected))
        kept = sum(1 for c in expected if c != SKIP)
        print(f"{key[0]:5s} h={key[1]:2d}: {len(items):5d} glyphs -> "
              f"{len(keep):2d} classes, {kept:2d} labelled  {expected}")

    with open(OUT_BIN, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", VERSION, len(templates)))
        for ink, label, h, w, pixels in templates:
            f.write(struct.pack("<B", len(ink))); f.write(ink.encode())
            f.write(struct.pack("<B", len(label))); f.write(label.encode())
            f.write(struct.pack("<II", h, w)); f.write(pixels.tobytes())
    print(f"\nwrote {OUT_BIN}: {len(templates)} templates, {os.path.getsize(OUT_BIN)} bytes")

    cell_w, cell_h = 34, 46
    width = cell_w * max(len(keep) for _, keep, _ in sheets) + 90
    sheet = Image.new("RGB", (width, cell_h * len(sheets) + 6), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    for row, (key, keep, expected) in enumerate(sheets):
        y = row * cell_h + 3
        draw.text((4, y + 12), f"{key[0][0]}{key[1]:02d}", fill=(140, 140, 140))
        for i, (rep, label) in enumerate(zip(keep, expected)):
            avg = (rep["sum"] / rep["n"]).clip(0, 255).astype(np.uint8)
            img = Image.fromarray((avg > 127).astype(np.uint8) * 255).convert("RGB")
            sheet.paste(img, (44 + i * cell_w, y))
            colour = (90, 90, 90) if label == SKIP else (255, 210, 90)
            draw.text((44 + i * cell_w + 8, y + 32), label, fill=colour)
    sheet.resize((sheet.width * 2, sheet.height * 2), Image.NEAREST).save(
        os.path.join(CAPTURES, "_glyphs", "labelled.png"))
    print("wrote _glyphs/labelled.png")
