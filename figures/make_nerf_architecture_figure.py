#!/usr/bin/env python
"""make_nerf_architecture_figure.py -- the NeRF network, layer by layer (figure 3.16).

    python make_nerf_architecture_figure.py --out ../Doc/images/diagrams

Writes `nerf_architecture.png`.

THE FIGURE IS CHECKED AGAINST THE MODULE, NOT AGAINST THE PAPER.  `report()` instantiates
the real `nerf.model.NeRF` with the configuration the thesis trained with and compares
every layer drawn here against `model.named_parameters()`, plus the parameter total.  A
diagram of a network is the kind of figure that goes quietly out of date, so if anyone
edits `nerf/model.py` this script fails instead of producing a picture that lies.

Three details a diagram redrawn from memory gets wrong, and which the check pins down:

  * `skips=(4,)` is tested against the index into `pts_linears`, so the encoded position
    is concatenated AFTER the ReLU of the fifth layer, and it is the SIXTH that is the
    widened Linear(319, 256).  Off by one in either direction is easy and invisible.

  * `feature_linear` is Linear(256, 256), not Linear(256, 128).  The 128 belongs to the
    view branch that comes after it.  The thesis text abbreviates over this layer, which
    is fair in prose, but a diagram that omits it has the direction meeting a 256-wide
    vector that came from nowhere.

  * The model emits FOUR RAW CHANNELS.  Neither the density nor the colour is activated
    inside it: `raw2outputs` (nerf/rays.py) applies the ReLU to sigma and the exponential
    (or the softplus) to the colour at compositing time.  Drawing the activations on the
    heads would put them in the wrong module and, worse, would hide that the choice
    between exp and softplus, which the next paragraph of the thesis is about, is made
    outside the network.

The last stage is therefore drawn as a separate compositing block, which is also where
the figure hands over to Section 3.4.2.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import _paths  # noqa: F401

plt.rcParams.update({"font.size": 13})

DPI = 190

# ── The configuration the thesis trained with ────────────────────────────────
# Not the NerfConfig defaults for everything: multires and multires_views are the
# defaults, but the operational block of images_generator.py is what fixes the rest.
D, W = 8, 256
SKIPS = (4,)
MULTIRES, MULTIRES_VIEWS = 10, 4
IN_PTS = 3 + 3 * 2 * MULTIRES            # 63
IN_DIRS = 3 + 3 * 2 * MULTIRES_VIEWS     # 27

# ── Geometry, in inches: one data unit is one inch ───────────────────────────
# The trunk layers are drawn as narrow bars whose HEIGHT is the channel count, the way
# the original paper's figure does it and the way this thesis already shows that figure
# (fig:nerf-pipeline).  Boxes carrying "Linear 256 -> 256" eight times over made the
# figure three times as wide as the page for information that is the same in every box;
# as bars, the one thing that does differ, the concatenation, is the one thing that shows.
UNIT = 1.15 / W                          # inches per channel
LAY_W, LAY_GAP = 0.44, 0.30
COL_GAP = 0.66
MARGIN = 0.22
FONT = 10.5
SMALL = 9.5

STYLE = {
    "enc":   dict(fc="#eef2f6", ec="#8a99a8", tc="#25303a"),
    # The channels the skip appends: the encoder's own grey would have been consistent
    # but at 63 channels against 256 the segment is small, and pale grey on white made
    # the one difference the trunk has to show almost invisible.
    "cat":   dict(fc="#bcd3ea", ec="#1b5e96", tc="#0d2c47"),
    "trunk": dict(fc="#d9e8f7", ec="#2f7ec4", tc="#123a5c"),
    "skip":  dict(fc="#cfe0f2", ec="#1b5e96", tc="#0d2c47"),
    "head":  dict(fc="#e7dcf7", ec="#8452c9", tc="#3d2a63"),
    "comp":  dict(fc="#fdeacd", ec="#d9932a", tc="#5c3d0c"),
    "out":   dict(fc="#dcf0dc", ec="#2ca02c", tc="#12470f"),
}

C_ARROW = "#7c8894"
C_SKIP = "#1b5e96"
C_DIR = "#8452c9"


def trunk_layers() -> list:
    """(name, in, out, is_widened) for the eight layers of `pts_linears`.

    Built by the same rule the module uses, so the two cannot drift: layer 0 takes the
    encoded position, and layer i for i >= 1 is widened when i - 1 is in `skips`, because
    that is the index at which the concatenation happened.
    """
    layers = [("pts_linears.0", IN_PTS, W, False)]
    for i in range(D - 1):
        widened = i in SKIPS
        layers.append((f"pts_linears.{i + 1}", W + IN_PTS if widened else W, W, widened))
    return layers


def box(ax, x, y, w, h, group, text, *, fontsize=FONT, zorder=2):
    s = STYLE[group]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.09",
        facecolor=s["fc"], edgecolor=s["ec"], linewidth=1.5, zorder=zorder))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=s["tc"],
            fontsize=fontsize, zorder=zorder + 1, linespacing=1.3)


def arrow(ax, p0, p1, *, color=C_ARROW, rad=0.0, lw=1.3, ls="-", zorder=1):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>,head_width=0.15,head_length=0.30",
        connectionstyle=f"arc3,rad={rad}", color=color, lw=lw, linestyle=ls,
        shrinkA=1.5, shrinkB=2.5, zorder=zorder))


def figure(out: Path) -> dict:
    """Draws the network.  Returns what it drew, for `report` to check."""
    layers = trunk_layers()
    n = len(layers)
    k_wide = next(k for k, (_, _, _, wdn) in enumerate(layers) if wdn)

    # SIZED FOR THE PAGE: the figure goes in at \linewidth, about 5.7 in, so a label lands
    # on paper at its point size times 5.7 / (figure width in inches).  Shrinking the
    # figure with the fonts left alone is what makes the text bigger in print, since the
    # drawing scales with the axes and the text does not.
    fig, ax = plt.subplots(figsize=(10.4, 4.45))

    # ── the trunk: one bar per layer, height = output channels ──
    # An extra slot is left in front of the widened layer, for the concatenation bar.
    x, y_mid = 0.0, 0.0
    xs = []
    for k in range(n):
        if k == k_wide:
            x += LAY_W + LAY_GAP
        xs.append(x)
        x += LAY_W + LAY_GAP
    trunk_right = xs[-1] + LAY_W
    h_bar = W * UNIT

    for k in range(n):
        box(ax, xs[k], y_mid - h_bar / 2, LAY_W, h_bar, "trunk", "")
        ax.text(xs[k] + LAY_W / 2, y_mid, str(k), ha="center", va="center",
                color="#123a5c", fontsize=SMALL - 1.0, zorder=4)
        if k and k != k_wide:
            arrow(ax, (xs[k - 1] + LAY_W, y_mid), (xs[k], y_mid))
    # Left-aligned under the trunk and not centred: centred it reached far enough right to
    # run into the compositing block, which sits below the head column.
    ax.text(xs[0], y_mid - h_bar / 2 - 1.30,
            rf"$D = 8$ layers of $W = {W}$," "\n" r"each $\tt Linear$ then ReLU",
            ha="left", va="top", color="#5f6c79", fontsize=SMALL, linespacing=1.3)

    # The concatenation, drawn as a bar TALLER than the others by exactly the channels it
    # adds: that height difference is the whole content of the skip connection, and it is
    # what makes the layer after it a Linear(319, 256) instead of a Linear(256, 256).
    h_cat = (W + IN_PTS) * UNIT
    cat_x = xs[k_wide] - LAY_GAP - LAY_W
    box(ax, cat_x, y_mid - h_bar / 2, LAY_W, h_bar, "trunk", "")
    box(ax, cat_x, y_mid + h_bar / 2, LAY_W, h_cat - h_bar, "cat", "")
    arrow(ax, (xs[k_wide - 1] + LAY_W, y_mid), (cat_x, y_mid))
    arrow(ax, (cat_x + LAY_W, y_mid), (xs[k_wide], y_mid))
    ax.text(cat_x + LAY_W / 2, y_mid - h_bar / 2 - 0.14,
            f"${W}+{IN_PTS}$\n$={W + IN_PTS}$", ha="center", va="top", color=C_SKIP,
            fontsize=SMALL - 0.5, linespacing=1.2)

    # ── the encoded position ──
    enc_w, enc_h = 1.50, 0.74
    box(ax, -COL_GAP - enc_w, y_mid - enc_h / 2, enc_w, enc_h, "enc",
        f"$\\gamma(\\mathbf{{x}})$\n{IN_PTS} channels", fontsize=SMALL)
    arrow(ax, (-COL_GAP, y_mid), (xs[0], y_mid))

    # ── the skip connection ──
    y_skip = y_mid + h_cat / 2 + 0.50
    ax.plot([-COL_GAP - enc_w / 2, -COL_GAP - enc_w / 2, cat_x + LAY_W / 2],
            [y_mid + enc_h / 2, y_skip, y_skip], color=C_SKIP, lw=1.5, zorder=1)
    arrow(ax, (cat_x + LAY_W / 2, y_skip), (cat_x + LAY_W / 2, y_mid + h_cat / 2),
          color=C_SKIP, lw=1.5)
    ax.text((-COL_GAP - enc_w / 2 + cat_x) / 2, y_skip + 0.08,
            r"skip: $\gamma(\mathbf{x})$ concatenated again",
            ha="center", va="bottom", color=C_SKIP, fontsize=SMALL)

    # ── the two heads.  Name over shape: side by side the labels overflowed the box. ──
    hx = trunk_right + COL_GAP
    # Sized on `views_linears.0`, the longest label: the monospace math face runs wider
    # per character than the sans one, so the prose labels are not what sets this.
    hw, hh = 3.55, 0.66
    y_sigma = y_mid - 1.34
    y_feat = y_mid + 0.74
    box(ax, hx, y_sigma, hw, hh, "head",
        r"$\tt alpha\_linear$" + "\n" + rf"${W} \to 1$", fontsize=SMALL)
    box(ax, hx, y_feat, hw, hh, "head",
        r"$\tt feature\_linear$" + "\n" + rf"${W} \to {W}$", fontsize=SMALL)
    arrow(ax, (trunk_right, y_mid), (hx, y_sigma + hh / 2), rad=-0.16)
    arrow(ax, (trunk_right, y_mid), (hx, y_feat + hh / 2), rad=0.16)

    # ── the view branch ──
    y_view = y_feat + hh + 0.80
    box(ax, hx, y_view, hw, hh, "head",
        r"$\tt views\_linears.0$" + "\n" + rf"${W + IN_DIRS} \to {W // 2}$, ReLU",
        fontsize=SMALL)
    y_rgb = y_view + hh + 0.64
    box(ax, hx, y_rgb, hw, hh, "head",
        r"$\tt rgb\_linear$" + "\n" + rf"${W // 2} \to 3$", fontsize=SMALL)

    # The concatenation sits at a third of the column and not at its centre, so that the
    # encoded direction can come in from the left of the head column instead of from on
    # top of `feature_linear`, which is what it overlapped when the column got wider.
    cat2_x, cat2_y = hx + hw * 0.30, y_feat + hh + 0.40
    ax.plot([cat2_x, cat2_x], [y_feat + hh, cat2_y - 0.09], color=C_ARROW, lw=1.3,
            zorder=1)
    ax.scatter([cat2_x], [cat2_y], s=64, facecolor="white", edgecolor=C_DIR,
               linewidth=1.5, zorder=4)
    ax.text(cat2_x, cat2_y, "+", ha="center", va="center", color=C_DIR, fontsize=11,
            zorder=5)
    arrow(ax, (cat2_x, cat2_y + 0.09), (cat2_x, y_view), color=C_ARROW)
    arrow(ax, (cat2_x, y_view + hh), (cat2_x, y_rgb), color=C_ARROW)

    # The encoded direction enters at the concatenation and nowhere else: it never
    # touches the trunk, which is the whole reason the density is view-independent.
    # Placed against the LEFT EDGE of the head column, not offset from the concatenation:
    # measured from the concatenation it ended up partly on top of `feature_linear`.
    dir_w = 1.50
    dir_right = hx - 0.16
    box(ax, dir_right - dir_w, cat2_y - 0.37, dir_w, 0.74, "enc",
        f"$\\gamma(\\mathbf{{d}})$\n{IN_DIRS} channels", fontsize=SMALL)
    arrow(ax, (dir_right, cat2_y), (cat2_x - 0.09, cat2_y), color=C_DIR)

    # ── compositing ──
    # Four short lines rather than two long ones: the box is as wide as the head column
    # and the title alone, on one line, was wider than that.
    cw, ch = hw, 1.34
    y_comp = y_sigma - 1.86
    box(ax, hx, y_comp, cw, ch, "comp",
        "compositing\n($\\tt raw2outputs$)\n"
        "$\\sigma$:  ReLU\n"
        "$\\mathbf{c}$:  $\\exp$ or softplus", fontsize=SMALL - 1.0)
    arrow(ax, (hx + cw * 0.26, y_sigma), (hx + cw * 0.26, y_comp + ch), color=C_ARROW)
    ax.text(hx + cw * 0.26 - 0.09, (y_sigma + y_comp + ch) / 2, r"$\sigma$ raw",
            ha="right", va="center", color="#5f6c79", fontsize=SMALL)
    # The colour comes from the top of the stack and has to come down the outside.
    x_rail = hx + cw + 0.48
    ax.plot([hx + cw, x_rail, x_rail], [y_rgb + hh / 2, y_rgb + hh / 2, y_comp + ch / 2],
            color=C_ARROW, lw=1.3, zorder=1)
    arrow(ax, (x_rail, y_comp + ch / 2), (hx + cw, y_comp + ch / 2), color=C_ARROW)
    ax.text(x_rail + 0.09, (y_rgb + y_comp + ch) / 2, r"$\mathbf{c}$ raw",
            ha="left", va="center", color="#5f6c79", fontsize=SMALL, rotation=-90)

    ax.set_aspect("equal")
    ax.set_xlim(-COL_GAP - enc_w - MARGIN, x_rail + 0.62 + MARGIN)
    ax.set_ylim(y_comp - MARGIN, y_rgb + hh + 0.26 + MARGIN)
    ax.axis("off")

    path = out / "nerf_architecture.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  + {path}")
    return dict(layers=layers, k_wide=k_wide)


def report(drawn: dict) -> None:
    """Compare what was drawn against the real module.

    This is the point of the script.  Every shape in the figure is read back out of
    `named_parameters()`, so the picture cannot survive a change to nerf/model.py.
    """
    import torch  # noqa: F401  (imported here so --help works without it)
    from nerf.model import NeRF

    model = NeRF(D=D, W=W, input_ch=IN_PTS, input_ch_views=IN_DIRS,
                 skips=list(SKIPS), use_viewdirs=True)
    shapes = {n: tuple(p.shape) for n, p in model.named_parameters()
              if n.endswith("weight")}
    total = sum(p.numel() for p in model.parameters())

    print(f"\n  built NeRF(D={D}, W={W}, input_ch={IN_PTS}, "
          f"input_ch_views={IN_DIRS}, skips={list(SKIPS)})")
    print(f"  {total:,} parameters")

    # The trunk, layer by layer.  torch stores a Linear weight as (out, in).
    for name, cin, cout, widened in drawn["layers"]:
        got = shapes[f"{name}.weight"]
        assert got == (cout, cin), \
            f"{name} is {got[1]} -> {got[0]} in the module and " \
            f"{cin} -> {cout} in the figure"
        print(f"  {name:16s} {cin:4d} -> {cout:4d}"
              + ("   <- the skip lands here" if widened else ""))

    # The heads.
    for name, want in (("alpha_linear", (1, W)),
                       ("feature_linear", (W, W)),
                       ("views_linears.0", (W // 2, W + IN_DIRS)),
                       ("rgb_linear", (3, W // 2))):
        got = shapes[f"{name}.weight"]
        assert got == want, f"{name} is {got} in the module and {want} in the figure"
        print(f"  {name:16s} {want[1]:4d} -> {want[0]:4d}")

    # The off-by-one the figure is most likely to get wrong: which layer is the wide one.
    wide = [k for k, (_, cin, _, _) in enumerate(drawn["layers"]) if cin == W + IN_PTS]
    assert wide == [SKIPS[0] + 1], \
        f"the widened layer is at index {wide} and skips={SKIPS} puts it at " \
        f"{[SKIPS[0] + 1]}: the concatenation is drawn in the wrong place"
    print(f"  the widened layer is pts_linears.{wide[0]}, "
          f"one past the skip index {SKIPS[0]}, as drawn")

    # The model must not activate its own outputs: if it ever does, the compositing block
    # in the figure stops being where the exp/softplus choice lives.
    import inspect
    src = inspect.getsource(NeRF.forward)
    for banned in ("sigmoid", "torch.exp", "softplus"):
        assert banned not in src, \
            f"NeRF.forward now applies {banned}: the figure still shows the model " \
            f"emitting raw channels and the activation happening at compositing"
    print("  NeRF.forward emits raw channels, as drawn")

    # "roughly 600 thousand parameters", implementation.tex:1084.
    assert 550_000 < total < 650_000, \
        f"the thesis says roughly 600 thousand parameters and the model has {total:,}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="../Doc/images/diagrams")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report(figure(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
