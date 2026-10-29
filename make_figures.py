"""Draw the README figures. All numbers are copied from the run logs (see LOG.md); no model is run here.
Usage: python3 make_figures.py  -> figures/*.png"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

os.makedirs("figures", exist_ok=True)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"  # validated categorical slots 1-3
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.facecolor": SURF, "figure.facecolor": SURF,
                     "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(f"figures/{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def style(ax, xgrid=True):
    ax.grid(axis="x" if xgrid else "y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- data (excess = real P-R minus null P-R)
# per layer: (noise meta, noise kmeans, shuffled meta, shuffled kmeans)
MIXED_GEMMA = {0: {12: (-.053, -.066, -.036, -.029), 20: (-.045, -.051, -.047, -.052), 28: (-.018, -.036, -.056, -.049)},
               1: {12: (-.053, -.068, -.039, -.037), 20: (-.042, -.043, -.041, -.036), 28: (-.032, -.034, -.069, -.047)},
               2: {12: (-.046, -.059, -.030, -.024), 20: (-.051, -.033, -.055, -.031), 28: (-.030, -.045, -.066, -.062)}}
MIXED_QWEN = {0: {10: (-.008, -.007, -.006, -.004), 16: (-.008, -.008, -.006, -.006), 23: (-.006, -.006, -.005, -.005)},
              1: {10: (-.009, -.007, -.006, -.006), 16: (-.007, -.009, -.005, -.006), 23: (-.007, -.006, -.005, -.004)},
              2: {10: (-.006, -.007, -.004, -.006), 16: (-.007, -.008, -.007, -.006), 23: (-.006, -.006, -.005, -.004)}}
PT_ONLY = {12: (.028, .003, .027, .019), 20: (.029, .028, .015, .023), 28: (.020, .007, -.007, -.006)}
IT_ONLY = {12: (-.013, -.034, -.019, -.031), 20: (-.001, -.002, -.023, -.016), 28: (.004, .006, -.042, -.026)}
MATRY = {"Gemma-3-4B": {"L12": (.034, -.014), "L20": (.031, -.027), "L28": (.072, -.056)},
         "Qwen2-VL-2B": {"L10": (-.003, -.002), "L16": (-.005, -.003), "L23": (-.004, -.002)}}
FVU = {"base-only SAE": {12: (.187, .420), 20: (.206, .500), 28: (.192, .428)},   # (own model, other model)
       "tuned-only SAE": {12: (.223, .396), 20: (.244, .472), 28: (.237, .396)}}
SIB = {"meta": {12: (.43, .53), 20: (.50, .80), 28: (.30, .40)},                    # (real, random groups)
       "k-means": {12: (.73, .57), 20: (.56, .82), 28: (.53, .43)}}


def noise_shuf(d):
    """dict layer -> 4-tuple  ->  (noise values, shuffled values)"""
    n = [v for t in d.values() for v in t[:2]]
    s = [v for t in d.values() for v in t[2:]]
    return n, s


# ---------------------------------------------------------------- fig 1: the question (schematic)
def fig_idea():
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, bold=False, tc=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, color=tc,
                fontweight="bold" if bold else "normal")

    box(1.3, 3.0, 2.4, 0.75, "parent: food", "#e8f0fb", C1, bold=True)
    kids = ["broccoli", "carrot", "pizza"]
    for i, k in enumerate(kids):
        x = 0.2 + i * 1.6
        box(x, 1.3, 1.4, 0.6, k, "#fdeee7", C2)
        ax.annotate("", xy=(x + 0.7, 1.9), xytext=(2.5, 3.0), arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    ax.text(2.5, 0.6, "SAE concept tree\n(level-2 SAE groups level-1 latents)", ha="center", va="center",
            fontsize=9, color=INK2)
    ax.text(5.35, 3.55, "Compare the base model and the instruction-tuned model on the same image tokens:",
            fontsize=10, color=INK)
    rows = [("Children change, parent stays", "the 'stable parent' story", C1),
            ("Parent changes like random groups", "no family structure in the change", C2)]
    for j, (a, b, c) in enumerate(rows):
        y = 2.55 - j * 1.0
        ax.add_patch(FancyBboxPatch((5.35, y - 0.32), 4.6, 0.72, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc=SURF, ec=c, lw=1.4))
        ax.text(5.55, y + 0.12, a, fontsize=10, color=INK, fontweight="bold")
        ax.text(5.55, y - 0.16, b, fontsize=9, color=INK2)
    ax.text(5.35, 0.45, "change = 1 - corr(base activation, tuned activation);\nparent activation = sum of its children",
            fontsize=9, color=INK2, va="center")
    save(fig, "fig1_question")


# ---------------------------------------------------------------- fig 2: the first result (mixed dictionary)
def fig_first():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    for ax, (name, data) in zip(axes, [("Gemma-3-4B (pt vs it)", MIXED_GEMMA), ("Qwen2-VL-2B (base vs Instruct)", MIXED_QWEN)]):
        layers = list(data[0].keys())
        for li, L in enumerate(layers):
            for s in range(3):
                t = data[s][L]
                for k, v in enumerate(t):
                    col = C1 if k < 2 else C2
                    x = li + (k - 1.5) * 0.09 + (s - 1) * 0.025
                    ax.plot(x, v, "o", ms=6, color=col, mec=SURF, mew=1)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(range(len(layers)), [f"layer {L}" for L in layers])
        ax.set_title(name, fontsize=10, color=INK, loc="left")
        style(ax, xgrid=False)
        lo = min(v for s in data.values() for t in s.values() for v in t)
        ax.set_ylim(lo * 1.25, -lo * 0.25)
    axes[0].set_ylabel("excess (real - null)\nbelow 0 = families more stable")
    leg = [Line2D([], [], marker="o", ls="", color=C1, ms=6, label="vs noise null"),
           Line2D([], [], marker="o", ls="", color=C2, ms=6, label="vs shuffled null")]
    fig.legend(handles=leg, loc="upper right", frameon=False, ncol=2, bbox_to_anchor=(1.0, 1.04))
    fig.suptitle("Mixed (base + tuned) dictionary: 72 of 72 checks below zero", x=0.01, ha="left", y=1.12,
                 fontsize=12, fontweight="bold")
    fig.text(0.01, -0.04, "Each layer: 2 grouping methods x 2 nulls x 3 SAE seeds. Panels have their own y-scales.",
             fontsize=8.5, color=INK2)
    save(fig, "fig2_first_result")


# ---------------------------------------------------------------- fig 3: the control (headline)
def fig_control():
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    rows = [("base + tuned mix", noise_shuf(MIXED_GEMMA[0])), ("tuned only", noise_shuf(IT_ONLY)),
            ("base only", noise_shuf(PT_ONLY))]
    for i, (lab, (n, s)) in enumerate(rows):
        y = len(rows) - 1 - i
        for j, v in enumerate(n):
            ax.plot(v, y + 0.12 + (j % 2) * 0.06, "o", ms=7, color=C1, mec=SURF, mew=1)
        for j, v in enumerate(s):
            ax.plot(v, y - 0.12 - (j % 2) * 0.06, "s", ms=6.5, color=C2, mec=SURF, mew=1)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlabel("excess (real P-R minus null P-R)")
    ax.text(-0.004, 2.5, "<- families more stable than chance", ha="right", va="center", fontsize=8.5, color=INK2)
    ax.text(0.004, 2.5, "no stable families ->", ha="left", va="center", fontsize=8.5, color=INK2)
    style(ax)
    leg = [Line2D([], [], marker="o", ls="", color=C1, ms=7, label="vs noise null"),
           Line2D([], [], marker="s", ls="", color=C2, ms=6.5, label="vs shuffled null")]
    fig.legend(handles=leg, loc="upper right", frameon=False, ncol=2, bbox_to_anchor=(1.0, 1.02))
    ax.set_title("Gemma-3-4B, seed 0: the same SAE trained on different activations",
                 fontsize=10, color=INK2, loc="left")
    fig.suptitle("The effect depends on what the dictionary was trained on", x=0.01, ha="left", y=1.08,
                 fontsize=12, fontweight="bold")
    fig.text(0.01, -0.05, "Each row: 3 layers (12, 20, 28) x 2 grouping methods (meta, k-means) x 2 nulls.",
             fontsize=8.5, color=INK2)
    save(fig, "fig3_dictionary_control")


# ---------------------------------------------------------------- fig 4: Matryoshka
def fig_matry():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
    for ax, (name, d) in zip(axes, MATRY.items()):
        labs = list(d.keys())
        for i, L in enumerate(labs):
            n, s = d[L]
            ax.plot(i - 0.08, n, "o", ms=8, color=C1, mec=SURF, mew=1)
            ax.plot(i + 0.08, s, "s", ms=7.5, color=C2, mec=SURF, mew=1)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(range(len(labs)), labs)
        ax.set_xlim(-0.5, len(labs) - 0.5)
        ax.set_title(name, fontsize=10, loc="left")
        style(ax, xgrid=False)
        m = max(abs(v) for t in d.values() for v in t)
        ax.set_ylim(-m * 1.3, m * 1.3)
    axes[0].set_ylabel("excess (real - null)")
    leg = [Line2D([], [], marker="o", ls="", color=C1, ms=7, label="vs noise null"),
           Line2D([], [], marker="s", ls="", color=C2, ms=6.5, label="vs shuffled null")]
    fig.legend(handles=leg, loc="upper right", frameon=False, ncol=2, bbox_to_anchor=(1.0, 1.04))
    fig.suptitle("Matryoshka SAE (mixed dictionary): sign depends on the null in Gemma", x=0.01, ha="left",
                 y=1.13, fontsize=12, fontweight="bold")
    save(fig, "fig4_matryoshka")


# ---------------------------------------------------------------- fig 5: cross-fit of single-model SAEs
def fig_fit():
    fig, ax = plt.subplots(figsize=(8.6, 2.9))
    labels = []
    y = 0
    for dname, d in FVU.items():
        for L, (own, other) in d.items():
            ax.plot([own, other], [y, y], color=GRID, lw=2.5, zorder=1)
            ax.plot(own, y, "o", ms=8, color=C3, mec=SURF, mew=1, zorder=2)
            ax.plot(other, y, "o", ms=8, color=C2, mec=SURF, mew=1, zorder=2)
            labels.append(f"{dname}, layer {L}")
            y += 1
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.6)
    ax.set_xlabel("fraction of variance unexplained (lower = better fit)")
    style(ax)
    leg = [Line2D([], [], marker="o", ls="", color=C3, ms=7, label="on the model it was trained on"),
           Line2D([], [], marker="o", ls="", color=C2, ms=7, label="on the other model")]
    fig.legend(handles=leg, loc="upper right", frameon=False, ncol=2, bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Single-model SAEs fit the other model about equally badly", x=0.01, ha="left", y=1.09,
                 fontsize=12, fontweight="bold")
    save(fig, "fig5_crossfit")


# ---------------------------------------------------------------- fig 6: twin mechanism
def fig_mech():
    fig, ax = plt.subplots(figsize=(8.6, 2.9))
    labels, y = [], 0
    for g, d in SIB.items():
        for L, (real, rnd) in d.items():
            ax.plot([real, rnd], [y, y], color=GRID, lw=2.5, zorder=1)
            ax.plot(rnd, y, "o", ms=8, color=INK2, mec=SURF, mew=1, zorder=2)
            ax.plot(real, y, "o", ms=8, color=C1, mec=SURF, mew=1, zorder=2)
            labels.append(f"{g} groups, layer {L}")
            y += 1
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of tuned-specific children that have a base-specific sibling in the same family")
    style(ax)
    leg = [Line2D([], [], marker="o", ls="", color=C1, ms=7, label="real families"),
           Line2D([], [], marker="o", ls="", color=INK2, ms=7, label="random groups, same sizes")]
    fig.legend(handles=leg, loc="upper right", frameon=False, ncol=2, bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("No 'base twin / tuned twin' pairing inside families", x=0.01, ha="left", y=1.09,
                 fontsize=12, fontweight="bold")
    save(fig, "fig6_mechanism")


# ---------------------------------------------------------------- fig 7: which weights change
def fig_vision():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.7))
    data = {"Gemma-3-4B": (0.0, 0.160, 0.054), "Qwen2-VL-2B": (0.0, 0.012, 0.0022)}
    parts = ["vision tower", "projector", "LM layer 0"]
    for ax, (name, vals) in zip(axes, data.items()):
        ax.bar(range(3), vals, width=0.55, color=C1, edgecolor=SURF, linewidth=2)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.04, "0 (bit-identical)" if v == 0 else f"{v:.3g}", ha="center",
                    fontsize=9, color=INK)
        ax.set_xticks(range(3), parts)
        ax.set_ylim(0, max(vals) * 1.3)
        ax.set_title(name, fontsize=10, loc="left")
        style(ax, xgrid=False)
    axes[0].set_ylabel("relative weight change\n||w_tuned - w_base|| / ||w_base||")
    fig.suptitle("Instruction tuning leaves the vision tower untouched", x=0.01, ha="left", y=1.05,
                 fontsize=12, fontweight="bold")
    save(fig, "fig7_weights")


for f in [fig_idea, fig_first, fig_control, fig_matry, fig_fit, fig_mech, fig_vision]:
    f()
print("wrote", sorted(os.listdir("figures")))
