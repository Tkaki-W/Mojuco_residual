"""学習中の曲線を TensorBoard のファイルから読んで画像にする（学習には影響しない。読むだけ）。

  python scripts/plot_progress.py --name run1   → results/<name>_progress.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

plt.rcParams["font.family"] = "Yu Gothic"
ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
SERIES = {"success_sphere": ("球", "#2a78d6"), "success_ellipsoid": ("楕円体", "#eb6834")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="run1")
    a = ap.parse_args()
    runs = sorted((ROOT / "results" / "tb").glob(f"{a.name}_*"))
    ea = EventAccumulator(str(runs[-1]))
    ea.Reload()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for key, (label, col) in SERIES.items():
        ev = ea.Scalars(f"grasp/{key}")
        x, y = [e.step for e in ev], [e.value for e in ev]
        ax1.plot(x, y, color=col, lw=2, marker="o", ms=4, label=label)
        ax1.annotate(f"{label} {y[-1]:.2f}", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                     va="center", fontsize=9, color=INK)
    ev = ea.Scalars("grasp/cop_dist_median")
    ax2.plot([e.step for e in ev], [e.value for e in ev], color="#2a78d6", lw=2, marker="o", ms=4)
    ax2.axhline(1.0, color=MUTED, lw=1, ls="--")
    ax2.annotate("成功の基準 1mm", (0, 1.0), xytext=(2, 4), textcoords="offset points", fontsize=8, color=MUTED)
    ax1.set(title="学習中の成功率（探索のばらつき込み）", xlabel="つかんだ回数", ylabel="成功率", ylim=(0, 1))
    ax2.set(title="圧力中心のずれ（中央値）", xlabel="つかんだ回数", ylabel="mm", ylim=(0, None))
    for ax in (ax1, ax2):
        ax.grid(color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(MUTED)
        ax.tick_params(colors=MUTED)
        ax.title.set_color(INK)
    ax1.legend(frameon=False, fontsize=9, loc="upper left")
    ax1.margins(x=0.12)
    fig.tight_layout()
    out = ROOT / "results" / f"{a.name}_progress.png"
    fig.savefig(out, dpi=110)
    print("保存:", out)


if __name__ == "__main__":
    main()
