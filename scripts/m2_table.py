"""M2：9種類のファントムを球の式で構えて閉じ、結果を表にする。

  python scripts/m2_table.py
出力：results/m2_table.csv、results/m2_table.md、results/m2_grasps.png（2本でつまむ形を指先側から見た図）
"""
import csv
import sys
from pathlib import Path

import matplotlib
import mujoco
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = "Yu Gothic"   # 日本語の表示用（Windows）

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hand_sim.baseline import hand_pose, sphere_formula  # noqa: E402
from hand_sim.config import DEFAULT_BASELINE  # noqa: E402
from hand_sim.grasp import GraspSim, Tumor  # noqa: E402
from show_grasp import apply, camera, scene_option  # noqa: E402

# φ25 の埋没50%の型は実測48%。φ20 は仕様どおり50%で計算（実機は48%）
PHANTOMS = [(d, e) for d in (15, 20, 25) for e in ((0.25, 0.48, 0.75) if d == 25 else (0.25, 0.5, 0.75))]
OUT = ROOT / "results"


def main():
    OUT.mkdir(exist_ok=True)
    sim = GraspSim()
    rows, shots = [], []
    for d, e in PHANTOMS:
        t = Tumor.sphere(d, e)
        sr = sphere_formula(t.top, t.b)
        res = sim.grasp(t, hand_pose(sr))
        c = res.contact
        ok = bool(c.cop_dist < 1.0 and not c.forbidden)
        rows.append({
            "ファントム": f"φ{d}・{round(e * 100)}%", "R": sr.R, "r": sr.r, "h_c": sr.h_c,
            "s": sr.s, "s_pad": sr.s_pad, "θ予測": sr.theta, "θ実際": res.flex1_gel,
            "ゲルの一番下(予測)": sr.gel_low, "判定": "2本" if sr.ok else "1本",
            "圧力中心_断面": c.cop[0], "圧力中心_長さ": c.cop[1], "ずれ": c.cop_dist,
            "タイル": c.n_tiles, "ゲルの力[N]": c.gel_force, "めり込み": c.max_pen,
            "皮膚の力の割合": c.gel_skin_ratio, "失敗の接触": " ".join(f"{a}×{b}" for a, b in c.forbidden) or "なし",
            "成功": ok,
        })
        if sr.ok:
            shots.append((rows[-1]["ファントム"], t, res))

    keys = list(rows[0])
    with open(OUT / "m2_table.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)

    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else ("○" if v is True else "×" if v is False else str(v))  # noqa: E731
    md = [f"Z＝{DEFAULT_BASELINE.Z}、ロール{DEFAULT_BASELINE.roll}°、補正なし（球の式だけ）。長さの単位 mm、角度は度\n",
          "| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    md += ["| " + " | ".join(fmt(r[k]) for k in keys) + " |" for r in rows]
    n2 = [r for r in rows if r["判定"] == "2本"]
    md.append(f"\n2本でつまむ {len(n2)} 種類のうち成功 {sum(r['成功'] for r in n2)}")
    (OUT / "m2_table.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

    # 図：2本でつまむ形を指先側から見た様子（閉じ終わり）
    fig, axes = plt.subplots(1, len(shots), figsize=(4 * len(shots), 3.4))
    with mujoco.Renderer(sim.m, 480, 640) as r:
        for ax, (name, t, res) in zip(np.atleast_1d(axes), shots):
            apply(sim, res.traj[-1])
            r.update_scene(sim.d, camera(np.array([0, 0, t.top * 6e-4]), 90, -5, 0.09), scene_option())
            ax.imshow(r.render())
            c = res.contact
            ax.set_title(f"{name}  ずれ {c.cop_dist:.2f}mm  タイル {c.n_tiles}/9", fontsize=10)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "m2_grasps.png", dpi=120)
    print("保存:", OUT / "m2_table.csv", OUT / "m2_table.md", OUT / "m2_grasps.png")


if __name__ == "__main__":
    main()
