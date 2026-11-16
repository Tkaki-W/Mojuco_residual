"""Z（ロール軸の高さ）をいくつか試して、球と楕円体でベースライン（球の式）の結果を比べる。

  python scripts/z_sweep.py [Z ...]
腫瘍の r は、実際の流れと同じく「見えている短軸方向の幅 w とてっぺん R」から球の式で求める。
"""
import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hand_sim.baseline import hand_pose, radius_from_profile, sphere_formula  # noqa: E402
from hand_sim.config import BaselineConfig  # noqa: E402
from hand_sim.grasp import GraspSim, Tumor  # noqa: E402


def shapes():
    out = []
    for d, e in itertools.product((15, 17.5, 20, 22.5, 25), (0.25, 0.35, 0.48, 0.5, 0.6, 0.75)):
        r = d / 2
        out.append(("球", Tumor(r, r, r, r * (1 - 2 * e)), e))
    for b, ab, cb, e in itertools.product((10, 11.25, 12.5), (1.2, 1.4), (0.7, 0.85), (0.25, 0.4, 0.5, 0.6)):
        c = b * cb
        out.append(("楕円", Tumor(b * ab, b, c, c * (1 - 2 * e)), e))
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    Zs = [float(z) for z in args] or [28, 28.5, 29, 30, 31, 32]
    extra = {"pad_extra_slide": 0.0} if "--same-slide" in sys.argv else {}   # --same-slide：両指同じスライド
    sim = GraspSim(visual_mesh=False)
    sh = shapes()
    print(f"形 {len(sh)} 種類（球 {sum(s[0]=='球' for s in sh)}、楕円体 {sum(s[0]=='楕円' for s in sh)}）")
    for Z in Zs:
        bl = BaselineConfig(Z=Z, **extra)
        t0 = time.time()
        stat = {"球": [0, 0, 0], "楕円": [0, 0, 0]}   # 2本と判定, そのうち成功, 判定外で成功
        smin, lowmin, fails = 1e9, 1e9, []
        for kind, t, e in sh:
            sr = sphere_formula(t.top, radius_from_profile(t.visible_w, t.top, e > 0.5), bl=bl)
            p = hand_pose(sr, bl=bl)
            res = sim.grasp(t, p)
            c = res.contact
            ok = c.cop_dist < 1.0 and not c.forbidden
            if sr.two_finger:
                stat[kind][0] += 1
                stat[kind][1] += ok
                smin, lowmin = min(smin, sr.s), min(lowmin, sr.gel_low)
                if not ok:
                    fails.append((kind, round(2 * t.b, 1), round(t.a / t.b, 1), round(t.c / t.b, 2), e,
                                  round(float(c.cop_dist), 2), c.forbidden))
            elif ok:
                stat[kind][2] += 1
        print(f"Z={Z:5.1f}  球: 2本 {stat['球'][0]} 成功 {stat['球'][1]}（判定外で成功 {stat['球'][2]}）"
              f"  楕円体: 2本 {stat['楕円'][0]} 成功 {stat['楕円'][1]}（判定外で成功 {stat['楕円'][2]}）"
              f"  s最小 {smin:.2f}  ゲル最下の最小 {lowmin:.2f}  [{time.time()-t0:.0f}s]", flush=True)
        for f in fails[:8]:
            print("    失敗:", f)


if __name__ == "__main__":
    main()
