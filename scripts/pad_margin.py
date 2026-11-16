"""パッド指をゲル指より何 mm 外に開けば干渉しないかを、全形状で調べる（Z は config の値）。

  python scripts/pad_margin.py
"""
import sys
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hand_sim.baseline import hand_pose, radius_from_profile, sphere_formula  # noqa: E402
from hand_sim.config import BaselineConfig  # noqa: E402
from hand_sim.grasp import GraspSim  # noqa: E402
from z_sweep import shapes, visible  # noqa: E402

MARGINS = (0, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10)


def main():
    sim = GraspSim(visual_mesh=False)
    m, d = sim.m, sim.d
    bl = BaselineConfig(pad_extra_slide=0.0)   # 開く量はこのスクリプトで足す
    targets = []
    for kind, t, e in shapes():
        w, R = visible(t)
        sr = sphere_formula(R, radius_from_profile(w, R, e > 0.5), bl=bl)
        if sr.two_finger:
            targets.append((kind, t, e, sr))
    print(f"2本でつまむ形 {len(targets)} 種類、Z={bl.Z}")
    print("ゲル指より外に開く量 [mm] → 成功数 / 干渉で失敗 / パッド指が腫瘍に届かない")
    for mg in MARGINS:
        ok = interf = miss = 0
        bad = []
        for kind, t, e, sr in targets:
            p = hand_pose(t.xy, t.yaw, sr, bl=bl)
            sim.set_tumor(t)
            sim.set_pose(p)
            v = min(sim.cfg.slide_max, p.slide + mg) * 1e-3
            m.eq_data[sim.eq["pad_slide"], 0] = v
            d.qpos[m.jnt_qposadr[sim.jid["pad_slide"]]] = v
            mujoco.mj_forward(m, d)
            r = sim.close()
            c = r.contact
            hit_pad = any("pad" in f for f in c.forbidden)
            interf += hit_pad
            miss += c.pad_force < 0.05
            good = c.cop_dist < 1.0 and not c.forbidden
            ok += good
            if not good:
                bad.append((kind, round(2 * t.b, 1), e))
        print(f"  +{mg:4.1f}: 成功 {ok}/{len(targets)}  干渉 {interf}  パッドが届かない {miss}  失敗 {bad[:4]}", flush=True)


if __name__ == "__main__":
    main()
