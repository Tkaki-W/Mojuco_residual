"""1回つかむ様子を見る（球の式の構え＋補正）。

  python scripts/show_grasp.py                         # φ20・埋没25%、ビューアでスロー再生（ウィンドウを閉じると終了）
  python scripts/show_grasp.py --d 25 --embed 0.48     # 大きさ・埋没率を変える
  python scripts/show_grasp.py --act 0 0 0 5           # 補正 (dx dy dz [mm], droll [度]) を足す
  python scripts/show_grasp.py --video results/grasp.mp4   # 動画に保存（ビューアは開かない）
  python scripts/show_grasp.py --image results/m1         # 静止画（指先側から・横から）を保存

ビューアの操作：左ドラッグ＝回転、右ドラッグ＝移動、ホイール＝拡大。
左のパネルの Group enable で 3 を ON にすると、衝突判定用の形（凸包）が見える。
"""
import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hand_sim.baseline import hand_pose, sphere_formula  # noqa: E402
from hand_sim.grasp import GraspSim, Tumor  # noqa: E402


def scene_option():
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    return opt


def camera(lookat, azimuth=90, elevation=-15, distance=0.13):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
    return cam


def apply(sim, frame):
    _, qpos, mpos, mquat = frame
    sim.d.qpos[:] = qpos
    sim.d.mocap_pos[:] = mpos
    sim.d.mocap_quat[:] = mquat
    mujoco.mj_forward(sim.m, sim.d)


def report(t, sr, p, res):
    c = res.contact
    print(f"腫瘍: 直径 {2*t.b:.1f}, てっぺん R={t.top:.2f}")
    print(f"球の式: s={sr.s:.2f}, θ予測={sr.theta:.2f}°, h_c={sr.h_c:.2f}, ゲルの一番下={sr.gel_low:.2f}, 2本でつまむ={sr.two_finger}")
    print(f"構え: ロール {p.roll:.1f}°, スライド ゲル指 {p.slide:.2f} / パッド指 {p.slide_pad:.2f}")
    print(f"結果: 屈曲1 ゲル指 {res.flex1_gel:.2f}° / パッド指 {res.flex1_pad:.2f}°（止まった={res.settled}）")
    print(f"      ゲルの力 {c.gel_force:.2f} N, めり込み {c.max_pen:.2f} mm, 当たったタイル {c.n_tiles}/9")
    print(f"      圧力中心 (断面方向 {c.cop[0]:+.2f}, 長さ方向 {c.cop[1]:+.2f}) mm → ゲル中心から {c.cop_dist:.2f} mm")
    print(f"      失敗の接触: {c.forbidden or 'なし'}")
    print("      タイルの力 [N]（行＝断面方向 内→外、列＝長さ方向 根元→指先）")
    print(np.array2string(res.contact.tile_force, precision=3, suppress_small=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=float, default=20.0, help="直径 mm")
    ap.add_argument("--embed", type=float, default=0.25, help="埋没率 0〜1")
    ap.add_argument("--act", type=float, nargs=4, default=[0, 0, 0, 0])
    ap.add_argument("--slow", type=float, default=0.15, help="再生の速さ（1＝実時間）")
    ap.add_argument("--video", type=str)
    ap.add_argument("--image", type=str)
    a = ap.parse_args()

    sim = GraspSim()
    t = Tumor.sphere(a.d, a.embed)
    sr = sphere_formula(t.top, t.b)
    p = hand_pose(sr, act=tuple(a.act))
    res = sim.grasp(t, p)
    report(t, sr, p, res)
    look = np.array([0.0, 0.0, t.top * 1e-3 * 0.6])

    if a.image:
        out = Path(a.image)
        out.parent.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as imageio
        with mujoco.Renderer(sim.m, 720, 960) as r:
            for tag, cam in [("tip", camera(look, 90, -5, 0.12)), ("side", camera(look, 0, -5, 0.30)),
                             ("oblique", camera(look, 60, -25, 0.16))]:
                for which, fr in [("open", res.traj[0]), ("closed", res.traj[-1])]:
                    apply(sim, fr)
                    r.update_scene(sim.d, cam, scene_option())
                    imageio.imwrite(f"{out}_{tag}_{which}.png", r.render())
        print("静止画を保存:", out.parent)
        return

    if a.video:
        import imageio.v2 as imageio
        out = Path(a.video)
        out.parent.mkdir(parents=True, exist_ok=True)
        fps = 30
        times = np.arange(0, res.traj[-1][0], a.slow / fps)
        idx = np.searchsorted([f[0] for f in res.traj], times).clip(0, len(res.traj) - 1)
        with mujoco.Renderer(sim.m, 720, 960) as r, imageio.get_writer(out, fps=fps) as w:
            cam = camera(look, 60, -20, 0.16)
            for i in list(idx) + [len(res.traj) - 1] * fps:   # 最後に1秒止める
                apply(sim, res.traj[i])
                r.update_scene(sim.d, cam, scene_option())
                w.append_data(r.render())
        print("動画を保存:", out)
        return

    from mujoco import viewer
    with viewer.launch_passive(sim.m, sim.d) as v:
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        v.cam.lookat[:] = look
        v.cam.azimuth, v.cam.elevation, v.cam.distance = 60, -20, 0.16
        t_end = res.traj[-1][0]
        while v.is_running():
            t0 = time.time()
            for fr in res.traj:          # 閉じる様子をスローで
                while (time.time() - t0) * a.slow < fr[0] and v.is_running():
                    time.sleep(0.005)
                apply(sim, fr)
                v.sync()
            hold = time.time()
            while time.time() - hold < 2.0 and v.is_running():   # 閉じた状態で2秒止める
                time.sleep(0.02)
                v.sync()


if __name__ == "__main__":
    main()
