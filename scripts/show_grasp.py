"""1回つかむ様子を見る。下の「設定」を書きかえて実行する（▷ ボタン、または python scripts/show_grasp.py）。

ビューアの操作：左ドラッグ＝回転、右ドラッグ＝移動、ホイール＝拡大。ウィンドウを閉じると終了。
左のパネルの Group enable で 3 を ON にすると、衝突判定用の形（凸包）が見える。
"""
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hand_sim.baseline import hand_pose  # noqa: E402
from hand_sim.env import ACT_SCALE, GraspEnv  # noqa: E402
from hand_sim.grasp import Tumor  # noqa: E402

# ===================== 設定（ここを書きかえる） =====================
# --- 腫瘍の形 ---
SHAPE = "ellipsoid"        # "sphere"＝球、"ellipsoid"＝楕円体
D = 20.0                # 短軸の直径 [mm]（球なら直径）。学習の範囲は 15〜25
EMBED = 0.25            # 埋没率（0〜1）。学習の範囲は 0.25〜0.75
ASPECT = 1.5            # 長径／短径（楕円体）。学習の範囲は 1.0〜2.0
HEIGHT = 0.85           # 縦の半径／短軸の半径（楕円体）。学習の範囲は 0.7〜1.0

# --- 構え ---
FINGERS = 2             # 2＝2本でつまむ（球の式）、1＝1本で押す（press_formula）
POLICY = "results/run6/final.zip"           # 学習した方策の補正で構えるなら "results/run6/final.zip" など。None なら式だけ
                        # （方策は FINGERS と同じ設定で学習したものを使う。run6 まで＝2本、run7＝1本）
ACT = (0.0, 0.0, 0.0, 0.0)  # 手で足す補正：指先側 [mm]、ゲル指側 [mm]、上 [mm]、ロール [度]（方策の補正にさらに足す）
RANDOMIZE = False       # True：位置のずれ・向きの誤差・深度ノイズ・摩擦を、学習と同じように乱数で入れる
SEED = 0                # 乱数の種（RANDOMIZE に使う）

# --- 見せ方 ---
MODE = "viewer"         # "viewer"＝ビューアでスロー再生、"video"＝mp4 に保存、"image"＝静止画を保存
OUT = "results/grasp"   # 保存先（video なら OUT.mp4、image なら OUT_tip_open.png など）
SLOW = 0.15             # 再生の速さ（1＝実時間）
# ===================================================================


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


def make_tumor():
    """設定から腫瘍を作る（env.py の sample_tumor と同じ作り方）"""
    b = D / 2
    if SHAPE == "sphere":
        return "sphere", Tumor.sphere(D, EMBED)
    a, c = b * ASPECT, b * HEIGHT
    return "ellipsoid", Tumor(a, b, c, c * (1 - 2 * EMBED))


def report(t, sr, p, res, act):
    c = res.contact
    print(f"腫瘍: {SHAPE}, 短軸の直径 {2*t.b:.1f}, 長軸の直径 {2*t.a:.1f}, 縦の半径 {t.c:.1f}, てっぺん R={t.top:.2f}")
    ok = "2本でつまめる" if FINGERS == 2 else "1本で押せる"
    print(f"式: r={sr.r:.2f}, s={sr.s:.2f}, θ予測={sr.theta:.2f}°, h_c={sr.h_c:.2f}, ゲルの一番下={sr.gel_low:.2f}, {ok}={sr.ok}")
    print(f"補正: 指先側 {act[0]:+.2f}, ゲル指側 {act[1]:+.2f}, 上 {act[2]:+.2f} mm, ロール {act[3]:+.2f}°")
    print(f"構え: ロール {p.roll:.1f}°, スライド ゲル指 {p.slide:.2f} / パッド指 {p.slide_pad:.2f}")
    print(f"結果: 屈曲1 ゲル指 {res.flex1_gel:.2f}° / パッド指 {res.flex1_pad:.2f}°（止まった={res.settled}）")
    print(f"      ゲルの力 {c.gel_force:.2f} N, めり込み {c.max_pen:.2f} mm, 当たったタイル {c.n_tiles}/9")
    print(f"      圧力中心 (断面方向 {c.cop[0]:+.2f}, 長さ方向 {c.cop[1]:+.2f}) mm → ゲル中心から {c.cop_dist:.2f} mm")
    print(f"      失敗の接触: {c.forbidden or 'なし'}")
    print("      タイルの力 [N]（行＝断面方向 内→外、列＝長さ方向 根元→指先）")
    print(np.array2string(res.contact.tile_force, precision=3, suppress_small=True))


def main():
    kind, t = make_tumor()
    # 学習・評価と同じ GraspEnv で、高さプロファイル → R・r の推定 → 式、の順に構えを出す
    # （本当の形を式に入れた結果は scripts/m2_table.py）
    env = GraspEnv(randomize=RANDOMIZE, visual_mesh=True, fingers=FINGERS)
    obs, _ = env.reset(seed=SEED, options={"tumor": (kind, t)})
    act = np.array(ACT, float)
    if POLICY:
        from stable_baselines3 import PPO
        model = PPO.load(ROOT / POLICY, device="cpu")
        act += np.clip(model.predict(obs, deterministic=True)[0], -1, 1) * ACT_SCALE
    sim, sr = env.sim, env.sr
    p = hand_pose(sr, act=tuple(act), center=env.center, yaw=env.yaw)
    res = sim.grasp(t, p)
    report(t, sr, p, res, act)
    look = np.array([0.0, 0.0, t.top * 1e-3 * 0.6])
    out = ROOT / OUT

    if MODE == "image":
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

    if MODE == "video":
        import imageio.v2 as imageio
        out = out.with_suffix(".mp4")
        out.parent.mkdir(parents=True, exist_ok=True)
        fps = 30
        times = np.arange(0, res.traj[-1][0], SLOW / fps)
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
        while v.is_running():
            t0 = time.time()
            for fr in res.traj:          # 閉じる様子をスローで
                while (time.time() - t0) * SLOW < fr[0] and v.is_running():
                    time.sleep(0.005)
                apply(sim, fr)
                v.sync()
            hold = time.time()
            while time.time() - hold < 2.0 and v.is_running():   # 閉じた状態で2秒止める
                time.sleep(0.02)
                v.sync()


if __name__ == "__main__":
    main()
