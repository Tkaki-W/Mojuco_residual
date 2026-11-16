"""学習した方策を評価する：球の式だけ（補正なし）と比べた成功率、学習曲線、動画。

  python scripts/eval.py --name run1              # 成功率の表・グラフ（results/<name>/eval.png）
  python scripts/eval.py --name run1 --video 3    # 方策と球の式だけの動画を3つずつ保存
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import mujoco
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = "Yu Gothic"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from hand_sim.env import GraspEnv  # noqa: E402

N_EVAL = 300          # 評価に使う腫瘍の数（seed 10000〜 で固定）
_VIDEO_ENV = None


def act(model, obs):
    if model is None:
        return np.zeros(4, np.float32)      # 球の式だけ
    return model.predict(obs, deterministic=True)[0]


def evaluate(model, env, n=N_EVAL):
    rows = []
    for i in range(n):
        obs, _ = env.reset(seed=10_000 + i)
        _, _, _, _, info = env.step(act(model, obs))
        rows.append((info["sphere"], info["success"], info["cop_dist"], info["forbidden"]))
    r = np.array(rows, float)
    sph, ell = r[r[:, 0] == 1], r[r[:, 0] == 0]
    return {"球": sph[:, 1].mean(), "楕円体": ell[:, 1].mean(), "全体": r[:, 1].mean(),
            "ずれ中央値": float(np.median(r[:, 2])), "禁止の接触": r[:, 3].mean()}


def save_video(model, path, seed=0, slow=0.15):
    """1回分を動画にする（指先側から斜めに見る）"""
    global _VIDEO_ENV
    from show_grasp import apply, camera, scene_option
    import imageio.v2 as imageio
    if _VIDEO_ENV is None:
        _VIDEO_ENV = GraspEnv(visual_mesh=True)
    env = _VIDEO_ENV
    obs, _ = env.reset(seed=seed)
    env.step(act(model, obs))
    res, sim = env.result, env.sim
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    times = [f[0] for f in res.traj]
    idx = np.searchsorted(times, np.arange(0, times[-1], slow / 30)).clip(0, len(times) - 1)
    look = np.array([env.tumor.xy[0], env.tumor.xy[1], env.tumor.top * 6e-4]) * [1e-3, 1e-3, 1]
    with mujoco.Renderer(sim.m, 480, 640) as r, imageio.get_writer(path, fps=30) as w:
        for i in list(idx) + [len(times) - 1] * 30:
            apply(sim, res.traj[i])
            r.update_scene(sim.d, camera(look, 60 + env.tumor.yaw, -20, 0.14), scene_option())
            w.append_data(r.render())
    c = res.contact
    print(f"動画 {path.name}: ずれ {c.cop_dist:.2f}mm, 失敗の接触 {c.forbidden or 'なし'}")


def main():
    from stable_baselines3 import PPO
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="run1")
    ap.add_argument("--video", type=int, default=0)
    a = ap.parse_args()
    out = ROOT / "results" / a.name
    env = GraspEnv()

    base = evaluate(None, env)
    print("球の式だけ:", {k: round(v, 3) for k, v in base.items()})
    # 学習曲線：チェックポイントごとに同じ腫瘍で評価
    ckpts = sorted((out / "checkpoints").glob("ppo_*_steps.zip"), key=lambda p: int(p.stem.split("_")[1]))
    curve = []
    for p in ckpts:
        e = evaluate(PPO.load(p, device="cpu"), env)
        curve.append((int(p.stem.split("_")[1]), e))
        print(p.stem, {k: round(v, 3) for k, v in e.items()})
    final = evaluate(PPO.load(out / "final", device="cpu"), env)
    print("学習後:", {k: round(v, 3) for k, v in final.items()})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for k, col in (("球", "C0"), ("楕円体", "C1")):
        if curve:
            ax1.plot([s for s, _ in curve], [e[k] for _, e in curve], "o-", color=col, label=f"{k}（強化学習）")
        ax1.axhline(base[k], color=col, ls="--", label=f"{k}（球の式だけ）")
    ax1.set(xlabel="学習ステップ", ylabel="成功率", ylim=(0, 1), title="学習曲線（同じ300個の腫瘍で評価）")
    ax1.legend(fontsize=8)
    x = np.arange(3)
    keys = ["球", "楕円体", "全体"]
    ax2.bar(x - 0.2, [base[k] for k in keys], 0.4, label="球の式だけ")
    ax2.bar(x + 0.2, [final[k] for k in keys], 0.4, label="強化学習の補正あり")
    ax2.set(xticks=x, xticklabels=keys, ylim=(0, 1), ylabel="成功率", title="形ごとの成功率")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out / "eval.png", dpi=120)
    print("保存:", out / "eval.png")

    model = PPO.load(out / "final", device="cpu")
    for i in range(a.video):
        save_video(None, out / "videos" / f"compare_{i}_baseline.mp4", seed=20_000 + i)
        save_video(model, out / "videos" / f"compare_{i}_policy.mp4", seed=20_000 + i)


if __name__ == "__main__":
    main()
