"""学習した方策を評価する：球の式だけ（補正なし）と比べた成功率、学習曲線、動画。
形（球・楕円体）ごとに、同じ n_eval 個の腫瘍（seed 10000〜）で評価する。

  python scripts/eval.py               # config.py の TrainConfig.name の run（results/<name>/eval.png）
  python scripts/eval.py --name run6 --fingers 2   # ほかの run（2本／1本は TrainConfig.fingers と同じにする）
  python scripts/eval.py --video 3     # 方策と球の式だけの動画を3つずつ保存
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
from hand_sim.config import TRAIN as C  # noqa: E402
from hand_sim.env import GraspEnv  # noqa: E402

KINDS = {"sphere": "球", "ellipsoid": "楕円体"}


def act(model, obs):
    if model is None:
        return np.zeros(4, np.float32)      # 球の式だけ
    return model.predict(obs, deterministic=True)[0]


def evaluate(model, env, n=C.n_eval):
    """形ごとの成功率と、全体のずれの中央値・禁止の接触の割合"""
    out, rows = {}, []
    for kind, label in KINDS.items():
        env.kind = kind
        r = []
        for i in range(n):
            obs, _ = env.reset(seed=10_000 + i)
            info = env.step(act(model, obs))[4]
            r.append((info["success"], info["cop_dist"], info["forbidden"]))
        out[label] = float(np.mean([x[0] for x in r]))
        rows += r
    r = np.array(rows, float)
    out.update({"全体": r[:, 0].mean(), "ずれ中央値": float(np.median(r[:, 1])), "禁止の接触": r[:, 2].mean()})
    return out


def save_video(model, path, seed=0, fingers=C.fingers, slow=0.15):
    """1回分を動画にする（指先側から斜めに見る）"""
    from show_grasp import apply, camera, scene_option
    import imageio.v2 as imageio
    env = GraspEnv(visual_mesh=True, fingers=fingers)
    obs, _ = env.reset(seed=seed)
    env.step(act(model, obs))
    res, sim = env.result, env.sim
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    times = [f[0] for f in res.traj]
    idx = np.searchsorted(times, np.arange(0, times[-1], slow / 30)).clip(0, len(times) - 1)
    look = np.array([0.0, 0.0, env.tumor.top * 6e-4])
    with mujoco.Renderer(sim.m, 480, 640) as r, imageio.get_writer(path, fps=30) as w:
        for i in list(idx) + [len(times) - 1] * 30:
            apply(sim, res.traj[i])
            r.update_scene(sim.d, camera(look, 60, -20, 0.14), scene_option())
            w.append_data(r.render())
    c = res.contact
    print(f"動画 {path.name}: ずれ {c.cop_dist:.2f}mm, 失敗の接触 {c.forbidden or 'なし'}")


def main():
    from stable_baselines3 import PPO
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=C.name, help="評価する run（results/<name>）")
    ap.add_argument("--fingers", type=int, default=C.fingers, help="2＝2本でつまむ、1＝1本で押す")
    ap.add_argument("--video", type=int, default=0)
    a = ap.parse_args()
    out = ROOT / "results" / a.name
    env = GraspEnv(fingers=a.fingers)

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
    for i, k in enumerate(KINDS.values()):
        col = f"C{i}"
        if curve:
            ax1.plot([s for s, _ in curve], [e[k] for _, e in curve], "o-", color=col, label=f"{k}（強化学習）")
        ax1.axhline(base[k], color=col, ls="--", label=f"{k}（球の式だけ）")
    ax1.set(xlabel="学習ステップ", ylabel="成功率", ylim=(0, 1), title=f"学習曲線（形ごとに同じ{C.n_eval}個の腫瘍で評価）")
    ax1.legend(fontsize=8)
    keys = [*KINDS.values(), "全体"]
    x = np.arange(len(keys))
    ax2.bar(x - 0.2, [base[k] for k in keys], 0.4, label="球の式だけ")
    ax2.bar(x + 0.2, [final[k] for k in keys], 0.4, label="強化学習の補正あり")
    ax2.set(xticks=x, xticklabels=keys, ylim=(0, 1), ylabel="成功率", title="形ごとの成功率")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out / "eval.png", dpi=120)
    print("保存:", out / "eval.png")

    model = PPO.load(out / "final", device="cpu")
    for i in range(a.video):
        save_video(None, out / "videos" / f"compare_{i}_baseline.mp4", seed=20_000 + i, fingers=a.fingers)
        save_video(model, out / "videos" / f"compare_{i}_policy.mp4", seed=20_000 + i, fingers=a.fingers)


if __name__ == "__main__":
    main()
