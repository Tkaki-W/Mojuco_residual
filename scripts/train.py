"""PPO で学習する（球と楕円体）。

  python scripts/train.py --steps 200000 --envs 8 --name run1
  python scripts/train.py --steps 50000 --name run1 --resume results/run1/checkpoints/ppo_100000_steps.zip   # 途中から続ける
学習の様子：tensorboard --logdir results/tb   （成功率・圧力中心のずれ・報酬の曲線）
動画：results/<name>/videos/ に、一定間隔で今の方策の1回分（同じ腫瘍）を保存
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from hand_sim.env import GraspEnv  # noqa: E402
from hand_sim.policy import AsymmetricPolicy  # noqa: E402


class LogAndVideo(BaseCallback):
    """エピソードの結果を TensorBoard に書き、一定間隔で動画を保存する"""

    def __init__(self, video_dir, video_every):
        super().__init__()
        self.video_dir, self.video_every, self.next_video = Path(video_dir), video_every, 0
        self.buf = []

    def _on_step(self):
        for info in self.locals["infos"]:
            self.buf.append(info)
        if len(self.buf) >= 200:
            for kind, sel in (("all", self.buf), ("sphere", [i for i in self.buf if i["sphere"]]),
                              ("ellipsoid", [i for i in self.buf if not i["sphere"]])):
                if sel:
                    self.logger.record(f"grasp/success_{kind}", np.mean([i["success"] for i in sel]))
            self.logger.record("grasp/cop_dist_median", np.median([i["cop_dist"] for i in self.buf]))
            self.logger.record("grasp/forbidden", np.mean([i["forbidden"] for i in self.buf]))
            self.buf = []
        if self.num_timesteps >= self.next_video:
            from eval import save_video
            save_video(self.model, self.video_dir / f"step{self.num_timesteps:07d}.mp4", seed=0)
            self.next_video += self.video_every
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--envs", type=int, default=8)   # 12 だとメモリが足りず止まることがあった
    ap.add_argument("--name", default="run1")
    ap.add_argument("--video_every", type=int, default=25_000, help="0 なら学習中は動画を作らない（メモリ節約）")
    ap.add_argument("--resume", help="途中の方策（.zip）から続ける")
    a = ap.parse_args()

    out = ROOT / "results" / a.name
    env = make_vec_env(GraspEnv, n_envs=a.envs, vec_env_cls=SubprocVecEnv, seed=0)
    if a.resume:
        model = PPO.load(a.resume, env=env, device="cpu", tensorboard_log=str(ROOT / "results" / "tb"))
    else:
        model = PPO(AsymmetricPolicy, env, n_steps=128, batch_size=256, n_epochs=10, learning_rate=3e-4,
                    gamma=0.0, gae_lambda=1.0, ent_coef=0.0, policy_kwargs=dict(net_arch=[128, 128], log_std_init=-1.0),   # 最初の探索は ±1mm 程度（球の式の近くから始める)
                    tensorboard_log=str(ROOT / "results" / "tb"), verbose=1, seed=0, device="cpu")
    cbs = [LogAndVideo(out / "videos", a.video_every or 10**12),
           CheckpointCallback(50_000 // a.envs, str(out / "checkpoints"), name_prefix="ppo")]
    model.learn(a.steps, callback=cbs, tb_log_name=a.name, reset_num_timesteps=not a.resume)
    model.save(out / "final")
    print("保存:", out / "final.zip")


if __name__ == "__main__":
    main()
