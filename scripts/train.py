"""PPO で学習する（球と楕円体）。設定は hand_sim/config.py の TrainConfig にまとめてある。

  python scripts/train.py                                                     # 最初から
  python scripts/train.py --resume results/run1/checkpoints/ppo_100000_steps.zip  # 途中から続ける（合計 total_steps まで）
学習の様子：python scripts/plot_progress.py、または tensorboard --logdir results/tb
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
from hand_sim.config import TRAIN as C  # noqa: E402
from hand_sim.env import GraspEnv  # noqa: E402
from hand_sim.policy import AsymmetricPolicy  # noqa: E402


class LogAndVideo(BaseCallback):
    """エピソードの結果を TensorBoard に書き、一定間隔で動画を保存する"""

    def __init__(self, video_dir):
        super().__init__()
        self.video_dir, self.next_video = Path(video_dir), 0
        self.buf = []

    def _on_step(self):
        self.buf += self.locals["infos"]
        if len(self.buf) >= 200:
            for kind, sel in (("all", self.buf), ("sphere", [i for i in self.buf if i["sphere"]]),
                              ("ellipsoid", [i for i in self.buf if not i["sphere"]])):
                if sel:
                    self.logger.record(f"grasp/success_{kind}", np.mean([i["success"] for i in sel]))
            self.logger.record("grasp/cop_dist_median", np.median([i["cop_dist"] for i in self.buf]))
            self.logger.record("grasp/forbidden", np.mean([i["forbidden"] for i in self.buf]))
            self.buf = []
        if C.video_every and self.num_timesteps >= self.next_video:
            from eval import save_video
            save_video(self.model, self.video_dir / f"step{self.num_timesteps:07d}.mp4", seed=0)
            self.next_video = self.num_timesteps + C.video_every
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", help="途中の方策（.zip）から続ける")
    a = ap.parse_args()

    out = ROOT / "results" / C.name
    tb = str(ROOT / "results" / "tb")
    env = make_vec_env(GraspEnv, n_envs=C.n_envs, vec_env_cls=SubprocVecEnv, seed=C.seed,
                       env_kwargs=dict(sphere_ratio=C.sphere_ratio))
    if a.resume:
        model = PPO.load(a.resume, env=env, device="cpu", tensorboard_log=tb)
    else:
        model = PPO(AsymmetricPolicy, env, n_steps=C.n_steps, batch_size=C.batch_size, n_epochs=C.n_epochs,
                    learning_rate=C.learning_rate, gamma=0.0, gae_lambda=1.0, ent_coef=0.0,
                    policy_kwargs=dict(net_arch=list(C.net_arch), log_std_init=C.log_std_init),
                    tensorboard_log=tb, verbose=1, seed=C.seed, device="cpu")   # 小さなネットワークなので CPU の方が速い
    cbs = [LogAndVideo(out / "videos"),
           CheckpointCallback(max(1, C.checkpoint_every // C.n_envs), str(out / "checkpoints"), name_prefix="ppo")]
    remaining = C.total_steps - model.num_timesteps
    model.learn(remaining, callback=cbs, tb_log_name=C.name, reset_num_timesteps=not a.resume)
    model.save(out / "final")
    print("保存:", out / "final.zip")


if __name__ == "__main__":
    main()
