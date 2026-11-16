"""Gym 環境：1エピソード＝1回つかむ（1ステップ）。

観測（Actor）：十字の高さプロファイル 28点、球の式の値（r, R, s, θ）、楕円の a, b         → 34個
Critic だけ  ：本当の形（a, b, c, 中心の高さ）、キャリブレーションのずれ（x, y）、向きの誤差、摩擦 → 8個
観測の並び：[Actor の 34個 | Critic だけの 8個]（policy.py で Actor 側は後ろ8個を 0 にする）
行動：[-1,1]^4 → 指の長さ方向 ±3、はさむ方向 ±3、高さ ±2 [mm]、ロール ±10°
"""
import math

import gymnasium as gym
import numpy as np

from .baseline import hand_pose, radius_from_profile, sphere_formula
from .grasp import GraspSim, Tumor
from .observation import estimate, profile

N_ACTOR, N_PRIV = 34, 8
ACT_SCALE = np.array([3.0, 3.0, 2.0, 10.0])
F_MIN = 0.5            # これより弱い押し方は「かすっただけ」[N]


def sample_tumor(rng, kind=None):
    """2本でつまめる（本当の形で球の式の判定を満たす）腫瘍をランダムに作る"""
    while True:
        k = kind or ("sphere" if rng.random() < 0.5 else "ellipsoid")
        b = rng.uniform(7.5, 12.5)
        a, c = (b, b) if k == "sphere" else (b * rng.uniform(1.0, 1.4), b * rng.uniform(0.7, 1.0))
        e = rng.uniform(0.25, 0.75)
        t = Tumor(a, b, c, c * (1 - 2 * e), yaw=rng.uniform(0, 180))
        w = 2 * b if t.center_z >= 0 else 2 * b * math.sqrt(1 - (t.center_z / c) ** 2)
        if sphere_formula(t.top, radius_from_profile(w, t.top, e > 0.5)).two_finger:
            return k, t


class GraspEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, kind=None, randomize=True, visual_mesh=False):
        self.kind, self.randomize = kind, randomize
        self.sim = GraspSim(visual_mesh=visual_mesh)
        m = self.sim.m
        self.hand_geoms = [i for i, n in enumerate(self.sim.names) if n not in ("skin", "tumor")]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (N_ACTOR + N_PRIV,), np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (4,), np.float32)
        self._friction0 = m.geom_friction[self.hand_geoms, 0].copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.shape_kind, self.tumor = sample_tumor(rng, self.kind)
        t = self.tumor
        r_ = self.randomize
        calib = rng.uniform(-1, 1, 2) if r_ else np.zeros(2)          # 位置のずれ ±1
        yaw_err = rng.uniform(-3, 3) if r_ else 0.0                    # 楕円の向きの誤差 ±3°
        sigma, drop = (rng.uniform(0.3, 0.7), rng.uniform(0, 0.1)) if r_ else (0.0, 0.0)
        mu = rng.uniform(0.5, 1.0) if r_ else 0.8                      # 摩擦
        self.sim.m.geom_friction[self.hand_geoms, 0] = mu

        # 推定した中心・向きで測ったプロファイル → 球の式
        self.center = np.array(t.xy) + calib
        self.yaw = t.yaw + yaw_err
        pu, pv = profile(t, self.center, self.yaw, rng if r_ else None, sigma, drop)
        R, r, a, b = estimate(pu, pv)
        self.sr = sphere_formula(R, r)
        actor = np.concatenate([pu, pv, [r, R, self.sr.s, self.sr.theta, a, b]]) / 10.0
        priv = np.array([t.a, t.b, t.c, t.center_z, *(calib * 10), yaw_err, mu * 10]) / 10.0
        self.obs = np.nan_to_num(np.concatenate([actor, priv])).astype(np.float32)
        return self.obs, {}

    def step(self, action):
        act = np.clip(action, -1, 1) * ACT_SCALE
        pose = hand_pose(self.center, self.yaw, self.sr, act=tuple(act))
        self.result = res = self.sim.grasp(self.tumor, pose)
        c = res.contact
        miss = not c.gel_force > 0          # ゲルが腫瘍から力を受けていない＝空振り
        weak = c.gel_force < F_MIN
        success = bool(not miss and not weak and c.cop_dist < 1.0 and not c.forbidden)
        if miss:
            reward = -2.0
        else:
            reward = -min(c.cop_dist, 5.0) / 2 - 1.0 * bool(c.forbidden) - 0.5 * weak + 1.0 * success
        info = {"success": success, "cop_dist": float(c.cop_dist) if not miss else 10.0,
                "forbidden": bool(c.forbidden), "miss": miss, "sphere": self.shape_kind == "sphere"}
        return self.obs, float(reward), True, False, info
