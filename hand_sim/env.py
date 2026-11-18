"""Gym 環境：1エピソード＝1回つかむ（1ステップ）。

観測（Actor）：十字の高さプロファイル 36点、式の値（r, R, s, θ）、楕円の a, b        → 42個
Critic だけ  ：本当の形（a, b, c, 中心の高さ）、キャリブレーションのずれ（x, y）、向きの誤差、摩擦 → 8個
観測の並び：[Actor の 42個 | Critic だけの 8個]（policy.py で Actor 側は後ろ8個を 0 にする）
行動：[-1,1]^4 → 指の長さ方向 ±3、はさむ方向 ±3、高さ ±2 [mm]、ロール ±10°
腫瘍はいつも原点・向き0（grasp.py の Tumor）。位置のずれ・向きの誤差は「推定した中心・向き」として、ハンドの置き方とプロファイルにだけ入れる
"""
import gymnasium as gym
import numpy as np

from .baseline import hand_pose, press_formula, radius_from_profile, sphere_formula
from .grasp import Bump, GraspSim, Tumor
from .observation import U, V, estimate, profile

N_ACTOR, N_PRIV = len(U) + len(V) + 6, 8   # プロファイル＋（r, R, s, θ, a, b）
ACT_SCALE = np.array([3.0, 3.0, 2.0, 10.0])
F_MIN = 0.5            # これより弱い押し方は「かすっただけ」[N]


def random_bump(rng, t: Tumor) -> Bump:
    """こぶ：本体の40〜70%の大きさ。長軸方向の前か後ろへ（長軸の半径の40〜70%）、横へ（短軸の半径の±30%）ずらす。
    こぶのてっぺんは本体のてっぺんの40〜80%の高さ"""
    k = rng.uniform(0.4, 0.7)
    y = (1 if rng.random() < 0.5 else -1) * rng.uniform(0.4, 0.7) * t.a
    x = rng.uniform(-0.3, 0.3) * t.b
    top = rng.uniform(0.4, 0.8) * t.top
    return Bump(k * t.a, k * t.b, k * t.c, x, y, top - k * t.c)


def sample_tumor(rng, kind=None, shapes=None, fingers=2):
    """腫瘍をランダムに作る。fingers=2 なら2本でつまめる形（本体の楕円体で球の式の判定を満たす）、
    fingers=1 なら2本ではつまめず、1本で押せる形（press_formula の判定を満たす）だけ。
    shapes＝形と割合（例 {"sphere": 0.2, "ellipsoid": 0.8}）。kind を指定すると、その形だけ"""
    shapes = shapes or {"sphere": 0.5, "ellipsoid": 0.5}
    names, cum = list(shapes), np.cumsum(list(shapes.values()))
    while True:
        k = kind or names[int(np.searchsorted(cum / cum[-1], rng.random(), side="right"))]
        b = rng.uniform(7.5, 12.5)
        a, c = (b, b) if k == "sphere" else (b * rng.uniform(1.0, 2.0), b * rng.uniform(0.7, 1.0))
        e = rng.uniform(0.25, 0.75)
        rng.uniform(0, 180)   # もとは腫瘍の向き（いまは0に固定）。前の評価と同じ腫瘍になるように乱数だけ引く
        t = Tumor(a, b, c, c * (1 - 2 * e))
        if k == "asym":
            t.bump = random_bump(rng, t)
        r = radius_from_profile(t.visible_w, t.top, e > 0.5)
        two = sphere_formula(t.top, r).ok
        if fingers == 2 and two:
            return k, t
        if fingers == 1 and not two and press_formula(t.top, r).ok:
            return k, t


class GraspEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, kind=None, randomize=True, visual_mesh=False, shapes=None, fingers=2):
        """fingers=2：2本でつまむ（sphere_formula、run6 まで）。fingers=1：1本で押す（press_formula、run7）"""
        self.kind, self.randomize, self.shapes, self.fingers = kind, randomize, shapes, fingers
        self.sim = GraspSim(visual_mesh=visual_mesh)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (N_ACTOR + N_PRIV,), np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (4,), np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.shape_kind, self.tumor = sample_tumor(rng, self.kind, self.shapes, self.fingers)
        t = self.tumor
        r_ = self.randomize
        calib = rng.uniform(-1, 1, 2) if r_ else np.zeros(2)          # 位置のずれ ±1
        yaw_err = rng.uniform(-3, 3) if r_ else 0.0                    # 楕円の向きの誤差 ±3°
        sigma, drop = (rng.uniform(0.3, 0.7), rng.uniform(0, 0.1)) if r_ else (0.0, 0.0)
        mu = rng.uniform(0.5, 1.0) if r_ else 0.8                      # 摩擦
        # MuJoCo は接触する2つの摩擦の大きい方を使うので、ハンドだけでなく腫瘍・皮膚も同じ値にする
        self.sim.m.geom_friction[:, 0] = mu

        # 推定した中心・向き（腫瘍は原点・向き0なので、ずれそのもの）で測ったプロファイル → 球の式
        self.center = calib
        self.yaw = yaw_err
        pu, pv = profile(t, self.center, self.yaw, rng if r_ else None, sigma, drop)
        R, r, a, b = estimate(pu, pv)
        self.sr = (sphere_formula if self.fingers == 2 else press_formula)(R, r)
        actor = np.concatenate([pu, pv, [r, R, self.sr.s, self.sr.theta, a, b]]) / 10.0
        priv = np.array([t.a, t.b, t.c, t.center_z, *(calib * 10), yaw_err, mu * 10]) / 10.0
        self.obs = np.nan_to_num(np.concatenate([actor, priv])).astype(np.float32)
        return self.obs, {}

    def step(self, action):
        act = np.clip(action, -1, 1) * ACT_SCALE
        pose = hand_pose(self.sr, act=tuple(act), center=self.center, yaw=self.yaw)
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
                "forbidden": bool(c.forbidden), "miss": miss, "kind": self.shape_kind}
        return self.obs, float(reward), True, False, info
