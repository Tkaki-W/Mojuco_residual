"""球の式（ベースライン）と、ハンドの置き方。単位 mm・度。"""
import math
from dataclasses import dataclass

import mujoco
import numpy as np

from .config import DEFAULT_BASELINE, DEFAULT_HAND, BaselineConfig, HandConfig


def radius_from_profile(w, R, embed_over_half: bool):
    """見えている幅 w と てっぺんの高さ R から半径 r（埋没50%超は弦から求める）"""
    if embed_over_half:
        return (w * w / 4 + R * R) / (2 * R)
    return w / 2


@dataclass
class SphereResult:
    R: float
    r: float
    h_c: float         # 接点の高さ
    dz: float
    s: float           # ゲル指のスライド
    s_pad: float       # パッド指のスライド
    theta: float       # 止まる角度 [度]
    adj: float         # A(1−cosθ)
    dx: float          # (r＋r_g)·sinθ
    gel_low: float     # 止まったときのゲルの一番下の高さ（予測）
    two_finger: bool

    @property
    def gel_offset(self):
        """伸ばした状態のゲル中心を、腫瘍中心から指先側に置く量"""
        return self.dx + self.adj


def sphere_formula(R, r, hand: HandConfig = DEFAULT_HAND, bl: BaselineConfig = DEFAULT_BASELINE) -> SphereResult:
    """球の式（修正版）。めり込みは入れない（強化学習の補正に任せる）。

    もとの式からの変更
      スライド：s ＝ dz ＋ r·cos45° − 20 − 1.3/cos45°（ゲル窓が断面方向に外側へ1.3mmずれている分）
      前後　　：(r ＋ r_g)·sinθ ＋ A(1 − cosθ)（もとは r·sinθ。ゲル表面が屈曲1の軸から r_g 離れている分が抜けていた）
      パッド指：s ＋ pad_extra_slide（ゲル指と干渉しないように外に開く）
    2本でつまむ判定（すべて満たす）
      r ≥ min_r、ゲルの一番下 h_c − sin45°·(8 ＋ 9·sinθ) ≥ min_gel_low、s ≥ 0
    """
    a = math.radians(bl.roll)
    off = hand.marker_center[0] - hand.roll_axis_x          # ゲル窓のずれ 1.3
    h_c = R + r * (math.sin(a) - 1)
    dz = bl.Z - h_c
    s = dz + r * math.cos(a) - hand.roll_axis_x - off / math.cos(a)
    th = math.asin(max(-1.0, min(1.0, (dz / math.sin(a) - hand.r_g) / hand.A)))
    adj = hand.A * (1 - math.cos(th))
    dx = (r + hand.r_g) * math.sin(th)
    half_u = (hand.gel_x[1] - hand.gel_x[0]) / 2             # 断面方向の半幅 8
    half_v = (hand.gel_y[1] - hand.gel_y[0]) / 2             # 長さ方向の半幅 9
    low = h_c - math.sin(a) * (half_u + half_v * math.sin(th))
    two = r >= bl.min_r and low >= bl.min_gel_low and s >= 0
    return SphereResult(R, r, h_c, dz, s, s + bl.pad_extra_slide, math.degrees(th), adj, dx, low, two)


@dataclass
class Pose:
    pos: np.ndarray    # ハンド原点の位置 [mm, 世界]
    quat: np.ndarray   # ハンドの向き
    slide: float       # ゲル指のスライド [mm]
    slide_pad: float   # パッド指のスライド [mm]
    roll: float        # 度（左右対称）


def hand_pose(sr: SphereResult, act=(0.0, 0.0, 0.0, 0.0), center=(0.0, 0.0), yaw=0.0,
              hand: HandConfig = DEFAULT_HAND, bl: BaselineConfig = DEFAULT_BASELINE) -> Pose:
    """球の式の構え＋補正 act=(dx: 指先側+, dy: ゲル指側+, dz: 上+, droll: 度)。

    ロール軸の中心線上でゲル断面の点（CAD: x=0, y=ゲル中心, z=ロール軸）を、
    腫瘍の中心 center から指先側へ gel_offset、高さ Z に置く。ハンドは傾けない（向きは z 軸まわりの yaw [度] だけ）。
    center と yaw は「推定した」中心と短軸の向き（位置のずれ・向きの誤差を入れるときに使う。腫瘍は原点・向き0）。
    """
    dx_a, dy_a, dz_a, droll = act
    t = math.radians(yaw)
    Rz = np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])
    p_cad = np.array([0.0, hand.marker_center[1], hand.roll_axis_z])
    target = np.array([center[0], center[1], bl.Z]) + Rz @ np.array([dy_a, -(sr.gel_offset + dx_a), dz_a])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, Rz.ravel())
    clip = lambda v: min(hand.slide_max, max(0.0, v))  # noqa: E731
    return Pose(target - Rz @ p_cad, quat, clip(sr.s), clip(sr.s_pad), bl.roll + droll)
