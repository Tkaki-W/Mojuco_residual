"""球の式（ベースライン）と、ハンドの置き方。単位 mm・度。

腫瘍を半径 r の球とみなして構えを決める式が2つある（楕円体との違いは強化学習の補正に任せる）
  sphere_formula：2本でつまむ（ロール45°、高さ Z＝29 固定）
  press_formula ：1本で押す（ロール0°、ゲルを上から当てる。2本でつまめない小さい・低い腫瘍用）
どちらも Stance（構え）を返し、hand_pose で補正を足してハンドを置く。
"""
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
class Stance:
    """式で決めた構え（補正なし）"""
    R: float           # 腫瘍のてっぺんの高さ
    r: float           # 半径
    Z: float           # ロール軸の高さ（皮膚から）
    roll: float        # ロール角 [度]
    s: float           # ゲル指のスライド
    s_pad: float       # パッド指のスライド
    shift: float       # アームで横（ゲル指側+）にずらす量
    gel_offset: float  # 伸ばした状態のゲル中心を、腫瘍中心から指先側に置く量 (r＋r_g)·sinθ ＋ A(1−cosθ)
    theta: float       # 屈曲1の止まる角度 [度]（予測）
    h_c: float         # 接点の高さ
    gel_low: float     # 止まったときのゲルの一番下の高さ（予測）
    ok: bool           # この式で構えてよいか（sphere_formula：2本でつまめる、press_formula：1本で押せる）


def _gel_offset(r, th, hand: HandConfig):
    """θ 曲げて当たるとき、伸ばした状態のゲル中心を腫瘍中心より指先側に置く量（2本・1本で共通）"""
    return (r + hand.r_g) * math.sin(th) + hand.A * (1 - math.cos(th))


def sphere_formula(R, r, hand: HandConfig = DEFAULT_HAND, bl: BaselineConfig = DEFAULT_BASELINE) -> Stance:
    """球の式（2本でつまむ、修正版）。めり込みは入れない（強化学習の補正に任せる）。

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
    half_u = (hand.gel_x[1] - hand.gel_x[0]) / 2             # 断面方向の半幅 8
    half_v = (hand.gel_y[1] - hand.gel_y[0]) / 2             # 長さ方向の半幅 9
    low = h_c - math.sin(a) * (half_u + half_v * math.sin(th))
    return Stance(R=R, r=r, Z=bl.Z, roll=bl.roll, s=s, s_pad=s + bl.pad_extra_slide, shift=0.0,
                  gel_offset=_gel_offset(r, th, hand), theta=math.degrees(th), h_c=h_c, gel_low=low,
                  ok=r >= bl.min_r and low >= bl.min_gel_low and s >= 0)


def press_formula(R, r, hand: HandConfig = DEFAULT_HAND, bl: BaselineConfig = DEFAULT_BASELINE) -> Stance:
    """1本で押す（2本でつまめない形、主に低い腫瘍）。ロール0°でゲルを下に向け、てっぺんに上から当てる。
    球の式と同じく、腫瘍を半径 r の球とみなす（楕円体の長軸方向の丸みの違いは強化学習の補正に任せる）。

    高さ：伸ばしたゲル面が、てっぺんの press_gap 上に来るように置く（Z＝R＋r_g＋press_gap）
    曲げ角 θ：ゲルが球に当たる条件 A·sinθ − (r＋r_g)(1−cosθ) ＝ press_gap を解く（9mm で約5°）
    前後：2本の式と同じ (r＋r_g)·sinθ ＋ A(1−cosθ)
    横：ゲル指のスライドは0にして、アームでゲル中心を腫瘍の真上にずらす（−21.3）。パッド指は外に逃がす（スライド上限）
    判定：ゲルの指先側の縁の高さ h_c − 9·sinθ ≥ min_gel_low
    """
    g = bl.press_gap
    th = math.asin(g / hand.A)
    for _ in range(5):                                      # 上の条件を少しずつ直して解く（すぐ収まる）
        th = math.asin((g + (r + hand.r_g) * (1 - math.cos(th))) / hand.A)
    h_c = R - r * (1 - math.cos(th))
    half_v = (hand.gel_y[1] - hand.gel_y[0]) / 2             # 長さ方向の半幅 9
    low = h_c - half_v * math.sin(th)
    return Stance(R=R, r=r, Z=R + hand.r_g + g, roll=0.0, s=0.0, s_pad=hand.slide_max, shift=-hand.marker_center[0],
                  gel_offset=_gel_offset(r, th, hand), theta=math.degrees(th), h_c=h_c, gel_low=low,
                  ok=low >= bl.min_gel_low)


@dataclass
class Pose:
    pos: np.ndarray    # ハンド原点の位置 [mm, 世界]
    quat: np.ndarray   # ハンドの向き
    slide: float       # ゲル指のスライド [mm]
    slide_pad: float   # パッド指のスライド [mm]
    roll: float        # 度（左右対称）


def hand_pose(sr: Stance, act=(0.0, 0.0, 0.0, 0.0), center=(0.0, 0.0), yaw=0.0,
              hand: HandConfig = DEFAULT_HAND) -> Pose:
    """式の構え sr ＋補正 act=(dx: 指先側+, dy: ゲル指側+, dz: 上+, droll: 度)。

    ロール軸の中心線上でゲル断面の点（CAD: x=0, y=ゲル中心, z=ロール軸）を、
    腫瘍の中心 center から指先側へ gel_offset、横へ shift、高さ Z に置く。ハンドは傾けない（向きは z 軸まわりの yaw [度] だけ）。
    center と yaw は「推定した」中心と短軸の向き（位置のずれ・向きの誤差を入れるときに使う。腫瘍は原点・向き0）。
    """
    dx_a, dy_a, dz_a, droll = act
    t = math.radians(yaw)
    Rz = np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])
    p_cad = np.array([0.0, hand.marker_center[1], hand.roll_axis_z])
    target = np.array([center[0], center[1], sr.Z]) + Rz @ np.array([sr.shift + dy_a, -(sr.gel_offset + dx_a), dz_a])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, Rz.ravel())
    clip = lambda v: min(hand.slide_max, max(0.0, v))  # noqa: E731
    return Pose(target - Rz @ p_cad, quat, clip(sr.s), clip(sr.s_pad), sr.roll + droll)
