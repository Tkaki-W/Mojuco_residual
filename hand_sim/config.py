"""寸法と設定（単位：mm・度。MuJoCo に渡すときに m・rad に直す）

座標は CAD と同じ（ハンドの座標系）：
  x＝横（指先側から見て右が＋、ゲル指が＋側）、y＝指の長さ方向（指先が −y）、z＝上下（手の甲が＋、ゲルは −z を向く）
"""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


@dataclass
class HandConfig:
    # --- 関節の位置（CAD） ---
    roll_axis_x: float = 20.0      # ゲル指 +20、パッド指 -20（スライド0のとき）
    roll_axis_z: float = -19.0
    mount_y: float = -73.2         # 取り付け面（屈曲1から38.5根元側）
    flex1_y: float = -111.7
    flex2_y: float = -158.7        # 屈曲1から47

    # --- ゲル（ゲル指） ---
    window_face_z: float = -29.2   # ゲル窓の面（ロール軸の10.2下）
    gel_x: tuple = (13.3, 29.3)    # ゲル窓の面の範囲（断面方向16）
    gel_y: tuple = (-226.7, -208.7)  # 同（長さ方向18）
    marker_center: tuple = (21.3, -217.7)  # ガラス窓10×10の中心＝タイルの中心＝ゲル中心
    gel_t: float = 2.0             # ゲルの厚み（窓の面からの出っ張り）。固定
    tile: float = 1.5              # タイル1枚の大きさ（マーカーの間隔）

    # --- パッド（パッド指） ---
    pad_face_z: float = -34.26     # パッド面（ロール軸の15.26下）
    pad_center: tuple = (-18.7, -217.7)

    # --- 動き ---
    slide_max: float = 25.0        # スライドの上限【仮】
    flex1_torque: float = 0.15     # 屈曲1のトルク上限 [N·m]【仮】
    flex1_target: float = 60.0     # 屈曲1の目標角（ここまで閉じようとする。止まるまで閉じるための大きめの値）
    flex1_speed: float = 60.0      # 何もないときの閉じる速さ [deg/s] の目安（関節の粘性で決める）
    close_time: float = 1.5        # 閉じる時間の上限 [s]
    settle_vel: float = 0.5        # これより遅く [deg/s] なったら「止まった」

    # --- 接触（めり込み 0.5〜1mm を目標に調整） ---
    contact_solref: tuple = (-300.0, -8.0)    # 負＝(剛性 N/m, 減衰)。φ20・25%で約0.7mm めり込む
    contact_solimp: tuple = (0.95, 0.99, 0.001, 0.5, 2.0)
    friction: float = 0.8

    # --- シミュレーション ---
    timestep: float = 0.0005       # 1ms だと接触が不安定になる

    @property
    def r_g(self):
        """ゲル指：ロール軸→ゲル表面"""
        return self.roll_axis_z - (self.window_face_z - self.gel_t)

    @property
    def A(self):
        """屈曲1→ゲル中心（長さ方向）"""
        return self.flex1_y - self.marker_center[1]


@dataclass
class BaselineConfig:
    Z: float = 29.0          # ロール軸の高さ（皮膚から、ゲル直上の断面で）。scripts/z_sweep.py で選んだ
    roll: float = 45.0
    # パッド指はゲル指より pad_extra_slide [mm] 外に開いてから閉じる（パッド指の当たり方は問わず、ゲル指との干渉を避けるためだけの値）。
    # 根拠：scripts/pad_margin.py（Z=29、2本でつまむ29形状）で、+2〜+8 なら全部成功。+1.5 以下は干渉、+10 はパッド指が届かない形が出る。
    # その真ん中寄りの値として 4。
    pad_extra_slide: float = 4.0
    min_r: float = 10.0          # 2本でつまむ判定：r の下限（指の本体の内側への張り出し）
    min_gel_low: float = 0.2     # 2本でつまむ判定：ゲルの一番下の高さ（皮膚とのすき間の予測）の下限 [mm]。φ20・50% が入る値


# 当たったら失敗になる組み合わせ（部品の名前の頭 → 相手）
FORBIDDEN = {
    # ゲル×皮膚は、皮膚からの力の割合で別に判定する（contact.py）
    "gel_tile": {"pad"},
    "gel_rim": {"pad"},
    "sensor_frame": {"tumor", "skin", "pad"},
    "gelfinger_body": {"tumor", "skin", "pad"},   # 左右の指どうしがぶつかるのも失敗
    # パッド指は皮膚に触れてよい（当たり方も問わない）。ゲル指との干渉だけ失敗（上の "pad"）
    "palm": {"tumor", "skin"},
}
GEL_SKIN_RATIO = 0.2   # ゲルが皮膚から受ける力が、ゲルにかかる力全体のこの割合以下なら許す

DEFAULT_HAND = HandConfig()
DEFAULT_BASELINE = BaselineConfig()
