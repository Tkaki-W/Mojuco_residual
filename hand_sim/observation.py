"""観測：十字の高さプロファイル（深度カメラの代わり）と、そこから求める楕円・球の式の値。単位 mm。

座標（プロファイルの向き）：u＝短軸方向（ゲル指側が＋）、v＝長軸方向（指先側が＋）。
推定した腫瘍の中心と向き（キャリブレーションのずれ・向きの誤差を含む）を原点にとる。
"""
import math

import numpy as np

from .baseline import radius_from_profile

U = np.arange(-16, 16.01, 2.0)   # 短軸方向 17点
V = np.arange(-21, 21.01, 3.0)   # 長軸方向 15点（長径は最大17.5なので ±21 まで測る）
BAND = (-1.0, 0.0, 1.0)          # 線の幅 ±1 の帯を平均
EDGE = 1.0                       # これより高ければ「腫瘍の上」とみなす（幅を測るとき）


def height(t, x, y):
    """世界の点 (x, y) での高さ（皮膚から）。腫瘍は原点にあり、短軸＝x・長軸＝y。上から見えるところだけ"""
    q = 1 - (x / t.b) ** 2 - (y / t.a) ** 2
    return max(0.0, t.center_z + t.c * math.sqrt(q)) if q > 0 else 0.0


def profile(t, center, yaw_deg, rng=None, sigma=0.0, drop=0.0):
    """十字の高さプロファイル（短軸17点, 長軸15点）。ノイズ・欠けつき"""
    yaw = math.radians(yaw_deg)
    eu = np.array([math.cos(yaw), math.sin(yaw)])          # 短軸（ゲル指側）
    ev = np.array([math.sin(yaw), -math.cos(yaw)])         # 長軸（指先側＝ハンドの −y）
    c = np.asarray(center, float)

    def h(u, v):
        p = c + u * eu + v * ev
        return height(t, p[0], p[1])

    pu = np.array([np.mean([h(u, w) for w in BAND]) for u in U])
    pv = np.array([np.mean([h(w, v) for w in BAND]) for v in V])
    if rng is not None:
        out = []
        for p, x in ((pu, U), (pv, V)):
            p = p + rng.normal(0, sigma, p.shape)
            keep = rng.random(p.shape) >= drop
            keep[[0, -1]] = True                            # 端は残す（補間のため）
            out.append(np.interp(x, x[keep], p[keep]))      # 欠けたところは両隣から補間
        pu, pv = out
    return pu, pv


def _width(x, p):
    """プロファイルが EDGE を超えている幅。
    端は、内側の点の高さの半分になる位置を線形補間で求める（縁が崖でも、なだらかでも偏りが小さい）"""
    above = np.where(p > EDGE)[0]
    if len(above) == 0:
        return 0.0
    i0, i1 = above[0], above[-1]
    lo = x[i0] if i0 == 0 else np.interp(p[i0] / 2, [p[i0 - 1], p[i0]], [x[i0 - 1], x[i0]])
    hi = x[i1] if i1 == len(x) - 1 else np.interp(p[i1] / 2, [p[i1 + 1], p[i1]], [x[i1 + 1], x[i1]])
    return float(hi - lo)


def estimate(pu, pv):
    """プロファイルから R（てっぺん）、楕円の a・b（見えている半幅）、球の式に入れる r"""
    smooth = np.convolve(pu, np.ones(3) / 3, mode="same")
    R = float(max(smooth.max(), 0.1))
    w_b = _width(U, pu)
    a, b = _width(V, pv) / 2, w_b / 2
    r = radius_from_profile(w_b, R, R < w_b / 2)          # R < w/2 ⇔ 中心が皮膚より下（埋没50%超）
    return R, r, a, b
