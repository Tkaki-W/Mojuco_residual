"""接触の集計：圧力中心・タイルごとの力・禁止の接触"""
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .config import DEFAULT_HAND, FORBIDDEN, GEL_SKIN_RATIO, HandConfig


def _role(name: str) -> str:
    for k in ("gel_tile", "gel_rim", "sensor_frame", "gelfinger_body", "pad", "palm", "tumor", "skin"):
        if name.startswith(k):
            return k
    return name


@dataclass
class ContactSummary:
    gel_force: float = 0.0               # ゲルが腫瘍から受ける垂直の力の合計 [N]
    cop: np.ndarray = field(default_factory=lambda: np.full(2, np.nan))  # 圧力中心 [mm]（u: 断面方向 外側+, v: 長さ方向 指先側+）
    cop_dist: float = np.nan             # 圧力中心とゲル中心の距離 [mm]
    tile_force: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))  # [断面方向, 長さ方向]
    rim_force: float = 0.0
    n_tiles: int = 0
    max_pen: float = 0.0                 # ゲルと腫瘍の一番深いめり込み [mm]
    force_angle: float = np.nan          # 力の向きとゲル法線の角度 [度]（記録用）
    pad_force: float = 0.0
    gel_skin_force: float = 0.0          # ゲルが皮膚から受ける力 [N]
    gel_skin_ratio: float = 0.0          # ゲルにかかる力全体のうち皮膚からの割合
    forbidden: list = field(default_factory=list)   # (部品, 相手)
    gel_on_tumor: bool = False


def summarize(m: mujoco.MjModel, d: mujoco.MjData, names, hand: HandConfig = DEFAULT_HAND,
              tile_threshold: float = 1e-3) -> ContactSummary:
    out = ContactSummary()
    bid = m.body("gel_link2").id
    Rb, pb = d.xmat[bid].reshape(3, 3), d.xpos[bid]
    center = np.array([hand.marker_center[0], hand.marker_center[1]]) * 1e-3
    f6 = np.zeros(6)
    wsum, psum, fvec = 0.0, np.zeros(2), np.zeros(3)
    for i in range(d.ncon):
        c = d.contact[i]
        n1, n2 = names[c.geom1], names[c.geom2]
        r1, r2 = _role(n1), _role(n2)
        for a, b in ((r1, r2), (r2, r1)):
            if b in FORBIDDEN.get(a, ()):
                pair = (a, b)
                if pair not in out.forbidden:
                    out.forbidden.append(pair)
        mujoco.mj_contactForce(m, d, i, f6)
        fn = f6[0]
        if "skin" in (r1, r2) and ({r1, r2} & {"gel_tile", "gel_rim"}):
            out.gel_skin_force += fn
        if "tumor" not in (r1, r2):
            continue
        other, oname = (r1, n1) if r2 == "tumor" else (r2, n2)
        if other == "pad":
            out.pad_force += fn
        if other not in ("gel_tile", "gel_rim"):
            continue
        out.gel_on_tumor = True
        local = Rb.T @ (c.pos - pb)          # gel_link2 の座標（＝CAD座標、m）
        u = (local[0] - center[0]) * 1e3      # 断面方向（外側＝+x が+）
        v = -(local[1] - center[1]) * 1e3     # 長さ方向（指先側が+）
        wsum += fn
        psum += fn * np.array([u, v])
        out.max_pen = max(out.max_pen, -c.dist * 1e3)
        # 力のベクトル（世界座標）：接触フレームの1行目が法線
        frame = c.frame.reshape(3, 3)
        fw = frame.T @ f6[:3]
        fvec += fw if names[c.geom2] == oname else -fw   # ゲルが受ける力（法線は geom1→geom2）
        if other == "gel_tile":
            i_, j_ = int(oname[-2]), int(oname[-1])
            out.tile_force[i_, j_] += fn
        else:
            out.rim_force += fn
    out.gel_force = wsum
    tot = wsum + out.gel_skin_force
    if out.gel_skin_force > 0:
        out.gel_skin_ratio = out.gel_skin_force / tot
        if out.gel_skin_ratio > GEL_SKIN_RATIO:
            out.forbidden.append(("gel", "skin"))
    if wsum > 0:
        out.cop = psum / wsum
        out.cop_dist = float(np.linalg.norm(out.cop))
        gel_normal = Rb @ np.array([0, 0, -1.0])     # ゲルの外向き法線
        f_on_gel = fvec / (np.linalg.norm(fvec) + 1e-12)
        out.force_angle = float(np.degrees(np.arccos(np.clip(-f_on_gel @ gel_normal, -1, 1))))
    out.n_tiles = int((out.tile_force > tile_threshold).sum())
    return out
