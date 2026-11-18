"""1回つかむ：腫瘍を置く → 構える → 屈曲1を止まるまで閉じる → 接触を集計"""
import json
import math
from dataclasses import dataclass

import mujoco
import numpy as np

from .baseline import Pose
from .config import ASSETS, DEFAULT_HAND, HandConfig
from .contact import ContactSummary, summarize
from .model import MM, build_model


@dataclass
class Tumor:
    """腫瘍（楕円体）。b＝短軸（はさむ方向）、a＝長軸、c＝縦の半径。center_z＝中心の高さ（皮膚から）。単位 mm

    いつも原点に、短軸を世界の x、長軸を y に向けて置く（向きや位置のずれはハンドの側で入れる）。
    MuJoCo（4.0）は、ワールドに直接置いた動かない物体の向き（geom_quat）を書きかえても反映しないため。
    """
    a: float
    b: float
    c: float
    center_z: float

    @classmethod
    def sphere(cls, diameter, embed):
        r = diameter / 2
        return cls(r, r, r, r * (1 - 2 * embed))

    @property
    def top(self):
        return self.center_z + self.c

    @property
    def visible_w(self):
        """皮膚の上に見えている短軸方向の幅"""
        if self.center_z >= 0:
            return 2 * self.b
        return 2 * self.b * math.sqrt(max(0.0, 1 - (self.center_z / self.c) ** 2))


@dataclass
class GraspResult:
    contact: ContactSummary
    flex1_gel: float     # 止まった角度 [度]
    flex1_pad: float
    settled: bool
    time: float
    traj: list           # 再生用 (time, qpos, mocap_pos, mocap_quat)


class GraspSim:
    def __init__(self, cfg: HandConfig = DEFAULT_HAND, visual_mesh: bool = True):
        self.cfg = cfg
        self.m = build_model(cfg, visual_mesh)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.names = [m.geom(i).name for i in range(m.ngeom)]
        self.tumor_gid = m.geom("tumor").id
        self.jid = {n: m.joint(n).id for n in ("gel_slide", "pad_slide", "gel_roll", "pad_roll", "gel_flex1", "pad_flex1")}
        self.eq = {n: m.equality(f"eq_{n}").id for n in ("gel_slide", "pad_slide", "gel_roll", "pad_roll")}
        # 手のひらは mocap（動かない物体）なので、MuJoCo は皮膚・腫瘍との接触を計算しない → 距離で調べる
        # ハンドの後ろ側（CAD の y>-20 にある部品、ゲルから約200mm以上根元側）は腫瘍の近くに来ないので、判定から外す
        back = {mm["name"] for mm in json.loads((ASSETS / "hulls.json").read_text(encoding="utf-8"))
                if mm["group"] == "palm" and mm["bbox_min"][1] > -20}
        self.palm_ids = [i for i, n in enumerate(self.names) if n.startswith("palm") and n not in back]
        self.env_ids = {"skin": [m.geom("skin").id], "tumor": [self.tumor_gid]}

    def palm_hits(self, tol=0.0):
        """手のひらが皮膚・腫瘍にめり込んでいれば、その相手の名前を返す"""
        ft = np.zeros(6)
        return [("palm", name) for name, ids in self.env_ids.items()
                if any(mujoco.mj_geomDistance(self.m, self.d, g, e, 0.01, ft) < tol
                       for e in ids for g in self.palm_ids)]

    # --- 腫瘍 ---
    def set_tumor(self, t: Tumor):
        g = self.tumor_gid
        size = np.array([t.b, t.a, t.c]) * MM      # x＝短軸, y＝長軸
        self.m.geom_size[g] = size
        self.m.geom_rbound[g] = size.max()
        self.m.geom_aabb[g] = np.concatenate([np.zeros(3), size])
        self.m.geom_pos[g] = np.array([0.0, 0.0, t.center_z]) * MM

    # --- 構える ---
    def set_pose(self, p: Pose):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.mocap_pos[0] = p.pos * MM
        d.mocap_quat[0] = p.quat
        vals = {"gel_slide": p.slide * MM, "pad_slide": p.slide_pad * MM,
                "gel_roll": math.radians(p.roll), "pad_roll": math.radians(p.roll)}
        for n, v in vals.items():
            m.eq_data[self.eq[n], 0] = v
            d.qpos[m.jnt_qposadr[self.jid[n]]] = v
        d.ctrl[:] = 0.0
        mujoco.mj_forward(m, d)

    # --- 閉じる ---
    def close(self, record_every: float = 0.005) -> GraspResult:
        m, d, c = self.m, self.d, self.cfg
        d.ctrl[:] = math.radians(c.flex1_target)
        qa = [m.jnt_qposadr[self.jid[n]] for n in ("gel_flex1", "pad_flex1")]
        va = [m.jnt_dofadr[self.jid[n]] for n in ("gel_flex1", "pad_flex1")]
        traj, next_rec = [], 0.0
        settled = False
        lim = math.radians(c.settle_vel)
        n_still = 0
        while d.time < c.close_time:
            mujoco.mj_step(m, d)
            if d.time >= next_rec:
                traj.append((d.time, d.qpos.copy(), d.mocap_pos.copy(), d.mocap_quat.copy()))
                next_rec += record_every
            if d.time > 0.05 and all(abs(d.qvel[v]) < lim for v in va):
                n_still += 1
                if n_still * m.opt.timestep > 0.02:   # 20ms 止まっていたら終わり
                    settled = True
                    break
            else:
                n_still = 0
        traj.append((d.time, d.qpos.copy(), d.mocap_pos.copy(), d.mocap_quat.copy()))
        cs = summarize(m, d, self.names, c)
        cs.forbidden += [h for h in self.palm_hits() if h not in cs.forbidden]
        return GraspResult(cs, math.degrees(d.qpos[qa[0]]), math.degrees(d.qpos[qa[1]]), settled, d.time, traj)

    def grasp(self, tumor: Tumor, pose: Pose) -> GraspResult:
        self.set_tumor(tumor)
        self.set_pose(pose)
        return self.close()
