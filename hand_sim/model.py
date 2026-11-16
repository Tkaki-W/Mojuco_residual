"""MuJoCo のモデル（MJCF）を組み立てる。

体の作り（すべての body の原点はハンドの原点＝CAD の原点。関節は pos で CAD の位置に置く）
  hand（mocap：位置と向きを直接決める）… 手のひら
   ├ gel_base   ← gel_slide（+x へスライド） ← gel_roll（ロール）
   │  └ gel_link1 ← gel_flex1（屈曲1）
   │     └ gel_link2 ← gel_flex2（屈曲2、0°で固定）… センサの枠・ゲル（タイル9枚＋ゲル外周）
   └ pad_base / pad_link1 / pad_link2（左右反対）

衝突のビット：皮膚・腫瘍 conaffinity=1／ゲル指 contype=1, conaffinity=4／パッド指 contype=5／手のひら contype=1
  → 指と皮膚・腫瘍、ゲル指とパッド指はぶつかる。手のひらと指、同じ指の部品どうしはぶつからない

部品の名前（接触の判定に使う）
  gel_tile_<i><j>（i: 断面方向 -1,0,+1 → 0,1,2、j: 長さ方向 指先側が2）、gel_rim_*、sensor_frame、
  gelfinger_body_*、pad_*、palm_*、tumor、skin
"""
import json
import math

import mujoco
import numpy as np

from .config import ASSETS, DEFAULT_HAND, HandConfig

MM = 0.001


def _v(*xs):
    return " ".join(f"{x:.6g}" for x in xs)


def _hulls():
    meta = json.loads((ASSETS / "hulls.json").read_text(encoding="utf-8"))
    verts = np.load(ASSETS / "hulls.npz")
    return meta, verts


def build_xml(cfg: HandConfig = DEFAULT_HAND, visual_mesh: bool = True) -> str:
    meta, verts = _hulls()
    c = cfg
    sr = _v(*c.contact_solref)
    si = _v(*c.contact_solimp)

    assets, link_geoms = [], {}
    # 見た目のメッシュ
    vis = {"palm": "palm_palm", "gel_base": "gel_base", "gel_link1": "gel_link1", "gel_link2": "gel_link2",
           "pad_base": "pad_base", "pad_link1": "pad_link1", "pad_link2": "pad_link2"}
    color = {"palm": "0.55 0.55 0.6 1", "gel": "0.85 0.85 0.88 1", "pad": "0.75 0.75 0.8 1"}
    if visual_mesh:
        for key, f in vis.items():
            grp = key.split("_")[0]
            for p in sorted((ASSETS / "meshes").glob(f"{f}_*.stl")):
                assets.append(f'<mesh name="vis_{p.stem}" file="{p.as_posix()}" scale="{MM} {MM} {MM}"/>')
                link_geoms.setdefault(key, []).append(
                    f'<geom class="visual" mesh="vis_{p.stem}" rgba="{color[grp]}"/>')
    # 衝突用の凸包
    for m in meta:
        name = m["name"]
        v = verts[name] * MM
        assets.append(f'<mesh name="hull_{name}" vertex="{" ".join(f"{x:.5f}" for x in v.ravel())}" maxhullvert="64"/>')
        key = "palm" if m["group"] == "palm" else f'{m["group"]}_{m["link"]}'
        cls = "frame" if name == "sensor_frame" else ("col_pad" if m["group"] == "pad" else
                                                      ("col_gel" if m["group"] == "gel" else "col"))
        link_geoms.setdefault(key, []).append(f'<geom name="{name}" class="{cls}" mesh="hull_{name}"/>')

    # ゲル（タイル9枚＋ゲル外周4枚）
    mx, my = c.marker_center
    zc = (c.window_face_z - c.gel_t / 2) * MM
    hz = c.gel_t / 2 * MM
    h = c.tile / 2
    gel = []
    for i in range(3):          # 断面方向（x）
        for j in range(3):      # 長さ方向（指先側 = -y が j=2）
            x = mx + (i - 1) * c.tile
            y = my - (j - 1) * c.tile
            gel.append(f'<geom name="gel_tile_{i}{j}" class="gel" type="box" pos="{_v(x*MM, y*MM, zc)}" '
                       f'size="{_v(h*MM, h*MM, hz)}" rgba="0.15 0.45 0.95 1"/>')
    x0, x1 = c.gel_x
    y0, y1 = c.gel_y
    tx0, tx1, ty0, ty1 = mx - 3 * h, mx + 3 * h, my - 3 * h, my + 3 * h
    rims = [  # (x範囲, y範囲)
        ((x0, x1), (y0, ty0)), ((x0, x1), (ty1, y1)),
        ((x0, tx0), (ty0, ty1)), ((tx1, x1), (ty0, ty1)),
    ]
    for k, ((a, b), (p, q)) in enumerate(rims):
        gel.append(f'<geom name="gel_rim_{k}" class="gel" type="box" pos="{_v((a+b)/2*MM, (p+q)/2*MM, zc)}" '
                   f'size="{_v((b-a)/2*MM, (q-p)/2*MM, hz)}" rgba="0.6 0.8 1 0.9"/>')
    link_geoms.setdefault("gel_link2", []).extend(gel)
    gel_site = f'<site name="gel_center" pos="{_v(mx*MM, my*MM, (c.window_face_z - c.gel_t)*MM)}" size="0.0008" rgba="1 0 0 1"/>'
    pad_site = f'<site name="pad_center" pos="{_v(c.pad_center[0]*MM, c.pad_center[1]*MM, c.pad_face_z*MM)}" size="0.0008" rgba="1 0 0 1"/>'

    def finger(side):
        s = 1 if side == "gel" else -1
        rx = s * c.roll_axis_x * MM
        rz = c.roll_axis_z * MM
        inert = '<inertial pos="{} {} {}" mass="0.02" diaginertia="1e-5 1e-5 1e-5"/>'
        g = lambda k: "\n".join(link_geoms.get(f"{side}_{k}", []))  # noqa: E731
        site = gel_site if side == "gel" else pad_site
        return f"""
    <body name="{side}_base" gravcomp="1">
      {inert.format(rx, (c.mount_y + c.flex1_y) / 2 * MM, rz)}
      <joint name="{side}_slide" type="slide" axis="{s} 0 0" range="0 {c.slide_max*MM}" limited="true" damping="5"/>
      <joint name="{side}_roll" type="hinge" pos="{rx} 0 {rz}" axis="0 {s} 0" range="-90 90" limited="true" damping="0.05"/>
      {g("base")}
      <body name="{side}_link1" gravcomp="1">
        {inert.format(rx, (c.flex1_y + c.flex2_y) / 2 * MM, rz)}
        <joint name="{side}_flex1" type="hinge" pos="{rx} {c.flex1_y*MM} {rz}" axis="1 0 0" range="-10 90" limited="true"
               damping="{c.flex1_torque / math.radians(c.flex1_speed):.6g}"/>
        {g("link1")}
        <body name="{side}_link2" gravcomp="1">
          {inert.format(rx, -0.2, rz)}
          <joint name="{side}_flex2" type="hinge" pos="{rx} {c.flex2_y*MM} {rz}" axis="1 0 0" range="-10 90" limited="true" damping="0.05"/>
          {g("link2")}
          {site}
        </body>
      </body>
    </body>"""

    xml = f"""
<mujoco model="tumor_grasp">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="{c.timestep}" integrator="implicitfast" cone="elliptic"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
    <scale contactwidth="0.002" contactheight="0.0005" forcewidth="0.0008" framelength="0.01" framewidth="0.001"/>
    <map force="0.0002"/>
  </visual>
  <default>
    <geom solref="{sr}" solimp="{si}" friction="{c.friction} 0.005 0.0001"/>
    <default class="visual"><geom type="mesh" contype="0" conaffinity="0" group="1" density="0"/></default>
    <default class="col"><geom type="mesh" contype="1" conaffinity="0" group="3" rgba="0.9 0.6 0.2 0.4" density="0"/></default>
    <default class="col_gel"><geom type="mesh" contype="1" conaffinity="4" group="3" rgba="0.9 0.6 0.2 0.4" density="0"/></default>
    <default class="col_pad"><geom type="mesh" contype="5" conaffinity="0" group="3" rgba="0.9 0.6 0.2 0.4" density="0"/></default>
    <default class="frame"><geom type="mesh" contype="1" conaffinity="4" group="3" rgba="1 0.2 0.2 0.5" density="0"/></default>
    <default class="gel"><geom contype="1" conaffinity="4" group="2" density="0"/></default>
  </default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="256" height="256" rgb1="0.93 0.82 0.72" rgb2="0.88 0.76 0.66"/>
    <material name="skin" texture="grid" texrepeat="20 20"/>
    {chr(10).join(assets)}
  </asset>
  <worldbody>
    <light pos="0 0 0.5" dir="0 0 -1" directional="true"/>
    <light pos="0.2 0.2 0.3" dir="-1 -1 -1.5"/>
    <geom name="skin" type="plane" size="0.15 0.15 0.01" material="skin" contype="0" conaffinity="1"/>
    <geom name="tumor" type="ellipsoid" size="0.01 0.01 0.01" pos="0 0 0.005" rgba="0.8 0.25 0.3 1"
          contype="0" conaffinity="1"/>
    <body name="hand" mocap="true" pos="0 0.2 0.05">
      {chr(10).join(link_geoms.get("palm", []))}
      {finger("gel")}
      {finger("pad")}
    </body>
  </worldbody>
  <equality>
    <joint name="eq_gel_slide" joint1="gel_slide" polycoef="0 0 0 0 0" solref="0.002 1"/>
    <joint name="eq_pad_slide" joint1="pad_slide" polycoef="0 0 0 0 0" solref="0.002 1"/>
    <joint name="eq_gel_roll" joint1="gel_roll" polycoef="0 0 0 0 0" solref="0.002 1"/>
    <joint name="eq_pad_roll" joint1="pad_roll" polycoef="0 0 0 0 0" solref="0.002 1"/>
    <joint name="eq_gel_flex2" joint1="gel_flex2" polycoef="0 0 0 0 0" solref="0.002 1"/>
    <joint name="eq_pad_flex2" joint1="pad_flex2" polycoef="0 0 0 0 0" solref="0.002 1"/>
  </equality>
  <actuator>
    <position name="gel_flex1" joint="gel_flex1" kp="5" forcerange="-{c.flex1_torque} {c.flex1_torque}"/>
    <position name="pad_flex1" joint="pad_flex1" kp="5" forcerange="-{c.flex1_torque} {c.flex1_torque}"/>
  </actuator>
</mujoco>
"""
    return xml


def build_model(cfg: HandConfig = DEFAULT_HAND, visual_mesh: bool = True) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(build_xml(cfg, visual_mesh))
