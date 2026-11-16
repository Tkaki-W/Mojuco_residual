"""CAD（BREP）から MuJoCo 用のメッシュを作る。cad_extract.py の後に1回だけ実行する。

  python scripts/cad_export.py

出力（座標はすべて CAD の座標系・mm のまま）
  assets/meshes/<group>_<link>.stl  … 見た目用（リンクごとにまとめた三角形）
  assets/hulls.npz                    … 衝突用（部品1つ＝凸包1つ、頂点だけ保存）
  assets/hulls.json                   … 各凸包がどのリンク・どの役割（名前）か

リンクの分け方（指）：部品の中心の y で決める
  y > -111.7（屈曲1より根元） → base、-111.7〜-158.7 → link1、それより先 → link2
"""
import json
import struct
from pathlib import Path

import numpy as np
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Shape

ROOT = Path(__file__).resolve().parents[1]
BREP = ROOT / "cadデータ" / "brep"
OUT = ROOT / "assets"
FLEX1_Y, FLEX2_Y = -111.7, -158.7
GROUPS = {"hand_body": "palm", "GEL_finger_right_seen_from_tip": "gel", "PAD_finger_left_seen_from_tip": "pad"}
SENSOR_BBOX_Y = (-233.3, -202.1)  # ゲル指のセンサ部（ゲル窓と斜面を含む部品）の y 範囲


def load(name):
    shp = TopoDS_Shape()
    BRepTools.Read_s(shp, str(BREP / f"{name}.brep"), BRep_Builder())
    return shp


def bbox(s):
    b = Bnd_Box()
    BRepBndLib.Add_s(s, b, False)
    lo, hi = b.CornerMin(), b.CornerMax()
    return np.array([lo.X(), lo.Y(), lo.Z()]), np.array([hi.X(), hi.Y(), hi.Z()])


def triangles(solid):
    """部品の三角形を (N,3,3) で返す"""
    tris = []
    ex = TopExp_Explorer(solid, TopAbs_FACE)
    while ex.More():
        f = TopoDS.Face(ex.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(f, loc)
        if tri is not None:
            tr = loc.Transformation()
            pts = []
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i).Transformed(tr)
                pts.append((p.X(), p.Y(), p.Z()))
            pts = np.array(pts)
            rev = f.Orientation() == TopAbs_REVERSED
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(i).Get()
                idx = (a - 1, c - 1, b - 1) if rev else (a - 1, b - 1, c - 1)
                tris.append(pts[list(idx)])
        ex.Next()
    return np.array(tris).reshape(-1, 3, 3)


def write_stl(path, tris):
    with open(path, "wb") as fp:
        fp.write(b"\0" * 80 + struct.pack("<I", len(tris)))
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        rec["n"], rec["v"] = n, tris
        fp.write(rec.tobytes())


def link_of(group, center_y):
    if group == "palm":
        return "palm"
    if center_y > FLEX1_Y:
        return "base"
    return "link1" if center_y > FLEX2_Y else "link2"


def main():
    (OUT / "meshes").mkdir(parents=True, exist_ok=True)
    visual, hulls, meta = {}, {}, []
    for gname, group in GROUPS.items():
        shp = load(gname)
        print(f"{gname}: 三角形に分割中...", flush=True)
        BRepMesh_IncrementalMesh(shp, 0.8 if group == "palm" else 0.3, False, 0.5, True)
        ex = TopExp_Explorer(shp, TopAbs_SOLID)
        k = 0
        while ex.More():
            s = ex.Current()
            lo, hi = bbox(s)
            size = hi - lo
            c = (lo + hi) / 2
            link = link_of(group, c[1])
            tris = triangles(s)
            if len(tris):
                visual.setdefault(f"{group}_{link}", []).append(tris)
            # 衝突用：小さい部品（ねじ・電子部品）は省く。手のひらは低い所（z<-20）にある部品だけ
            keep = size.max() >= 4.0 and (group != "palm" or lo[2] < -20)
            if keep and len(tris):
                if group == "gel" and lo[1] > SENSOR_BBOX_Y[0] and hi[1] < SENSOR_BBOX_Y[1] and lo[2] < -29:
                    role = "sensor_frame"
                elif group == "gel":
                    role = f"gelfinger_body_{k}"
                else:
                    role = f"{group}_{k}"
                v = np.unique(np.round(tris.reshape(-1, 3), 2), axis=0)
                hulls[role] = v
                meta.append({"name": role, "group": group, "link": link,
                             "bbox_min": lo.round(2).tolist(), "bbox_max": hi.round(2).tolist()})
            k += 1
            ex.Next()
    for name, parts in visual.items():
        t = np.concatenate(parts)
        for old in (OUT / "meshes").glob(f"{name}*.stl"):
            old.unlink()
        n = 190000  # MuJoCo は1ファイル20万三角形まで
        for i in range(0, len(t), n):
            write_stl(OUT / "meshes" / f"{name}_{i // n}.stl", t[i:i + n])
        print(f"見た目 {name}: 三角形 {len(t)}")
    np.savez_compressed(OUT / "hulls.npz", **hulls)
    (OUT / "hulls.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    roles = [m["name"] for m in meta]
    print(f"衝突用の凸包 {len(roles)} 個（sensor_frame: {roles.count('sensor_frame')}）")
    for g in ("palm", "gel", "pad"):
        print(" ", g, sum(m["group"] == g for m in meta))


if __name__ == "__main__":
    main()
