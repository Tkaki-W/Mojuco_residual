"""STEP からグループごとの形を取り出して BREP に保存する（読み込みが重いので1回だけ）。

  python scripts/cad_extract.py
出力: cadデータ/brep/<グループ名>.brep（ハンド全体の座標系のまま）
"""
from pathlib import Path

from OCP.BRepTools import BRepTools
from OCP.collections import Sequence_TDF_Label
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDF import TDF_Label
from OCP.TDataStd import TDataStd_Name
from OCP.TDocStd import TDocStd_Document
from OCP.TopLoc import TopLoc_Location
from OCP.XCAFDoc import XCAFDoc_DocumentTool

ROOT = Path(__file__).resolve().parents[1]
STEP = ROOT / "cadデータ" / "hand_new_fingers.step"
OUT = ROOT / "cadデータ" / "brep"
GROUPS = ["hand_body", "GEL_finger_right_seen_from_tip", "PAD_finger_left_seen_from_tip"]


def name_of(label):
    attr = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), attr):
        return attr.Get().ToExtString()
    return ""


def main():
    doc = TDocStd_Document(TCollection_ExtendedString("doc"))
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    print("STEP を読み込み中...", flush=True)
    reader.ReadFile(str(STEP))
    reader.Transfer(doc)
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    OUT.mkdir(exist_ok=True)
    found = {}

    def walk(label, parent_loc, depth):
        loc = parent_loc
        proto = label
        if st.IsReference_s(label):
            loc = parent_loc.Multiplied(st.GetLocation_s(label))
            proto = TDF_Label()
            st.GetReferredShape_s(label, proto)
        nm = name_of(proto) or name_of(label)
        if depth <= 3:
            print("  " * depth + nm, flush=True)
        g = next((g for g in GROUPS if g == nm), None)
        if g and g not in found:
            found[g] = st.GetShape_s(proto).Moved(loc)
            return
        kids = Sequence_TDF_Label()
        st.GetComponents_s(proto, kids)
        for i in range(1, kids.Length() + 1):
            walk(kids.Value(i), loc, depth + 1)

    roots = Sequence_TDF_Label()
    st.GetFreeShapes(roots)
    for i in range(1, roots.Length() + 1):
        walk(roots.Value(i), TopLoc_Location(), 0)

    for g, shp in found.items():
        p = OUT / f"{g}.brep"
        BRepTools.Write_s(shp, str(p))
        print("保存:", p.name)


if __name__ == "__main__":
    main()
