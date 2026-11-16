"""M0 確認: MuJoCo が動き、表示できるかを確かめる。

使い方:
  python scripts/m0_check.py          # 画像に保存 (results/m0_check.png)
  python scripts/m0_check.py --view   # ビューアで表示（ウィンドウを閉じると終了）
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <light pos="0 0 1"/>
    <geom name="skin" type="plane" size="0.2 0.2 0.01" rgba="0.9 0.8 0.7 1"/>
    <body pos="0 0 0.1">
      <freejoint/>
      <geom name="tumor" type="sphere" size="0.01" rgba="0.8 0.2 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def main():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    for _ in range(500):  # 1秒落下させる
        mujoco.mj_step(model, data)
    print(f"mujoco {mujoco.__version__}: 球の高さ {data.qpos[2]*1000:.1f} mm（床の上で約10 mm なら正常）")

    if "--view" in sys.argv:
        from mujoco import viewer
        viewer.launch(model, data)
        return

    out = Path(__file__).resolve().parents[1] / "results" / "m0_check.png"
    out.parent.mkdir(exist_ok=True)
    with mujoco.Renderer(model, 240, 320) as r:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0, 0, 0.01]
        cam.distance, cam.elevation = 0.15, -30
        r.update_scene(data, cam)
        img = r.render()
    import matplotlib.pyplot as plt
    plt.imsave(out, img)
    print(f"画像を保存: {out}（平均輝度 {np.mean(img):.0f}）")


if __name__ == "__main__":
    main()
