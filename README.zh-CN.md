<div align="center">
  <img src="docs/readme/hero.svg" alt="Auto Camera Geometry — workflow overview" width="100%">
  <h1>Auto Camera Geometry</h1>
  <p><strong>从标定像素到原生射线，让同一套几何服务多种视觉任务。</strong></p>
  <p><a href="https://github.com/eust-w/camera-undistortion/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/eust-w/camera-undistortion/ci.yml?branch=main&amp;style=flat-square&amp;label=CI"></a>
  <img alt="Python: 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-3776ab?style=flat-square">
  <img alt="Models: KB4 / DS / EUCM / BC" src="https://img.shields.io/badge/Models-KB4%20%2F%20DS%20%2F%20EUCM%20%2F%20BC-0891b2?style=flat-square">
  <img alt="Code license: Apache-2.0" src="https://img.shields.io/badge/Code%20license-Apache--2.0-d99536?style=flat-square"></p>
  <p>
    <a href="#overview">项目亮点</a> ·
    <a href="#quick-start">快速开始</a> ·
    <a href="#workflow">工作流</a> ·
    <a href="#validation">验证与边界</a> ·
    <a href="#documentation">文档</a>
  </p>
  <p><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a></p>
</div>

---

<a id="overview"></a>
## 一份标定，多种下游视图

**Auto Camera Geometry** 将鱼眼与双目相机的工厂标定整理成统一几何接口：解析、检查并缓存标定，再输出原生射线、虚拟针孔视图、双目校正图像和深度，供不同下游应用使用。

| 保留原生几何 | 提供标准视图 | 记录质量状态 |
| :--- | :--- | :--- |
| Pixel ↔ Ray 支持 SLAM、三角化和 3D 渲染 | 虚拟针孔预设面向 VLM/VLA 与 2D 视觉 | 标定检查、分阶段恢复、失败隔离及原因记录 |

### 真实数据参考

![TUM-VI 原始鱼眼图像及衍生相机视图](docs/demo_tumvi.png)

*仓库现有演示：TUM-VI 图像与 Double Sphere 标定。展示的是相机处理流程，不代表下游 SLAM 或策略模型的评测结果。*

<a id="quick-start"></a>
## 快速开始

需要 **Python 3.11+**。先使用仓库中的示例标定，无需下载完整数据集。

```bash
git clone https://github.com/eust-w/camera-undistortion.git
cd camera-undistortion
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

camera-geometry inspect-camera --config configs/device_orbbec_example.yaml
camera-geometry validate-calibration --config configs/device_orbbec_example.yaml
```

示例中的右相机内参是占位值，只用于演示数据结构和命令，不代表你的设备已经完成标定。

处理双目 episode 时，使用属于该设备的标定，并按 [pipeline](src/camera_geometry/pipeline.py) 所需布局组织输入：

```bash
camera-geometry process-dataset \
  --input /path/to/episode --output /path/to/output \
  --config /path/to/device.yaml
```

### Python 接口

```python
import numpy as np
from camera_geometry.calibration.parser import load_device_calibration
from camera_geometry.models.factory import build_camera

calibration = load_device_calibration("configs/device_orbbec_example.yaml")
camera = build_camera(calibration.camera("IR_L"))
pixels = np.array([[800.0, 650.0]])
rays = camera.pixel_to_ray(pixels)
reprojected = camera.ray_to_pixel(rays)
```

<a id="workflow"></a>
## 从标定到产物

```text
工厂标定 → Schema / 旋转 / 基线 / 投影检查
                         │
                         ├─ 通过 → 缓存几何 → 射线 / 视图 / 双目 / 深度
                         └─ 恢复 → 双目优化 → 可选 COLMAP / 稠密射线拟合
                                               └─ 未解决 → 隔离并记录原因
```

| 相机模型 | 对应实现 |
| :--- | :--- |
| Kannala–Brandt 4 | [KB4](src/camera_geometry/models/kb4.py) |
| Double Sphere | [DS](src/camera_geometry/models/double_sphere.py)，支持负 `xi` |
| EUCM | [扩展统一相机模型](src/camera_geometry/models/eucm.py) |
| Brown–Conrady | [Brown](src/camera_geometry/models/brown.py) |
| Pinhole | [虚拟与校正相机](src/camera_geometry/models/pinhole.py) |

```bash
python examples/slam_adapter.py
python examples/render_for_3dgs.py
python examples/handpose_3d.py
```

这些示例展示几何接口的接法；仓库不包含完整 SLAM 后端、Gaussian 渲染器或手部姿态模型。

<details>
<summary><strong>可选 HTTP 服务</strong></summary>

```bash
python -m pip install -e '.[server]'
uvicorn camera_geometry.service:app --host 127.0.0.1 --port 8300
```

[服务实现](src/camera_geometry/service.py) 提供标定解析、射线图、虚拟相机、双目校正、作业、`/healthz` 和 `/metrics` 接口。

</details>

<a id="validation"></a>
## 验证与使用边界

```bash
python -m pytest
python -m mypy src/camera_geometry
python scripts/benchmark.py --help
```

| 项目 | 当前边界 |
| :--- | :--- |
| 标定输入 | 仅有视频不足以确定相机几何；需要匹配设备的标定。 |
| 虚拟视图 | 内置 90° / 100° / 110° 预设；大视场针孔投影会放大边缘拉伸。 |
| 双目优化 | 水平基线的极线约束不能辨识全部参数；`fx/cx` 保持锚定，参考相机固定。 |
| 稠密射线恢复 | 需要兼容 provider，仓库不附带 AnyCalib 预训练权重。 |
| 滚动快门 | 行级接口需要外部读出时间与运动元数据。 |
| 视频脚本 | `scripts/undistort_video.py` 内置特定工厂参数，不能原样用于其他相机。 |

TUM-VI 的数据准备流程见 [English README](README.md#validation) 与 [导入脚本](scripts/import_tumvi.py)；图像下载后仍需单独提供匹配的 Double Sphere 标定 JSON。

<a id="documentation"></a>
## 仓库导航

| 入口 | 内容 |
| :--- | :--- |
| [示例标定](configs/device_orbbec_example.yaml) | 设备、内参、外参布局 |
| [处理流水线](src/camera_geometry/pipeline.py) | Episode 处理与输出 |
| [恢复阶段](src/camera_geometry/fallback/) | 标定回退与外部接入 |
| [测试](tests/) · [CI](.github/workflows/ci.yml) | 几何、标定、流程与回归检查 |

## 许可证

代码采用 [Apache-2.0](LICENSE)。外部数据集、标定文件与模型资源遵循各自的使用条款。
