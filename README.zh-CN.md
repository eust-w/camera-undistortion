# Auto Camera Geometry

<p align="center">
  <b>面向 Ego 双目鱼眼设备的 Pixel ↔ Ray ↔ 3D 相机几何基础设施</b>
</p>

<p align="center">
  <a href="https://github.com/eust-w/camera-undistortion/actions"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/eust-w/camera-undistortion/ci.yml?branch=main&label=CI&labelColor=333b45&color=2ea44f"></a>
  <a href="#在真实数据上的验证tum-vi"><img alt="Tests" src="https://img.shields.io/badge/%E6%B5%8B%E8%AF%95-79%20%E9%80%9A%E8%BF%87-2ea44f?labelColor=333b45"></a>
  <a href="https://github.com/eust-w/camera-undistortion/actions/workflows/ci.yml"><img alt="mypy" src="https://img.shields.io/badge/mypy-%E9%9B%B6%E5%91%8A%E8%AD%A6-2ea44f?labelColor=333b45"></a>
  <a href="https://github.com/eust-w/camera-undistortion"><img alt="models" src="https://img.shields.io/badge/%E6%A8%A1%E5%9E%8B-KB4%C2%B7DS%C2%B7EUCM%C2%B7BC-blue?labelColor=333b45"></a>
  <img alt="python" src="https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776ab?labelColor=333b45">
  <a href="LICENSE"><img alt="Code license" src="https://img.shields.io/badge/license-Apache--2.0-orange?labelColor=333b45"></a>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> |
  <a href="#下游适配器示例">示例</a> |
  <a href="#真实数据验证tum-vi">TUM-VI 验证</a> |
  <a href="#service-api">Service API</a> |
  <a href="scripts/benchmark.py">性能基准</a>
</p>

<p align="center">
  <a href="README.md">English</a> |
  简体中文
</p>

---

这是面向双目鱼眼 Ego 设备（Orbbec 式 IR 立体）的端到端相机几何基础设施：出厂标定被自动解析、验证并给出健康分；ray map、虚拟针孔视图、立体校正对与深度全部自动派生并缓存——**全程零人工**：不手动选模型、不手动调参，标定 QA 失败时自动走 fallback 链（双目精化 → AnyCalib → 隔离）。

**它不是“去畸变”工具**：去畸变只是规范抽象 `pixel ↔ ray ↔ 3D` 的一个输出。

## 演示 —— TUM-VI 真实帧过全流程

原始鱼眼双目（TUM-VI，官方 DoubleSphere 标定）→ 自动 QA → 虚拟针孔视图 + 立体校正对。以下输出全部由管线自动产生：

![demo](docs/demo_tumvi.png)

## 为什么

VLM/VLA 需要标准针孔图；SLAM / 3DGS / 深度需要全 FOV 原生射线；立体深度需要校正对。本项目用单一规范表示同时服务三者：

```
原始鱼眼（永不修改）
      │
      ├─→ 原生射线        pixel_to_ray() → SLAM / 3DGS / 手部·物体 3D
      ├─→ 虚拟针孔         90°/100°/110° 预设 → VLM / VLA / 2D 检测器
      └─→ 立体校正         水平极线，Z = f·B/d → 深度
```

## 能力边界

集成前请先读这一节。以下规则要么是数学极限、要么经过实验验证——不是“还没做的功能”。

| | 能力 | 边界 |
|---|---|---|
| ✅ | 双目鱼眼 + 出厂标定（KB4/DS） | 全流程：QA → 射线 → 虚拟视图 → 立体校正 → 深度 |
| ✅ | 单目视频/图片去畸变 | 必须有该相机的出厂标定（视频文件本身不含内参） |
| ✅ | 虚拟针孔视图 | 上限 **110°**——直线透视超过 ~120° 在数学上不可行（边缘拉伸无穷大） |
| ✅ | 自动 QA + fallback 链 | 垂直通道参数（fy/cy/k1）可通过固定-E 视差恢复 |
| ⚠️ | `fx`/`cx` 精化 | 水平基线视差约束下**不可观**——锚定出厂值（需 plumb-line/多视图等额外观测） |
| ⚠️ | 双相机共模偏差 | 仅靠双目视差无法与“单相机误差”区分——按约定冻结参考相机，可能残留 |
| ❌ | 完全无标定的单目自标定 | 实验证明理论不可行（自由深度+自由 E 吸收一切）；必须接入 `DenseRayProvider`（如 AnyCalib 网络） |
| ❌ | 鱼眼原生 FOV（>150°）→ 直线透视 | 投影数学上不可行；要保留全 FOV 需立体投影（stereographic，暂未纳入） |
| ⚠️ | fallback 精度地板 | 合成低纹理数据收敛到 ~1.5px 而非 0.5px PASS 门 → 按设计自动 QUARANTINE；真实丰富纹理预期更好 |
| ⚠️ | Rolling shutter | 模型与 row-aware API 就绪，但需要你 VIO 栈的 IMU/读出时间元数据 |
| ❌ | 特征前端 / SLAM 后端 / 3DGS 渲染器本体 | 本项目只提供几何层；examples 展示接入契约 |

性能参考（M 系列 Mac）：unproject 约 800 万点/秒，ray map 1600×1300 约 0.7s，remap 约 390fps（CPU），12 帧双目 episode 冷启动约 12s。质量门：79 个测试、golden 回归 1e-6°、mypy 零告警。

## 快速开始

```bash
git clone https://github.com/eust-w/camera-undistortion.git
cd camera-undistortion
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
pytest          # 79 个测试：单元/属性/集成/golden/示例
```

用已知出厂标定去畸变一段视频：

```bash
python scripts/undistort_video.py input.mp4 --preset ego_100 --size 1024
```

对双目图像目录跑全自动 episode DAG：

```bash
python -m camera_geometry process-dataset --input episode/ --output episode/ --config device.yaml
camera-geometry validate-calibration --config device.yaml     # QA + 健康分
camera-geometry inspect-camera       --config device.yaml     # 解析后的规范视图
```

Python API：

```python
from camera_geometry.calibration.parser import load_device_calibration
from camera_geometry.models.factory import build_camera

cam = build_camera(load_device_calibration("device.yaml").camera("IR_L"))
rays = cam.pixel_to_ray(pixels)   # [...,2] -> [...,3] 单位射线（全 FOV）
px   = cam.ray_to_pixel(rays)     # 反投影
mask = cam.valid_mask()           # 数值有效区域
```

## 真实数据验证（TUM-VI）

在公开的 [TUM VI Benchmark](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset)（全局快门鱼眼双目）上，使用 basalt 官方 DoubleSphere 标定做了端到端验证：

| 检查 | 结果 |
|---|---|
| DS 模型投影往返 | 机器精度 |
| 自动 QA + 健康分 | PASS（schema / rotation / baseline / round-trip） |
| Fallback 链 | 注入损坏自动检测并恢复 |
| 选择性数据下载 | 1.4 GB 归档仅传输 ~16 MB（Range 流式 tar 抽取） |

复现方式见英文版 [Validation 一节](README.md#validation-on-real-data-tum-vi)。

## 相机模型

| 模型 | 状态 | 说明 |
|---|---|---|
| Kannala-Brandt 4 | ✅ | theta-LUT Newton 反投影，约 800 万点/秒，往返机器精度 |
| Double Sphere | ✅ | 闭式反投影，支持负 xi（TUM-VI 官方标定） |
| EUCM | ✅ | alpha ∈ [0,1)，α=0 即针孔 |
| Brown-Conrady | ✅ | plumb_bob，普通镜头 |
| Pinhole | ✅ | 虚拟视图 / 校正相机 |

## 自动标定管线

```
出厂标定
   │ QA：schema · RᵀR≈I · 基线范围 · 往返 P99 · 双目视差
 PASS ─────────► 缓存 LUT ► 射线/视图/立体/深度
   │ FAIL
   ▼
阶段 1  双目精化（固定出厂本质矩阵 + 带符号 sampson +
        鲁棒坐标下降 + LM；只优化可辨识子集；参考相机冻结）
   ▼ FAIL
阶段 1.5 外部 COLMAP SfM 钩子（可选二进制）
   ▼ FAIL
阶段 2  AnyCalib 式拟合：稠密射线提供器 → 逐帧鲁棒拟合、
        中位数共识、模型选择（KB4/EUCM/DS/pinhole）
   ▼ FAIL
QUARANTINE（记录原因与阶段；可重跑，绝不静默放行）
```

已编码进设计的可辨识性结论（均经实验验证）：单目两视带自由深度的重投影无法辨识内参；自由本质矩阵会吸收光滑畸变误差；水平基线的固定-E 视差只约束垂直通道——因此 `fx/cx` 保持锚定、参考相机按约定冻结。

## 下游适配器示例

```bash
python examples/slam_adapter.py     # 原始特征像素 -> 射线 -> 三角化
python examples/render_for_3dgs.py  # 高斯点 -> 原始鱼眼像素 + ray map
python examples/handpose_3d.py      # 虚拟视图检测 -> 3D 关键点
```

## Service API

```bash
uvicorn camera_geometry.service:app --port 8300
# POST /camera/resolve | validate | ray-map | virtual-camera
# POST /camera/stereo-rectify | stereo-rectify-upload | reprocess (+ GET /jobs/{id})
# GET  /camera/{calibration_id} | /metrics (Prometheus) | /healthz
```

所有含标定的响应都携带 `calibration_id`、`camera_model`、`quality_status`、`artifact_uri`。`GET /metrics` 以 Prometheus 格式暴露 spec 第 43 节指标。

## 存储、Docker、CI

* LUT 缓存支持本地文件系统或任意 S3 兼容桶（`CAMERA_GEOMETRY_LUT_BACKEND=s3`）
* 时间戳为 parquet（pyarrow），无依赖时回退 CSV
* `docker build -t camera-geometry . && docker run -p 8300:8300 camera-geometry`
* CI 在 Python 3.11/3.12 上运行 ruff + mypy（48 文件零告警）+ pytest

## Roadmap（需要外部输入）

* Phase-0 Orbbec 厂商 golden（`scripts/export_golden.py` 已留 `orbbec_pixel_to_ray()` 槽位）
* 精化后 0.5 px 双目门限的真机场景确认
* Rolling-shutter 数据闭环（模型 + row-aware API 已就绪，需 IMU/readout 元数据）
* AnyCalib 网络权重（接入既有 `DenseRayProvider` 接口）

## 许可证

Apache-2.0。见 [LICENSE](LICENSE)。
