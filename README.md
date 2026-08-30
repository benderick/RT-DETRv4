# RT-DETRv4-OBB：可研究、可诊断的旋转目标检测框架

本仓库在 RT-DETRv4/D-FINE 上建立统一的 oriented bounding box（OBB）训练、评估、
推理和诊断底座。当前稳定方法只有两条：**直接角度回归（direct-angle）**与
**O²-DFINE 复现（O² ADR）**。新想法必须在独立 Git worktree 中验证，未通过的
实验不会残留在稳定模型路径里。

项目当前面向 DOTA 四点标注规范，已接入 CODrone 与 UAV-ROD；支持原图训练/评估、
DOTA-style 切片训练、切片预测恢复到原图后合并评估，以及论文分析所需的逐层几何、
匹配质量、显存和耗时诊断。

## 稳定能力

- 统一 OBB 表示 `(cx, cy, w, h, θ)`、四点多边形转换、旋转增强与坐标恢复；
- rotated IoU、DOTA-07 AP、COCO-style `mAP@[.50:.95]` 和通用 DOTA OBB evaluator；
- direct-angle 与 O² ADR 两种 D-FINE 旋转框精炼方式；
- full-image DETR 集合预测，以及 overlapping tiles 恢复原图后的 rotated NMS 合并；
- 训练、评估、推理、可视化与结构化诊断日志；
- 稳定路径回归测试，以及隔离研究 idea 的 worktree/ledger 管理工具。

完整设计约定从[项目文档地图](docs/README.md)进入。任何新数据集、模型、损失、
求值器或诊断模块都应先阅读[集成契约](docs/framework/INTEGRATION_CONTRACT.md)。

## 目录

```text
configs/
  dataset/                 数据协议
  dfine/                   CODrone 稳定 OBB 配置
  experiments/uav_rod/     UAV-ROD 稳定配置
engine/
  data/                    数据集与旋转增强
  evaluation/obb/          通用 DOTA OBB 评估和 tile 合并
  rtv4/obb/                公共几何、模型与稳定方法
docs/                      随代码提交的规范、审计和使用文档
test/                      全部自动化测试
tools/                     数据转换、推理、分析和研究管理工具
logs/                      本机训练/评估产物（Git 忽略）
research/                  本机研究台账和论文资料（Git 忽略）
```

`docs/` 只放能随当前代码复现的稳定事实。活动 idea 的命题卡位于其分支内的
`docs/ideas/<idea_id>/`；退役经验和外部论文资料分别保存在本机
`research/ledger/` 与 `research/references/`。`logs/` 只保存运行产物。

## 环境与数据

本机验收统一使用：

```text
/icislab/volume1/liuxiaolong/anaconda3/envs/wyq-deim/bin/python
```

新环境需根据 CUDA/PyTorch 版本安装依赖：

```bash
pip install -r requirements.txt
```

默认数据路径与仓库平级：

```text
../data/CODrone/
../data/UAV-ROD/
```

CODrone 的 `train/val/test` 是正式划分，`train_t/val_t/test_t` 仅用于小数据流程
检查；`crossview_patches` 和 `crossview_splits` 不在本项目范围内。详细的数据布局、
切片协议与转换方法见 [CODrone 文档](docs/datasets/codrone/README.md)和
[UAV-ROD 文档](docs/datasets/uav_rod/README.md)。HGNetv2 预训练权重默认位于
`pretrain/hgnetv2/`，缺失时底层实现会尝试下载。

## 稳定配置

| 数据/协议 | direct-angle | O² ADR |
|---|---|---|
| CODrone 原图 | `configs/dfine/dfine_obb_angle.yml` | `configs/dfine/dfine_obb_o2.yml` |
| CODrone 标准切片 | `configs/dfine/dfine_obb_angle_tile.yml` | `configs/dfine/dfine_obb_o2_tile.yml` |
| UAV-ROD 原图 | `configs/experiments/uav_rod/dfine_obb_angle.yml` | `configs/experiments/uav_rod/dfine_obb_o2.yml` |

`*_tile.yml` 只更换数据与合并评估协议，不是第三种模型。原图 O² 推理保持 DETR
集合预测语义，不做 rotated NMS；只有多个重叠 tile 的预测被恢复到同一原图时，
才用 rotated NMS 去除跨 tile 重复框。

## 主要命令

以下命令均在仓库根目录运行。

### 训练

CODrone O²：

```bash
CUDA_VISIBLE_DEVICES=0 \
/icislab/volume1/liuxiaolong/anaconda3/envs/wyq-deim/bin/python train.py \
  -c configs/dfine/dfine_obb_o2.yml
```

UAV-ROD O²：

```bash
CUDA_VISIBLE_DEVICES=0 \
/icislab/volume1/liuxiaolong/anaconda3/envs/wyq-deim/bin/python train.py \
  -c configs/experiments/uav_rod/dfine_obb_o2.yml
```

训练 direct-angle 时只需换成表中的对应配置。AMP、训练轮数、batch size 和输出目录
均由 YAML 管理；不要添加已经废弃的 `--use-amp` 参数。

### 续训与冻结 checkpoint 评估

```bash
# 从同一实验配置续训
CUDA_VISIBLE_DEVICES=0 \
/icislab/volume1/liuxiaolong/anaconda3/envs/wyq-deim/bin/python train.py \
  -c configs/dfine/dfine_obb_o2.yml \
  -r logs/dfine_obb_o2/last.pth

# 只评估，不训练
CUDA_VISIBLE_DEVICES=0 \
/icislab/volume1/liuxiaolong/anaconda3/envs/wyq-deim/bin/python train.py \
  -c configs/dfine/dfine_obb_o2.yml --test-only \
  -r logs/dfine_obb_o2/best_stg1.pth
```

checkpoint 名称以实际实验目录为准。正式对比必须记录配置、Git 提交、checkpoint
哈希和数据 provenance，不能只凭目录名判断实验身份。

### 原图推理与可视化

```bash
CUDA_VISIBLE_DEVICES=0 \
/icislab/volume1/liuxiaolong/anaconda3/envs/wyq-deim/bin/python \
  tools/inference/obb_infer.py \
  --config configs/dfine/dfine_obb_o2.yml \
  --checkpoint logs/dfine_obb_o2/best_stg1.pth \
  --input path/to/image_or_directory \
  --output logs/inference/dfine_obb_o2 \
  --device cuda
```

输出包含叠框图和逐图 DOTA 文本。

### 切片推理、恢复与合并

先按 [CODrone 切片协议](docs/datasets/codrone/tiling.md)物化标准切片，再运行：

```bash
CUDA_VISIBLE_DEVICES=0 \
/icislab/volume1/liuxiaolong/anaconda3/envs/wyq-deim/bin/python \
  tools/inference/obb_tile_infer.py \
  --config configs/dfine/dfine_obb_o2_tile.yml \
  --checkpoint logs/dfine_obb_o2_tile/best_stg1.pth \
  --tile-root ../data/CODrone/standard_patches/test \
  --output logs/inference/dfine_obb_o2_tile \
  --device cuda --batch-size 4 --workers 0
```

输出目录必须不存在；确实要覆盖时显式添加 `--overwrite`。内存受限时优先使用
`--workers 0` 并降低 batch size。

### UAV-ROD 标注转换

```bash
/icislab/volume1/liuxiaolong/anaconda3/envs/wyq-deim/bin/python \
  tools/dataset/convert_uav_rod_to_dota.py \
  --dataset-root ../data/UAV-ROD
```

已有转换可用 `--validate-only` 只读复核。

### 测试

```bash
/icislab/volume1/liuxiaolong/anaconda3/envs/wyq-deim/bin/python test/run_all.py
```

测试脚本统一位于 `test/`。CPU 环境会跳过明确要求 CUDA 的 smoke test；正式模型
改动还应在 CUDA 上完成 forward/backward/inference 验收。当前稳定验收状态见
[OBB 底座报告](docs/testing/obb_baseline_report.md)。

## 并行研究 idea

`main` 只维护稳定框架。每个新命题使用独立的 `idea/<idea_id>` 分支和同名 worktree，
候选、Stage 0、prototype、晋级和清退均遵守
[idea 生命周期](docs/development/IDEA_LIFECYCLE.md)与
[worktree 契约/入门手册](docs/development/WORKTREE_CONTRACT.md)。常用管理命令：

```bash
# 查看共享的本地研究台账
/icislab/volume1/liuxiaolong/anaconda3/envs/wyq-deim/bin/python \
  tools/research/manage_ideas.py ledger-path

# 验证台账与活动 idea 边界
/icislab/volume1/liuxiaolong/anaconda3/envs/wyq-deim/bin/python \
  tools/research/manage_ideas.py validate

# 查看已登记 idea
/icislab/volume1/liuxiaolong/anaconda3/envs/wyq-deim/bin/python \
  tools/research/manage_ideas.py list
```

`research/` 被 Git 忽略，因此不会为记录失败经验制造新的 `main` 提交；它也不会随
clone、pull 或 push 迁移，必须单独备份。训练日志仍只进入 `logs/`。

## 文档入口

- [文档总览](docs/README.md)
- [OBB 坐标与几何约定](docs/framework/obb_conventions.md)
- [框架集成契约](docs/framework/INTEGRATION_CONTRACT.md)
- [O²-DFINE 复现、实现审计与诊断协议](docs/methods/o2/README.md)
- [CODrone 数据与切片协议](docs/datasets/codrone/README.md)
- [UAV-ROD 数据与转换协议](docs/datasets/uav_rod/README.md)
- [研究 worktree 使用方法](docs/development/WORKTREE_CONTRACT.md)

## 上游项目与引用

本仓库不是 RT-DETRv4 官方发布版；它在其代码基础上扩展 OBB 能力，并参考了
[RT-DETR](https://github.com/lyuwenyu/RT-DETR)、
[D-FINE](https://github.com/Peterande/D-FINE) 和
[DEIM](https://github.com/Intellindust-AI-Lab/DEIM)。O² 稳定路径复现论文
*Real-Time Oriented Object Detection Transformer in Remote Sensing Images*；具体复现
边界和与原文的对应关系见[实现审计](docs/methods/o2/implementation_audit.md)。

使用本仓库时请同时遵守根目录 [LICENSE](LICENSE) 及相关上游项目许可。RT-DETRv4
引用信息：

```bibtex
@article{liao2025rtdetrv4,
  title={RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models},
  author={Zijun Liao and Yian Zhao and Xin Shan and Yu Yan and Chang Liu and Lei Lu and Xiangyang Ji and Jie Chen},
  journal={arXiv preprint arXiv:2510.25257},
  year={2025}
}
```
