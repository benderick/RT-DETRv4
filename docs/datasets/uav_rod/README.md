# UAV-ROD 数据适配

UAV-ROD 是低空无人机场景的单类旋转车辆数据集。当前本地发布包包含官方
`train/test` 两个划分：训练集 1,150 张，测试集 427 张，总计 1,577 张图像和
30,090 个 `car` 实例。项目不重新随机拆分；带标注的官方 `test` 在配置中作为
验证与最终评估集。

## 原始标注与转换

原始标注是扩展 VOC XML：

```text
robndbox = (cx, cy, w, h, angle)
```

原作者的可视化代码直接调用
[`cv2.boxPoints(((cx, cy), (w, h), angle * 180 / π))`](https://github.com/fengkaibit/UAV-ROD/blob/master/show_groundtruth.py)：
`angle` 为弧度，在图像坐标中顺时针增大，并将其中一条 `h` 边标成有向车头。
转换器与该 OpenCV 公式逐点数值对齐，生成连续四点的 DOTA 标注，并保留 VOC
`difficult` 字段：

```text
x1 y1 x2 y2 x3 y3 x4 y4 car difficult
```

UAV-ROD 还可用于车辆头向识别，其原始角度具有 `2π` 有向语义；本项目当前做的是
普通 OBB 检测，框方向按 `π` 周期等价，因此 DOTA 转换不声称保留车头朝向。

在仓库根目录运行：

```bash
/icislab/volume1/liuxiaolong/anaconda3/envs/wyq-deim/bin/python \
  tools/dataset/convert_uav_rod_to_dota.py \
  --dataset-root ../data/UAV-ROD
```

转换在每个 split 下生成 `annfile/` 和 `conversion_manifest.json`。转换器默认拒绝
覆盖已有目录。以后可用只读模式逐文件重算并验证：

```bash
/icislab/volume1/liuxiaolong/anaconda3/envs/wyq-deim/bin/python \
  tools/dataset/convert_uav_rod_to_dota.py \
  --dataset-root ../data/UAV-ROD --validate-only
```

manifest 记录图像/目标数量、尺寸分布、越界目标、类别、源清单 hash 和转换结果
hash；训练诊断会把这些 provenance 一同写入日志。

本地发布包的实测尺寸并不全是论文概述中的 `1920×1080`：1,284 张为
`1920×1080`，293 张为 `2720×1530`。这由转换 manifest 原样记录，配置不会假设
所有原图等宽等高。

## 项目配置

数据协议：

```text
configs/dataset/uav_rod_obb.yml
```

两个稳定模型在 UAV-ROD 上的可运行绑定：

```text
configs/experiments/uav_rod/dfine_obb_angle.yml
configs/experiments/uav_rod/dfine_obb_o2.yml
```

训练命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
/icislab/volume1/liuxiaolong/anaconda3/envs/wyq-deim/bin/python train.py \
  -c configs/experiments/uav_rod/dfine_obb_o2.yml
```

direct-angle 只需替换为对应配置。两者都将完整图保持比例缩放并补齐到
`1024×1024`，不切片、不使用 Mosaic/MixUp。O² full-image 推理继续保持 DETR
集合预测语义，不做 rotated NMS。

原论文将 DOTA 切片到 `1024×1024`，而 UAV-ROD 等非 DOTA 数据缩放到
`1333×800`。本项目的 `1024×1024` 是为了在同一 OBB 框架内严格比较 direct-angle
与 O²；与 UAV-ROD 原论文表格比较时必须明确这一预处理差异。

## 评估

`UAVRODDetection` 只负责类别、路径、DOTA 文本解析和视频/帧 provenance。指标
统一复用 `DotaOBBEvaluator`，同时输出 DOTA-07 AP50/AP75 以及 diagnostic
COCO-style `mAP@[.50:.95]`，不创建数据集专用 evaluator。
