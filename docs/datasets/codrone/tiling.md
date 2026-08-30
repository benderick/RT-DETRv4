# CODrone 1180/200 局部切片协议

tile 是普通独立训练样本，但指标只在预测恢复到源图坐标并完成全局合并后计算；
tile AP 不作为报告结果。direct-angle 与 O² 必须共享同一份物化数据和合并参数。

## 冻结数据协议

- 源划分：`train`、`val`、`test`；tiny 流程检查使用对应 `_t` 划分；
- 窗口：`1180 x 1180`，DOTA `gap=200`，名义步长 980；末窗吸附边界；
- 有效图像比例：严格大于 0.6；无候选通过时使用官方 splitter fallback；
- 目标保留：polygon/window IOF（除以源目标面积）不小于 0.7；
- 截断目标：平移原 polygon、不裁剪，写 DOTA difficulty 2；
- padding：OpenCV BGR `(104,116,124)`，RGB 解码值为 `(124,116,104)`；
- 默认存储：JPEG quality 95；PNG 只用于显式无损像素审计；
- 网络画布：保持比例 resize/pad 到 `1024 x 1024`。

一张 `3840 x 2160` 图像的 x 起点为 `[0,980,1960,2660]`，y 起点为
`[0,980]`，共 8 个 tile；边界吸附令最后一次横向重叠为 480。

## 数据物化

先在 tiny split 验证：

```bash
/icislab/volume1/liuxiaolong/anaconda3/envs/wyq-deim/bin/python \
  tools/dataset/split_codrone_obb.py \
  --source-root ../data/CODrone \
  --output-root ../data/CODrone/standard_patches_t \
  --splits train_t val_t test_t --nproc 4 --preview-samples 4
```

正式数据：

```bash
/icislab/volume1/liuxiaolong/anaconda3/envs/wyq-deim/bin/python \
  tools/dataset/split_codrone_obb.py \
  --source-root ../data/CODrone \
  --output-root ../data/CODrone/standard_patches \
  --splits train val test --nproc 10 --preview-samples 12
```

已有输出不会被静默复用或删除；重新生成必须显式传 `--overwrite`。每个 split
包含 `images/`、`annfile/`、`metadata/`、`manifest.json`、`_SUCCESS`，以及
`diagnostics/{images,tiles,objects,invalid_annotations}.jsonl.gz` 和审计可视化。
manifest 固定协议、命令、Git 状态和源 inventory SHA-256。

## 训练、评估和推理

```bash
CUDA_VISIBLE_DEVICES=0 \
/icislab/volume1/liuxiaolong/anaconda3/envs/wyq-deim/bin/python train.py \
  -c configs/dfine/dfine_obb_o2_tile.yml

CUDA_VISIBLE_DEVICES=0 \
/icislab/volume1/liuxiaolong/anaconda3/envs/wyq-deim/bin/python train.py \
  -c configs/dfine/dfine_obb_o2_tile.yml \
  -r logs/dfine_obb_o2_tile/best_stg1.pth --test-only

CUDA_VISIBLE_DEVICES=0 \
/icislab/volume1/liuxiaolong/anaconda3/envs/wyq-deim/bin/python \
  tools/inference/obb_tile_infer.py \
  --config configs/dfine/dfine_obb_o2_tile.yml \
  --checkpoint logs/dfine_obb_o2_tile/best_stg1.pth \
  --tile-root ../data/CODrone/standard_patches/test \
  --output ./tile_test_predictions
```

direct-angle 对照将配置和日志目录中的 `o2` 换成 `angle`。两条路径均为 72 epochs、
同 HGNetv2-B2 容量、无 Mosaic/MixUp/batch multi-scale。选择 epoch 或超参数不得
查看测试集指标。

每个 tile 先做类别感知 rotated NMS，再加 origin 回源图；所有 rank 收齐完整 tile
后，在源图执行第二次类别感知 rotated NMS，最后用 `MergedDotaOBBEvaluator` 只求
一次原图 AP。缺 tile 直接失败。

## 机制与回归证据

日志记录局部/全局 OBB、边界距离、精确 NMS suppressor、source-object 身份和
抑制类型。分析入口：

```bash
/icislab/volume1/liuxiaolong/anaconda3/envs/wyq-deim/bin/python \
  tools/analysis/summarize_obb_diagnostics.py \
  logs/dfine_obb_o2_tile --output ./tile_analysis

/icislab/volume1/liuxiaolong/anaconda3/envs/wyq-deim/bin/python \
  tools/analysis/visualize_tile_merge_mechanisms.py \
  logs/dfine_obb_o2_tile --mode collision --count 20 --output ./tile_collisions
```

自动测试会从真实 `val_t` annotation 构造理想 tile predictions，并验证 IOF、坐标
平移、全局 NMS 和 evaluator ceiling。这是每次测试重新计算的工程回归，不是旧
训练实验结果；任何模型 AP、收敛、速度或显存结论必须来自清理后的新 run。
