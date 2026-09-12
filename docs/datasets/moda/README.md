# MODA 八波段旋转检测

唯一主配方为 [FressDet 论文设置](fressdet_alignment.md)：20 epochs、宽 1216 × 高 928、
全局 batch8、从头训练、无随机增强、ProbIoU AP。模型入口为
[O²](../../../configs/experiments/moda/dfine_obb_o2_fressdet.yml) 与
[direct-angle](../../../configs/experiments/moda/dfine_obb_angle_fressdet.yml)。
BCSR 的配置和科学命题见 [方法文档](../../ideas/bcsr/README.md)。

## 数据

目录为 `data/MODA/{train,test}/{images,labels}`。原始 uint8 `.npy` 是
`[8,W,H]`，本地为 `[8,1200,900]`；[适配器](../../../engine/data/dataset/moda_dataset.py)
交换空间轴成为 `[8,H,W]`，保持八波段原始顺序，以 float/255 输入。
当前不把通道索引解释为已知中心波长。可视化使用注明的第 0 波段灰度图。

类别顺序为 `car, van, truck, bus, tricycle, bike, awning-bike, pedestrian`。
源 difficulty=0/1/2 均作为有效目标，另存 `source_difficulty`，不修改源标注。
这与 MODA loader 的默认 difficulty 阈值 100 对应。输入补边同步生成 `valid_mask`；
矩形 OBB 的中心和边长统一以画布最长边归一化。

正式入口读取全部 train 9,156 图、test 4,885 图，split_file 为空，每轮按 AP50
选择模型。训练标签含 239,859 个目标，测试标签含 90,323 个目标。首次接入时本地
训练图仅 1,000 张、标签完整；缺图会报错，不静默缩成可用前缀。
数据和显式清单均记录 SHA256。

## 运行与检查

完整训练、测试、双 3090 显存预检查命令统一维护在 [运行说明](fressdet_alignment.md)。
主配置及 BCSR 三个消融的实际解析结果与清理前相同；旧训练入口和重复的测试入口已删除。
历史日志保留其当时的 runtime 配置，仅用于追溯既有代码验证，不再作为运行入口。

小样本检查使用显式 debug 清单；本地已有 16 图清单，无需重复生成：

```bash
conda activate wyq-deim
python tools/dataset/prepare_moda_splits.py --output logs/moda/debug_split --debug-images 16
python tools/dataset/moda_preflight.py --debug-split logs/moda/debug_split/debug.json --height 64 --width 96 --queries 20
```

清单目录存在时工具拒绝覆盖；新清单使用新目录。预检查覆盖真实标注、loss/DN、
反向更新和推理，不提供正式 AP 或目标机器耗时估计。分组清单工具仍支持显式
`--group-map`，供后续有明确场景映射的分析使用，主配方不依赖 train/dev 清单。

## 通用扩展接口

`RotatedDFINETransformer.geometry_adapter` 默认 None，不增加旧参数树键。
启用时实现 `build_context(images, targets, diagnostics=bool)`、
`forward(queries, detached_current_boxes, context, layer_index)` 和
`diagnostics(context)`。当前返回 [B,Q,D] 残差只直接进入几何头；几何变化会通过
LQE 和下一层引用框间接影响分类。上下文限定在单次 forward 内，推理不读取 GT。

隔离模块通过配置 `imports: [engine.…]` 注册；稳定模块不导入任何 idea。
零残差测试覆盖 direct-angle/O² 的输出、loss、梯度精确等价及 strict loading。
