# MODA 八波段旋转检测

唯一主配方为 [FressDet 论文设置](fressdet_alignment.md)：20 epochs、宽 1216 × 高 928、
全局 batch8、从头训练、无随机增强、ProbIoU AP。模型入口为
[O²](../../../configs/experiments/moda/dfine_obb_o2_fressdet.yml) 与
[direct-angle](../../../configs/experiments/moda/dfine_obb_angle_fressdet.yml)。

当前主方案已冻结为 [局部背景条件化的光谱判别](../../ideas/background_conditioned_spectral_design.md)：
保留 O² 基线，让符合对象且能区别于周围背景的光谱证据参与实例聚合。
对齐仅作为控制因素。事实、查新与本轮位置对照统一维护在
[研究问题与证据](research_status.md)；当前尚无检测训练收益结论。

## 数据

目录为 `data/MODA/{train,test}/{images,labels}`。原始 uint8 `.npy` 是
`[8,W,H]`，本地为 `[8,1200,900]`；[适配器](../../../engine/data/dataset/moda_dataset.py)
交换空间轴成为 `[8,H,W]`，保持八波段原始顺序，以 float/255 输入。
当前不把通道索引解释为已知中心波长。自动图集保留八波段原图，机制面板注明以 B4 为底图。

类别顺序为 `car, van, truck, bus, tricycle, bike, awning-bike, pedestrian`。
源 difficulty=0/1/2 均作为有效目标，另存 `source_difficulty`，不修改源标注。
这与 MODA loader 的默认 difficulty 阈值 100 对应。输入补边同步生成 `valid_mask`；
矩形 OBB 的中心和边长统一以画布最长边归一化。

正式入口读取全部 train 9,156 图、test 4,885 图，split_file 为空，每轮按 AP50
选择模型。训练标签含 239,859 个目标，测试标签含 90,323 个目标。2026-09-13 已核实
本地训练图与标签各 9,156 份、测试图与标签各 4,885 份，一一对应且无缺失；全部
数组文件头与标注检查通过。缺图检查仍保留，不静默缩成可用前缀。
最新[数据审计](../../../logs/research/moda/data_ready_2026_09_13/audit.json)覆盖全部
文件头与标注；其中像素统计来自固定种子抽取的 64 张训练图，并非全量像素检查。
数据和显式清单均记录 SHA256。

2026-09-13 的[训练前光谱研究](../../../logs/research/moda/mechanism_probe_2026_09_13/RESULTS.md)
进一步使用 256 张训练图分析谱形、共享读出与背景参考偏差；这些诊断结果不属于检测 AP。
后续的[空间与八波段分解](../../../logs/research/moda/spatial_evidence_2026_09_13/STUDY.md)
展示实际像素上的算子作用，并核查候选背景集合与简单均值对照。
随后进行的[实例结构与跨波段对应检查](../../../logs/research/moda/instance_structure_2026_09_13/STUDY.md)
保留难辨认对象，比较均值、分布和空间表示；另以原始八波段图与 128 张随机训练图核查位置差异。
当前简单候选尚无稳定整体优势，原 bike 诊断例已撤出“真实目标像素被误删”的主证据。
后续[群体探针记录](../../../logs/research/moda/population_evidence_2026_09_13/STUDY.md)
另抽 384 张训练图、4,795 个对象，比较背景相关性、局部与全局度量及波段数量。
4,712 个可测对象有平均数值收益，但逐对象 GT 拟合、有限背景支持和跨波段对应
限制了机制解释。冻结规则、换用新的背景像素后增益仍保留，尚未形成检测 AP 结论。
旧记录的方案推荐已降为候选，后续结论以研究问题与证据文档为准。

## 运行与检查

基线、创新、消融、训练、续训、评估和双 3090 预检查命令统一维护在 [实验运行指南](RUN_EXPERIMENTS.md)。当前方法入口为 `tools/experiments/moda.py`，各组共用 20 轮配方。

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

`RotatedDFINETransformer.query_adapter` 默认 None，不增加基线参数树键。
启用时实现 `build_context(images, targets, diagnostics=bool)`、
`forward(queries, detached_current_boxes, context, layer_index)` 和 `diagnostics(context)`。
当前完整方法在第二层 decoder 更新一次查询，分类、定位及后续层共享更新后的实例表示。
上下文只在单次 forward 内存在；encoder P3 对象分数提供背景排除权重，推理不读取 GT。

模块通过配置 `imports: [engine.rtv4.spectral_evidence]` 注册。零残差检查覆盖 direct-angle/O²
输出、loss、梯度精确等价及 strict loading；源像素诊断使用 `spectral-evidence-v1`。
