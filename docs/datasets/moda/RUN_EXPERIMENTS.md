# MODA：基线、创新与消融运行指南

所有命令从仓库根目录执行，使用 `conda activate wyq-deim`。入口为
[`tools/experiments/moda.py`](../../../tools/experiments/moda.py)，不需要安装新包。
按统一配置从头训练，输出仍使用 `logs/moda/experiments/`。实现检查不作为检测收益结论。

## 实验对应关系

| 名称 | 运行参数 | 内容 | 对照目的 |
|---|---|---|---|
| 基线 B0 | `baseline` | 八波段 O²，原 HGNetv2/encoder/ADR/head | 基础检测性能 |
| 分支对照 B1 | `plain` | 增加同样的光谱分支，普通语义聚合 | 排除额外分支和参数带来的收益 |
| 对角统计消融 A1 | `diagonal` | 嵌入各通道背景方差＋对象兼容性 | 检验联合背景统计的价值 |
| 去对象约束 A2 | `no_compatibility` | 联合背景统计，去掉对象兼容性 | 检验背景异常误取问题 |
| 完整方法 M | `full` | 联合背景统计＋对象兼容性 | 当前主方案 |

基线配置为 [`dfine_obb_o2_fressdet.yml`](../../../configs/experiments/moda/dfine_obb_o2_fressdet.yml)，
完整方法配置为 [`spectral_full.yml`](../../../configs/experiments/moda/spectral_full.yml)。
其他三组配置与完整方法只差 `mode` 和输出目录。四组分支控制保持相同参数、初始化、取样和计算路径；
关闭的因子以零系数保留在梯度图中，兼容 `find_unused_parameters=False` 的 DDP。
这里的对角统计作用于学习嵌入的八个通道，不等同于对原始八波段分别做标准化。

基线参数量 19,551,626；四组分支模型均为 19,627,914，增加 76,288。
分支在第二层 decoder（索引 1）更新一次实例特征，同时进入分类、定位和后续层。
主干、ADR、DN、matcher、criterion 不因方法改变。输出投影零初始化，初始预测与同种子基线一致。

建议先完成 `baseline` 和 `full`，再补 `plain`、`diagonal`、`no_compatibility`。
五组是五次独立训练；12 小时预算不是五组总耗时承诺。

## 统一配方

全部使用 [`fressdet_paper_protocol.yml`](../../../configs/experiments/moda/fressdet_paper_protocol.yml)：

- 完整 train 9,156 图，官方 test 4,885 图；八波段、八类。
- 20 轮；全局 batch 8；两张 GPU 时各 4；500 个普通 query。
- 原图宽 1200×高 900，右下补零至 1216×928；从头训练，无随机增强。
- 默认 FP32，AdamW 和前三轮分组 warmup、EMA 等参数沿用现有论文优先配方。
- 每轮在官方 test 上评估；按 ProbIoU AP50 保存 `best_stg1.pth`，另存 `last.pth`。
- 新实验默认 DETR top-300，无 NMS；500 个普通 query 不变。统一输入／预算／AP，不强制复制 YOLO 后处理。
- 汇报各类别 AP50、整体 mAP50、mAP75、mAP@[.50:.95]；`--geometric` 可另外计算 `riou_` 几何指标。

完整设置见 [配方说明](fressdet_alignment.md)。所有配对实验使用相同 seed、精度、GPU 数和输入。
`AP50` 是当前评估器的整体 mAP50 字段，`per_class_metrics.<类别>.AP50` 是各类别 AP50。
主指标对照 FressDet 发布代码的 `batch_probiou` 路径；`riou_AP50` 另用真实旋转矩形交并比，不能混用。
每轮 `paper_metrics.md/csv` 按 Car、Bus、Van、Awi.、Tru.、Tri.、Bike、Ped. 顺序显示各类 AP50，
最后是整体 mAP50、mAP75、mAP，均为百分数；`metrics.json` 保持 [0,1] 原始值。

## 先检查入口和代码

```bash
conda activate wyq-deim
python tools/experiments/moda.py list
python tools/experiments/moda.py check
python tools/experiments/moda.py train full --gpus 0,1 --dry-run
```

`--dry-run` 只打印最终命令，不创建训练目录，不启动训练。

CPU 小样本检查执行真实 MODA 预处理、loss、DN、warmup、反向和推理：

```bash
python tools/experiments/moda.py preflight baseline
python tools/experiments/moda.py preflight full
```

它使用 64×96 的调试画布、20 个普通 query、batch2、4 次更新，仅验证代码。
不改变正式训练配置，不产生正式 AP，也不估计完整训练速度。

可选双进程 CPU 通信检查（本机需允许 Gloo 通信端口）：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m torch.distributed.run --standalone --nproc_per_node=2 tools/experiments/moda_ddp_smoke.py
```

## 在双 3090 上先测显存

以下命令只使用物理 GPU 0，模拟正式训练每卡 batch4，读取最密集的真实图像；
`--full-size` 保留正式的原图 resize/pad 顺序和 500 queries。

```bash
python tools/experiments/moda.py preflight baseline --gpus 0 --device cuda:0 --full-size --batch-size 4 --dense
python tools/experiments/moda.py preflight full --gpus 0 --device cuda:0 --full-size --batch-size 4 --dense
```

报告中的 `peak_allocated_bytes` 为该单进程预检峰值，未包含正式 EMA、DDP 等全部开销。
双 3090 24GB 的正式峰值和训练耗时尚未实测。脚本不会自动缩小 batch 或分辨率。

如决定使用 AMP，所有配对实验统一添加 `--precision amp`，并先以同一选项重新做 GPU 预检。
AMP 使用独立输出目录，属于相对默认 FP32 的显式精度变更；协方差计算和小矩阵求解保持 FP32。

## 正式训练

```bash
python tools/experiments/moda.py train baseline --gpus 0,1 --seed 0
python tools/experiments/moda.py train full --gpus 0,1 --seed 0
```

完成主对照后运行消融：

```bash
python tools/experiments/moda.py train plain --gpus 0,1 --seed 0
python tools/experiments/moda.py train diagonal --gpus 0,1 --seed 0
python tools/experiments/moda.py train no_compatibility --gpus 0,1 --seed 0
```

每条命令占用两张卡，应依次运行。入口自动调用当前环境的 `torch.distributed.run`，显式传 seed。
增加种子时将 `--seed 0` 改为 `--seed 1` 等，各方法保持配对。

默认输出：

```text
logs/moda/experiments/<variant>/seed0_fp32/
  run_spec.json       # 方法、种子、精度与实际启动命令
  configs.json        # 训练器保存的实际配置
  best_stg1.pth       # AP50 最佳模型
  last.pth           # 完整续训状态
  diagnostics/       # 训练、评估及源像素机制记录
    eval/epoch_XXXX/paper_metrics.md   # FressDet 表格顺序，百分数
    eval/epoch_XXXX/predictions/       # 全部最终查询，可离线重放后处理
    source_evidence/panel.json        # 固定随机子集与类别／尺度图例
    source_evidence/epoch_XXXX/index.html
```

有既存训练记录时 `train` 拒绝覆盖。可使用新的 `--runs-dir logs/moda/repeat1` 创建独立实验系列。

## 断点续训与复查评估

```bash
python tools/experiments/moda.py resume baseline --gpus 0,1 --seed 0
python tools/experiments/moda.py resume full --gpus 0,1 --seed 0
python tools/experiments/moda.py eval baseline --gpus 0,1 --seed 0
python tools/experiments/moda.py eval full --gpus 0,1 --seed 0
```

`resume` 默认读取该方法、seed、精度目录的 `last.pth`，继续原 20 轮预算，不另加 20 轮。
`eval` 默认读取 `best_stg1.pth`，输出到 `eval_best_stg1_detr/`。
`--details` 开启全测试集逐对象重诊断，耗时和磁盘开销较大；日常机制图采用下面的固定训练子集。
其他 checkpoint 可用 `--checkpoint /path/to/checkpoint.pth` 指定；已有 `run_spec.json` 时会核对方法。
AMP 运行的续训与默认路径评估需继续指定 `--precision amp`。

可选 NMS 对照使用相同 top-300 候选，仅切换抑制操作；训练配方及主评估协议不变。

```bash
python tools/experiments/moda.py eval baseline --gpus 0,1 --postprocess detr
python tools/experiments/moda.py eval baseline --gpus 0,1 --postprocess nms
```

也可直接重放已保存的最终查询，无 GPU、无训练、无网络前向：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python tools/analysis/replay_moda_postprocess.py --run logs/moda/experiments/baseline/seed0_fp32 --epoch latest
```

默认同时生成 `detr` 和 `nms` 两份全测试集结果。`--geometric` 额外报告几何 AP；
`--limit-images` 仅用于检查工具，输出明确标为子集，不能进入论文比较。无 NMS 是新实验的架构默认，是否最佳仍由配对评估检验。

## 训练自动留下哪些机制证据

每组实验固定同一随机种子，抽取 96 张训练图作为群体观察子集，另按类别与对象面积分层选 16 个图例。
自动在初始化、第 1、5、10、15、20 轮采集；按标注对象身份重新关联查询，保留失败和低分对象。
取样时只把图像有效区域元数据交给模型；标注在前向后用于身份关联，不进入光谱分支。

保存八波段原图、背景支持、对象兼容性、背景相对分数、实际聚合贡献 PNG，以及对应 NPZ、JSON、HTML。
NPZ 额外保留嵌入、原型、背景均值／协方差、更新前后查询和基线实际 attention 位置。
训练步日志分普通查询／DN 查询记录门值、背景支持、贡献总量及残差；不在每步导出巨大图集。

图例位于 `diagnostics/source_evidence/<阶段>/index.html`。完整采集约定和读图方法见
[机制证据说明](SOURCE_EVIDENCE.md)。需要重建图集时可从保留的 checkpoint 补采：

```bash
python tools/analysis/capture_moda_evidence.py --run logs/moda/experiments/baseline/seed0_fp32 --device cuda:0
```

这里 `cuda:0` 是当前进程可见的卡；补采应在卡空闲后执行。缺失 checkpoint 自动跳过，已有完整阶段不重复采集。

## 直接查看创新机制

完整方法的诊断 schema 为 `spectral-evidence-v1`。有效字段在 decoder 索引 1；
其他层的 `active=False` 和零数组表示模块未在该层调用，不能当作真实零背景统计。
记录含候选／背景坐标、背景权重、均值／协方差、目标原型、对象兼容性、相对分数、实际聚合贡献与残差范数。
坐标按画布宽高分别归一化；它们与按最长边归一化的 OBB 参数不是同一种坐标表示。

训练完成后，以下命令在固定随机训练图上选最高预测分数的 query，直接绘制取样证据：

```bash
python tools/analysis/visualize_spectral_evidence.py --checkpoint logs/moda/experiments/full/seed0_fp32/best_stg1.pth --device cuda:0 --output logs/moda/figures/full_seed0
```

可添加 `--image IMAGE_STEM --query QUERY_INDEX` 固定图像和实例查询；`--split test` 显式切换测试图。
工具保存 PNG、原图和逐点数值 NPZ，以及来源 JSON；图例并非额外 AP 评估。
这是手动追加个例的入口；论文主图优先使用上面自动预先选定的图例。

在同一模型、同一对象查询上，替换另一张图的背景统计：

```bash
python tools/analysis/visualize_spectral_evidence.py --checkpoint logs/moda/experiments/full/seed0_fp32/best_stg1.pth --image IMAGE_STEM --donor-image ANOTHER_IMAGE_STEM --device cuda:0 --output logs/moda/figures/background_intervention
```

将占位名称替换为实际 `.npy` 文件名（不带扩展名）。工具验证目标侧取样、原型和兼容性不变，
单独保存背景替换后的分数与贡献。它是冻结模型的机制干预，不是自然场景因果证明。

## 实现边界与当前验证

本分支在长度归一化的八维学习嵌入上建模相对谱形；对象原型采用同样归一化，避免 warmup 中尺度失控。
它不是物理反射率；原始亮度和空间信息仍由原主干处理。背景排除使用已有 encoder P3 对象分数，
不新增 GT 分割监督。有效背景支持不足 16 时回退对角统计，不足 2 时关闭统计证据。

局部检查已覆盖：矩形画布采样、等价 OBB、无效区域、无 GT 依赖、联合统计、消融控制、
零初始化基线等价、分类和定位反向、真实 MODA 短程更新、DDP 参数同步。
详细结果见 [实现验证记录](implementation_verification.md) 和 `logs/moda/implementation/`；代码检查通过不代表 AP 已提升。
研究主线与贡献边界见 [方案设计](../../ideas/background_conditioned_spectral_design.md)。
