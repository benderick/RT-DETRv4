# BCSR：边界条件光谱细化

状态：feasibility，代码可运行性验证阶段，无正式 AP 结果。完整数据/运行协议见
[MODA 文档](../../datasets/moda/README.md)，配置与所有权见 [manifest](manifest.json)。
已完成检查及限制见 [验证记录](verification.md)。

## 科学问题与方法

暂定主张：同一目标不同边缘的有效光谱证据可能不同；用当前预测框的内外边界对比
选择波段，并随 decoder 更新采样位置，可能改善密集、小目标的定位精度。
该现象尚未在真实 MODA 上被实验确认。边界模糊、跨波段配准误差和早期框不准都可能
使局部读取失效，整框共享光谱权重也可能已经足够。

保持本项目 O² 的 ADR、matcher、DN、损失和固定分布 anchor。一个共享的小型 CNN
分别处理八个波段，保留 band 轴，得到 stride-4 特征。每层使用该层**输入预测框**的
四条边，每边 4 个位置，采样内侧和外侧；偏移为短边的 0.15 倍，限制在 1–8 输入
像素。有效内外采样对组成 `[inside,outside,inside-outside]` 描述子，与 query、
波段 ID embedding、物理边方向共同预测光谱 softmax 权重。边特征与边方向先做
非线性交互，再作集合平均，输出零初始化的几何头残差。

没有单独边界监督，没有 GT 框驱动采样，没有固定波长假设。物理像素空间构造网格，
适用于非方形画布。四边共享参数，无边索引 embedding，因此等价的半周角表示、
宽高交换表示只置换边集合；这是**框表示等价性**，不等于整个模型的旋转等变性。
有效区域 mask 同步增强，内外有一端落入空白则去掉该采样对；全无效时残差严格为零。

零初始化保留初始基线预测；因为原几何头也有零初始化，早期几个更新后才应观察到
内部路由/浅层分支的非零梯度。需要用实际梯度和残差诊断证明激活，不能凭参数存在
判断机制已学会。残差只直接加到几何头输入，但 LQE 和下一层引用框会使分类间接受影响。

## 与 FressDet 的区别

[本地源码审计](../../datasets/moda/fressdet_reference_audit.md) 已核对用户提供的 v2。
FressDet 的 SpeIW 处理连续有序光谱场，ReCoW 用光谱/空间路由一致性调制特征，配合
C4 网络和 oriented-aware head。本方法研究的最小命题是：**预测框边界上的光谱证据，
是否比整框共享光谱权重更有利于 O² 定位细化**。不以“光谱＋旋转”作为创新，不将
零初始化残差或 grid_sample 自身当论文贡献。更完整的新颖性审查和实证证据仍需继续。

## 配对实验

| 组别 | 配置 | 回答的问题 |
|---|---|---|
| B0 | `configs/experiments/moda/dfine_obb_o2.yml` | 八通道 O² 基线 |
| B1 | `configs/incubator/bcsr/moda_object.yml` | 相同分支、整框共享光谱权重的收益 |
| B2 | `configs/incubator/bcsr/moda_edge.yml` | 每边独立权重是否额外改善定位 |
| B3 | `configs/incubator/bcsr/moda_initial.yml` | 每层更新采样框是否必要 |

B1/B2/B3 参数名和形状完全相同，区别仅为权重共享或采样引用选择；B3 的固定框是
**第一层输入 reference**，不是 O² 内部另一个固定 pre-box ADR anchor。B0 和 B2
参数量不同，不能单靠 B2>B0 支持边界路由主张。

保持数据分割、图像尺度、源 difficulty、预训练、seed、优化器、普通 queries、DN
策略、训练轮数、评估器、max_det 一致。先用完整 train 来源的 train/dev 建立基线
并对齐学习曲线；开发阶段不读取 test。条件允许时配对 seed 42/43/44。冻结超参数
与训练轮数后，以全部 9,156 张 train 训练最终模型，并用固定末轮 checkpoint 测试。
若 12 小时预算内未收敛，报告学习曲线和实际耗时，不把不同训练进度当方法差异。

首个训练验证关注：新增分支是否持续获得有限非零梯度、残差是否激活、是否只有少数
波段永久占优、有效采样比例、小目标 AP75 和原图几何 IoU。完整配对训练才回答 AP
收益；短程代码 smoke、合成测试和可用图前缀不支持泛化或论文性能主张。

B2 若只胜 B0、不胜 B1，应收窄为额外光谱分支的工程收益。未激活先查实现和优化；
统计不足则结论为证据不足，不据一次短训练否定核心假设。边界故事若缺少机制证据，
不能用选择好看的注意力图替代。

## 运行

```bash
conda activate wyq-deim
python -m unittest test.research.bcsr.test_adapter -v
python tools/dataset/moda_preflight.py --config configs/incubator/bcsr/moda_edge.yml --debug-split logs/moda/debug_split/debug.json --steps 4 --output logs/research/bcsr/preflight_cpu.json
```

在目标机器先用 `moda_preflight.py --device cuda:0 --amp --size 1024 --queries 500
--batch-size 4 --dense` 测量密集 batch 显存。完整数据、开发分割和资源准备好后：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py -c configs/incubator/bcsr/moda_edge.yml --seed 42
```

全训练集固定轮数入口为 `configs/incubator/bcsr/moda_edge_fulltrain.yml`，不构造
验证集；测试入口为 `configs/incubator/bcsr/moda_edge_test.yml`，需
`--test-only -r logs/research/bcsr/moda_edge_fulltrain/last.pth`。不同消融/seed 使用
不同 `--output-dir`，正式训练前 `git status --short` 必须为空。双 3090 的正式显存、
吞吐、12 小时完成性均待测，当前配置的全局 batch 8 只是起点。

小型两进程 CPU 验证（需要本机 loopback socket）：

```bash
torchrun --standalone --nproc_per_node=2 tools/research/bcsr/smoke_ddp.py
```

训练通用日志有 `boundary_spectral` 梯度组。详细 eval query 记录中的
`stages[*].method_diagnostics` 保存 `bcsr-query-v1` 的 `band_weights`、
`sampling_points`（归一化画布坐标）、`valid_fraction`、`residual_norm`、
`input_reference`。权重全无效时仍有 softmax 数值，解释时必须用有效比例屏蔽。
这些记录可用于逐边波段图、定位变化、有效读取比例及机制激活分析；本阶段尚未生成
论文结果图。
