# D-FINE OBB 版本

## 文档语言约定

除命令、配置名、字段名和必要论文术语外，`docs/` 下项目文档默认使用中文编写和维护。

## 四个正式入口

项目只暴露四个 OBB 训练/评估入口：

| 配置 | box head | 数据路径 | 评估器 |
|---|---|---|---|
| `dfine_obb_o2.yml` | O² ADR，六个分布 | 原图 | 通用 DOTA OBB |
| `dfine_obb_angle.yml` | 四个 FDR 分布加标量角度 | 原图 | 通用 DOTA OBB |
| `dfine_obb_o2_tile.yml` | O² ADR，六个分布 | 标准切片 | 通用 DOTA 合并回原图 |
| `dfine_obb_angle_tile.yml` | 四个 FDR 分布加标量角度 | 标准切片 | 通用 DOTA 合并回原图 |

O² 是主实现。它的默认语义是 box OCD、六变量 ADR、D-FINE decoder-wide geometry union matching，以及 O²-DFINE 论文公式中的 squared four-corner Chamfer assignment。angle 版本是刻意保留的独立架构对照，不是局部 O² 配置，也不承担已删除 checkpoint 的兼容职责。

配置名只描述模型差异。它们故意不编码数据集、backbone 大小、tile 尺寸或输入分辨率。当前可运行 recipe 注入 CODrone 数据和 HGNetv2-B2，但这些是实验选择，不是 OBB 实现名称的一部分。

项目没有累计式 OCD/ADR/Chamfer 消融配置，也没有独立 smoke 配置。测试会对这四个正式配置做临时 data-loader override，避免第五个配置逐渐偏离真正训练的代码。

## 命令

主原图模型：

```bash
CUDA_VISIBLE_DEVICES=2 python train.py -c configs/dfine/dfine_obb_o2.yml
```

直角度对照：

```bash
CUDA_VISIBLE_DEVICES=3 python train.py -c configs/dfine/dfine_obb_angle.yml
```

标准 tile 数据物化后，使用对应的 `_tile.yml` 配置。两个 tile 评估器都会把预测平移回源图，并在计算 AP 前执行类别感知 rotated NMS。

## O² 复现边界

当前 O²-DFINE 是“[论文](https://arxiv.org/abs/2603.15497)推导 + [已公开 O²-RTDETR](https://github.com/wokaikaixinxin/O2-RT-DETR) 语义对齐 + 本项目数值补全”，不是作者官方 DFINE 源码的逐行移植。截止 2026-08-20，作者仓库仍只公开了 O²-RTDETR，[O²-DFINE/O²-DEIM 的公开请求](https://github.com/wokaikaixinxin/ai4rs/issues/9)尚未得到实现。因此配置和日志必须区分三种来源：

| 语义 | 当前选择 | 来源 |
|---|---|---|
| ADR 六分布、`N=32,a=0.5,c=0.25` | 启用 | O²-DFINE 论文 |
| KLD | `sqrt=False, fun=log1p, tau=1` | 已公开 O² 配置 |
| Chamfer | `paper_squared` | O²-DFINE 论文 Eq. 10 |
| Chamfer 备选 | `released_l2` | 已公开 O²-RTDETR `ChamferCost` |
| 几何监督匹配 | decoder/pre/encoder assignment union | D-FINE 原语义 |
| OCD crowded policy | `released_dynamic` | 已公开 O² 去噪逻辑 |
| O² 角度 loss | `periodic_pi` | 论文复现默认语义 |
| direct-angle 正方形 loss | `square_aware_soft` | 本项目稳健性扩展，不属于 O² |
| 不一致六值解码 | ordered 顶点相邻边直接转 OBB | 本项目最小几何闭合；非论文显式公式 |

`num_denoising: 200` 表示期望的正负 DN query 总量。公开 O² 配置中的 `num_dn_queries: 100` 是单侧/分组基数，实际正负展开后约为 200；两个字段名不能按字面直接比较。`released_dynamic` 会优先保留每个 GT，在极拥挤图像中实际 DN 数可能超过请求值。需要硬上限时才使用 `strict_budget_random`，且日志会记录被保留和丢弃的 GT 索引。

## ADR 几何修复与已知限制

本项目已修复三类会直接改变框几何的问题：

1. 顶点选择改为中心相对坐标和统一机器精度轴吸附，避免 `torch.isclose` 的相对容差随绝对图像坐标变化。
2. 四边残差按左右、上下联合跨度修复，保留中心校正，不再逐边截断负值。
3. 六个独立分布组合成不一致四边形时，只按 ordered top/right/bottom/left 顶点的两条相邻边直接转 OBB；禁止混入其他六参数 coder 的径向投影。
4. ADR 不含隐藏的 dtype 提升；AMP、loss 计算和几何解码各自遵循框架统一精度契约。数值稳定性由正确参数化与闭式 KLD 保证，不以 GradScaler 阈值替代。

参数等价关系

\[
(w,h,\theta)\equiv(h,w,\theta+\pi/2)
\]

现在编码为同一个 ADR 几何结果，正方形本身不会因为交换长短边而产生两套目标。但 ADR 仍有另一条真实的坐标缝：当框跨过图像坐标轴时，“top vertex / right vertex”的身份会换支，归一化的 `epsilon/Wr`、`eta/Hr` 可以在接近 0 与接近 1 之间跳变，而 OBB 几何保持连续。这不是数值 bug，不能靠增加 epsilon 根治。

诊断日志已为每个 matched box 记录：

- `adr_epsilon_fraction`、`adr_eta_fraction`；
- `adr_chart_seam_distance` 及 0.001/0.01/0.05 邻域计数；
- ADR 目标残差及 offset 残差幅度；
- 超出 codebook 的 component/box 数量；
- 普通角度误差、正方形等价类修正后的误差和目标各向异性。

这条缝是 O² ADR 的已知表示限制；论文中不能宣称当前 O² 已经在拓扑上完全连续。

## 回归验证

核心测试集中在：

- `test/research/o2/test_adr_geometry.py`：近轴、平移、正方形等价类、非法六值、有限梯度和坐标缝刻画；
- `test/research/o2/test_training_parity.py`：KLD/Chamfer、union matching、OCD、配置与日志语义；
- `test/research/o2/test_dfine.py`：ADR decoder 及逐层精炼；
- `test/framework/model/test_model_pipeline.py`、`test/framework/diagnostics/test_engine_integration.py`：训练、推理、评估和诊断落盘链路。

所有机制测试都必须先于 GPU 训练通过；完整训练由实验端执行。
