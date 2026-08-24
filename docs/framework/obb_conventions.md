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

O² 是主实现。它的默认语义是 box OCD、六变量 ADR、D-FINE decoder-wide geometry union matching，以及已公开 O²-RTDETR 实际使用的 L2 four-corner Chamfer assignment。angle 版本是刻意保留的独立架构对照，不是局部 O² 配置，也不承担已删除 checkpoint 的兼容职责。

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

当前 O²-DFINE 是“[论文](https://arxiv.org/abs/2603.15497)推导 + [已公开 O²-RTDETR](https://github.com/wokaikaixinxin/O2-RT-DETR) 语义对齐 + 本项目数值补全”，不是作者官方 DFINE 源码的逐行移植。截止 2026-08-24，作者仓库仍只公开了 O²-RTDETR，[O²-DFINE/O²-DEIM 的公开请求](https://github.com/wokaikaixinxin/ai4rs/issues/9)尚未得到实现。因此配置和日志必须区分三种来源：

| 语义 | 当前选择 | 来源 |
|---|---|---|
| ADR 六分布、`N=32,a=0.5,c=0.25` | 启用 | O²-DFINE 论文 |
| KLD | `sqrt=False, fun=log1p, tau=1` | 已公开 O² 配置 |
| Chamfer 默认 | `released_l2` | 已公开 O²-RTDETR `ChamferCost`，对应公开结果的可执行语义 |
| Chamfer 公式审计 | `paper_squared` | O²-DFINE 论文 Eq. 10；与公开代码存在明确差异，不作为默认复现 |
| 几何监督匹配 | decoder/pre/encoder assignment union | D-FINE 原语义 |
| OCD crowded policy | `released_dynamic` | 已公开 O² 去噪逻辑 |
| O² 角度 loss | `periodic_pi` | 论文复现默认语义 |
| direct-angle 正方形 loss | `square_aware_soft` | 本项目稳健性扩展，不属于 O² |
| 六值解码 | 四边 HBox + 两个顶点偏移 + 等对角线矩形闭合 | 论文定义与标准 midpoint/gliding-vertex 几何共同确定 |

`num_denoising: 100` 与 D-FINE、已公开 O² 的字段语义一致：它是正负展开前的动态分组基数，实际 DN query 名义数量为 200。`released_dynamic` 会优先保留每个 GT；若单图 GT 数超过 100，至少保留一组正负样本，实际数量会超过 200。需要硬上限时才使用 `strict_budget_random`，且日志会同时记录 group base、名义/实际 query 数和被保留/丢弃的 GT 索引。

## ADR 几何契约与已知限制

ADR 使用一条完整而唯一的几何链：

```text
初始 OBB
  -> 初始外接 HBox 及 epsilon/eta
  -> 六组 D-FINE 式累计分布
  -> 四边得到精炼 HBox
  -> epsilon/eta 得到 top/right 两条中心对角方向
  -> 等对角线闭合为严格矩形
  -> canonical OBB，作为下一解码层的旋转 reference
```

其中前四组不是另起炉灶的 OBB 表示，而是原样继承 D-FINE 的 left/top/right/bottom 累计精炼；后两组才把水平外接框变成旋转框。六个预测通道都有监督和梯度。合法矩形满足

\[
\epsilon(W_r-\epsilon)=\eta(H_r-\eta),
\]

这是六值中间态的一条矩形一致性关系。训练中的六个独立期望不必天然满足它，因此标准 midpoint/gliding-vertex 几何把 top/right 两条中心对角线缩放到共同半径；中心对称且等对角线的四边形必为矩形。实现使用纯 PyTorch 闭式，不调用 OpenCV `minAreaRect`。

实现必须满足以下几何约束：

1. 顶点选择改为中心相对坐标和统一机器精度轴吸附，避免 `torch.isclose` 的相对容差随绝对图像坐标变化。
2. 四边残差按左右、上下联合跨度修复，保留中心校正，不再逐边截断负值。
3. 轴对齐时按偏移起点选择 top-right 与 bottom-right，因此 `epsilon=eta=0`；等价的 `(w,h,theta)` 参数化由全局几何极值编码为同一 ADR target。
4. 六值预测使用等对角线矩形闭合，任何中间输出都得到严格矩形。
5. ADR 不含隐藏的 dtype 提升；AMP、loss 计算和几何解码各自遵循框架统一精度契约。数值稳定性由正确参数化与闭式 KLD 保证，不以 GradScaler 阈值替代。

参数等价关系

\[
(w,h,\theta)\equiv(h,w,\theta+\pi/2)
\]

现在编码为同一个 ADR 几何结果：top/right 是图像坐标中的全局极值顶点，不是随局部 `(w,h)` 命名交换的顶点。因此提问中“交换参数后 top/right 必然交换”的前提不成立，ADR 不会仅因这组等价参数化生成两个 target。

需要区分三个问题：KLD 对正方形的任意角度都失去辨别力，CDC 用四角点集合解决的是这个 matching 问题；精确正方形旋转 90° 本来就是同一几何，CDC 也应给零；O² 的五参数 L1 在精确正方形上仍可能惩罚一个几何等价的 90° 参数化，这是论文基线保留的潜在 loss 局限。ADR 另有一条非正方形专属的坐标缝：当框跨过图像坐标轴时，top/right 极值顶点换支，`epsilon/Wr`、`eta/Hr` 可在接近 0 与 1 之间跳变，而 OBB 几何连续。

诊断日志已为每个 matched box 记录：

- `adr_epsilon_fraction`、`adr_eta_fraction`；
- `adr_chart_seam_distance` 及 0.001/0.01/0.05 邻域计数；
- ADR 目标残差及 offset 残差幅度；
- 超出 codebook 的 component/box 数量；
- 等对角闭合前 raw gliding quadrilateral 的正交误差；
- 普通角度误差、正方形等价类修正后的误差和目标各向异性。

这条缝是 O² ADR 的已知表示限制；论文中不能宣称当前 O² 已经在拓扑上完全连续。

O² checkpoint 持久化 `adr_geometry_signature=[4,2,1]`，分别表示四个外接边界、
两个顶点偏移和等对角矩形闭合。验收严格检查该签名，防止不同几何契约的权重
被静默加载。direct-angle checkpoint 不包含该字段。

## 回归验证

核心测试集中在：

- `test/research/o2/test_adr_geometry.py`：近轴、平移、正方形等价类、非法六值、有限梯度和坐标缝刻画；
- `test/research/o2/test_training_parity.py`：KLD/Chamfer、union matching、OCD、配置与日志语义；
- `test/research/o2/test_dfine.py`：ADR decoder 及逐层精炼；
- `test/framework/model/test_model_pipeline.py`、`test/framework/diagnostics/test_engine_integration.py`：训练、推理、评估和诊断落盘链路。

所有机制测试都必须先于 GPU 训练通过；完整训练由实验端执行。
