# O²-DFINE 复现逻辑审计

## 结论与来源边界

当前 `refinement_mode: o2_adr` 是可运行、可测试的 paper-derived
O²-DFINE reproduction。作者公开仓库提供 O²-RTDETR，但没有提供完整的
O²-DFINE/ADR decoder，因此本项目将论文明确规定、公开代码语义和必要的几何
补全分别标记，不能称为作者官方源码的逐行复刻。

稳定入口只有：

```text
configs/dfine/dfine_obb_o2.yml
configs/dfine/dfine_obb_o2_tile.yml
```

## ADR：四边界与两偏移共同精炼 OBB

第一层 decoder 同时产生：

1. traditional OBB head 给出的初始 `(cx,cy,w,h,theta)`；
2. O²-DFINE head 给出的六组初始分布。

初始 OBB 转换为外接 HBox。前四组分布严格继承 D-FINE，对固定初始中心到
left/top/right/bottom 的距离进行累计 logits 精炼；后两组分布分别精炼：

- `epsilon`：外接框 top-right 到 OBB top vertex 的水平距离；
- `eta`：外接框 bottom-right 到 OBB right vertex 的竖直距离。

每层使用同一个初始 OBB 和初始外接尺寸作为锚：

```text
initial OBB
  -> initial external HBox + epsilon/eta
  -> cumulative six-distribution logits
  -> four refined HBox boundaries + two refined vertex offsets
  -> top/right/bottom/left vertices
  -> exact rectangle
  -> next-layer 5D rotated reference
```

六个量不是独立的六自由度，也没有多余 head。合法矩形满足

[
epsilon(W_r-epsilon)=eta(H_r-eta).
]

六个独立分布的训练中间态不必天然满足这条关系。标准 midpoint/gliding-vertex
解码把 top/right 两条中心对角方向缩放到共同半径；中心对称且等对角线的
四边形必为矩形。该步骤只定义任意中间六值怎样落到合法 OBB，不增加预测量。
实现为纯 PyTorch，不调用 OpenCV。

checkpoint 持久化 `adr_geometry_signature=[4,2,1]`：四个外接边界、两个
顶点偏移、一个等对角矩形闭合契约。验收使用 strict load 并检查该签名。

## 四个 O² 机制

| 机制 | 当前实现 | 来源 |
|---|---|---|
| ADR | 六个非均匀分布、逐层累计 logits、FGL 监督六分量 | O²-DFINE 论文 |
| CDC | KLD + 四顶点 Chamfer Hungarian cost | 论文与公开 O² 代码 |
| OCD | box/angle/geometric/probability 原语，主配置为 box noise | 论文与公开 O² 代码 |
| Rotated cross-attention | 局部采样偏移按 OBB 方向旋转 | O² 论文描述 |

criterion 记录 decoder 层间 query-index instability。它是验证 OCD 动机的诊断量，
不是第五个模型模块。

## CDC 与 KLD square problem

KLD 把 OBB 映射为高斯协方差。精确正方形的协方差与角度无关，因此 KLD 无法
区分同中心、同尺寸但几何朝向不同的正方形。CDC 直接比较四个角点集合：

- 相差 45° 的两个正方形角点集合不同，Chamfer 大于零；
- 相差 90° 的精确正方形本来就是同一几何，Chamfer 为零。

论文 Chamfer 公式写 squared L2，而公开 O²-RTDETR `ChamferCost` 使用 ordinary
L2。主配置采用公开可执行语义 `released_l2`；`paper_squared` 只作为明确命名
的论文公式审计模式。

主 matcher 权重为 focal 2、KLD 2、Chamfer 5，bbox/angle matching cost 为零。
criterion 的 KLD 使用公开配置的 `sqrt=False, fun=log1p, tau=1`。

## OCD：为什么选择 box noise

主配置 `ocd_mode: box` 对齐公开 `only_xyxy` 路径：保持数值 theta 不变，
把 `(cx,cy,w,h)` 转为 `xyxy`，分别扰动四个坐标，再代数转换回
`(cx,cy,w,h)`。

论文消融中 box noise 最好。angle noise 只改变 theta，变化范围较窄；geometric
noise 同时改变 box 与 angle，容易造成过强 rIoU 下降；probability noise 主要
改变协方差，不能有效移动中心。

`num_denoising=100` 是正负展开前的动态分组基数，实际名义 query 数约为 200。
`released_dynamic` 在单图 GT 超过 100 时保留至少一组正负 query，因此实际数量
可以超过 200；日志记录名义数量、实际数量和 crowded policy。

## Rotated cross-attention

当 reference 是 `(cx,cy,w,h,theta)` 时，每个 D-FINE 非均匀 sampling offset
先在局部坐标按 `(w,h)` 缩放，再将完整二维向量按 theta 旋转到图像坐标，最后
加到中心位置。测试覆盖任意二维向量和各向异性 reference，不只覆盖 90° 特例。

公开 O²-RTDETR 实现将旋转矩阵作用于 `(w,h)` scale 后再与 offset 逐坐标相乘，
与论文“sampling points rotate”的描述并不等价。未公开的 O²-DFINE 无法逐行
核对；当前采用论文所描述的向量旋转，并在来源记录中明确披露该差异。

## ADR 是否存在 square problem

[
(w,h,	heta)equiv(h,w,	heta+pi/2)
]

不会产生两套 ADR target。ADR 的 top/right 是图像坐标中的全局极值顶点，不是
局部宽边/高边的顶点编号；等价参数化具有同一个角点集合，因此编码得到相同的
`epsilon/eta`。轴对齐 tie 使用固定的 top-right/bottom-right 规则，此时
`epsilon=eta=0`。

需要保留两个真实限制：

1. 框连续穿过图像坐标轴时，top/right 极值顶点可能换支，offset chart 存在 seam；
2. 论文基准的五参数 L1 对精确正方形的 90° 等价参数化仍可能产生非零损失。

日志记录 offset fraction、chart seam distance、codebook overflow 和等对角闭合前
的 raw orthogonality error，使这些问题可以直接从实验日志分析。

## 当前实验契约

CODrone recipe 使用 1024 输入、HGNetv2-B2/M、四层 decoder 和 600 matching
queries。论文通用设置写的是默认 top-K 300；600 是 direct-angle 与 O² 共同采用的
CODrone 密集目标适配，不属于 O² 创新，也不能用来声称绝对 DOTA 数值复现。

完整原图推理不执行 rotated NMS；tile 配置在预测平移回源图后执行类别感知
rotated NMS。

新训练完成前，O² 验收状态为 **PENDING**。训练后使用
`tools/research/o2/run_acceptance.py` 在同一完整验证集上报告 prebox、每个
decoder layer、最终输出、NMS 对照、逐类 AP、matched rIoU、过修正案例和
raw ADR consistency。

## 回归门槛

- `test/research/o2/test_adr_geometry.py`：encode/decode、轴边界、平移、
  square、非法六值和完整 KLD backward；
- `test/research/o2/test_dfine.py`：累计六分布、OCD、CDC、旋转采样和配置；
- `test/research/o2/test_training_parity.py`：来源语义、square matching、
  union matching 和 crowded DN；
- `test/framework/model/test_model_pipeline.py`：训练、推理和后处理；
- `test/framework/model/test_full_smoke.py`：真实 1024 CUDA autocast
  forward、loss、backward 和 inference。

任何后续方法必须使用独立 `refinement_mode` 或研究配置，不得修改 O² 默认语义。
