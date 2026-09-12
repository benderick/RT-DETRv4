# 本地 FressDet v2 核对

核对日期：2026-09-12。来源是用户提供的本地参考目录，未在本项目中训练 FressDet，
也未验证其论文数字。论文版本路径：
`research/references/FressDet/paper/arXiv-2607.05148v2/main.tex`；代码根目录：
`research/references/FressDet/code/FressDet/`。外部资料不进入本项目 Git。

## 方法边界

v2 描述 SpeIW 连续、有序、单调的光谱重采样；ReCoW 的光谱 soft routing 与空间
hard routing 产生一致性驱动的残差调制；C4 旋转等变 backbone/neck 与 oriented-aware
head。v2 明确修正了早期附录的描述：ReCoW 不是两个分支的标量凸组合。
不能把“光谱注意力＋旋转”本身作为新颖性，也不能把 C4 的条件等变性泛化成任意角度
的严格等变性。

## 对比时必须统一的项

| 项目 | 本地 FressDet 源码 | 本项目 MODA 入口 |
|---|---|---|
| 输入存储 | `data/base.py` 中 `transpose(2,1,0)` 成 HWC | CWH→CHW，八通道 |
| 类别 ID | `car,pedestrian,bike,awning-bike,van,truck,bus,tricycle` | `car,van,truck,bus,tricycle,bike,awning-bike,pedestrian` |
| 框匹配 | `models/yolo/obb/val.py: _process_batch` 调用 `batch_probiou` | 旋转矩形几何 IoU |
| AP | Ultralytics 路径；需进一步锁定插值/匹配语义 | DOTA-07 及 101 点 dense AP |
| 数据入口 | `cfg/datasets/drmod.yaml` 占位路径，images/train、val、test | 原始 MODA train/test，加版本化 dev 清单 |
| 示例训练 | `train.py`：imgsz=1200、20 epochs、batch=12、两卡 | 1024、36 epochs、global batch=8 的待测配方 |

如果预测遵循 FressDet 附带 YAML 类别编号，转换到原始 MODA ID 的映射为
`[0,7,5,6,1,2,3,4]`。这只适用于已确认使用该 YAML 的权重，不能盲目套到所有权重。

论文 v2 附录写“rotated-rectangle IoU”，但本地代码验证路径实际调用 ProbIoU；两者
不是同一函数。这是尚待作者或外部复现实验澄清的协议差异，不能直接推断论文结果错误。
附带配置没有提供与当前 train/dev/test 对应的样本清单，也不能单凭名称 DrMOD 推断
已经与本地 MODA 测试集一致。

正式比较应导出两边原图坐标预测、核实类别映射与样本清单，在同一 evaluator 中重算，
同时记录各自原生指标作为补充。保持 difficulty、置信度阈值、max_det、图像尺度、
数据增强、预训练和训练预算可审计。当前未完成统一评估，不把论文报告值与本地 dense
AP 拼成一张直接排名表。
