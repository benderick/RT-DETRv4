# 研究经验日志

本日志只保存会改变以后决策的经验，不保存失败实现的流水账。每张卡包含命题、证据、
否决原因和以后可复用的判断规则；详细逐样本数据由链接的实验报告负责。

## EXP-2026-001：同框静默 ADR 状态不足以成为新精炼主线

- **命题**：O²-DFINE 的累计 ADR 分布中，保持当前 OBB 不变的“静默方向”包含超出
  常规不确定性的未来精炼信息，并具有可利用的下一层因果作用。
- **证据**：UAV-ROD 完整验证集 427 张，冻结 epoch 67 O² checkpoint；报告位于
  `logs/uav_rod/dfine_obb_o2/visible_silent_stage0/report.json`。
- **结果**：正式 verdict 为 `NOT_SUPPORTED`。相对 uncertainty probe 的 OOF
  delta-AUC 在三次 transition 上分别为 `+0.00985/-0.00136/-0.00097`，没有一次达到
  预注册的 `0.02`。六值 fiber 的完整集 mAP50-95 最大绝对效应仅 `0.000231`。
- **重要混杂**：把分布强制替换为双 bin 虽在早层造成 `-0.05546` mAP50-95，但同时
  把分布熵和下一层有效更新幅度大幅压缩；它证明解码器依赖累计 logits，不证明存在
  可利用的隐藏几何信息。
- **以后规则**：多对一表示中的零空间存在，并不推出零空间承载有用信号。干预强度
  必须匹配熵/浓度等低阶统计；“能改变后续输出”与“能改善任务”必须分别检验。

## EXP-2026-002：漂亮的表示定理不等于可训练的检测机制

- **历史范围**：项目曾探索 quotient/Z-FINE、边界位移、transition consistency、
  GADR 和若干潜变量精炼方向；旧实现已经清理，本卡只保留跨尝试一致出现的经验。
- **共同失败模式**：几何表示与 decoder 的特征采样、query 语义、匹配监督和更新规律
  脱节。局部求解器还会引入秩亏、尺度条件数和额外分支，随后靠例外修补使故事失去
  统一性。
- **以后规则**：新表示必须同时定义“query 看什么、状态如何更新、损失监督什么、输出
  如何求值”。如果只替换最后的 box head，即使数学上连续或等价，也不值得启动训练。
- **成本规则**：先用冻结完整集做因果/可预测性审计；Stage 0 没有非平凡效应就停止，
  不能用更长训练掩盖命题本身缺乏证据。

## EXP-2026-003：O² 复现的不可再犯错误

- O² 的四个 HBB 边界量与 `epsilon/eta` 两个顶点偏移共同定义合法旋转矩形；它不是
  任意六自由度多边形，也不能把两个偏移从水平框精炼中割裂出来解释。
- ADR 是 D-FINE 累积分布精炼在 OBB 上的一种联合实例化。验收必须观察 fixed initial
  anchor、每层累计分布、每层解码框和最终 evaluator，而不能只看最终 AP。
- 对 near-square 情形，应区分“参数代表元不唯一”和“几何框不合法”。前者需要周期/
  对称感知的损失与诊断，不能被误报成六值不构成矩形。

## EXP-2026-004：多头 deformable attention 不能事后平均成单个目标测度

- **命题**：把冻结 O²-DFINE 每个 query 的采样位置与权重跨头平均成一个归一化空间
  测度，用其中心矩同时解释特征获取、OBB 几何和逐层 refinement；若完整集审计通过，
  再进入可学习 prototype。

- **证据**：UAV-ROD 完整验证集 427/427 张、11,461 个 GT、45,844 条 query-layer
  记录，冻结 epoch 67 的 O² EMA checkpoint。正式报告位于
  `logs/research/geometric_measure_queries/uav_rod_o2_stage0/report.json`。
- **结果**：数学闭环和几何信息 gate 通过，但 refinement 行为 gate 失败，正式 verdict
  为 `NOT_SUPPORTED`。measure 相对当前框/分数 control 的 OOF 几何残差 R² 增益在四层
  为 `+0.05796/+0.05296/+0.04151/+0.03450`；更新成功 ROC-AUC 增益却只有
  `+0.01519/+0.02743/+0.01164/+0.00410`，仅 layer 1 达到预注册的 `0.02`。
- **稳健性检查**：最终 rIoU `>=0.75` 的 11,094 条高质量轨迹占全部轨迹 `96.80%`；
  在该子集上行为增益仍只有 layer 1 过线，因此结论不是由强制 Hungarian 坏匹配造成。
  测度矩形平均 rIoU 从 layer 0 到 3 为 `0.792/0.711/0.719/0.742`，分别只有
  `3.40%/0.95%/1.48%/1.27%` 比该层实际输出更好，不能直接替代 box readout。
- **原因**：DFINE 的每个 attention head 在自己的通道子空间独立取样、积分，随后拼接；
  它并未跨头平均。实测多头质心分歧约为 `0.37–0.46` 个当前框尺度。注意力权重又由
  query 线性预测，而非对采样 value 的目标占据后验。把头身份消去后得到的标量分布仍
  含尺度/形状相关信息，但不是 O² 实际采用的几何控制状态，也不能解释后层更新成败。
- **以后规则**：不能仅因 attention 权重非负且归一化，就把它解释成“物体在哪里”的
  概率测度。若重启该方向，必须建立新的 idea id，保留 head/channel 条件结构，并先用
  冻结模型检验这种向量值测度是否提供超出标量平均的样本外与干预证据；不得把本次失败
  事后改写为现有命题通过。

## EXP-2026-005：必要的 head–value 配对不等于额外矩读取器会带来增益

- **命题区分**：Stage 0 已证明，破坏 O²-DFINE 原 MSDA 中每个 head 的空间测度与 value
  子空间配对会显著破坏后续预测；Stage 1 检验的是更强命题——在原配对机制仍完整存在
  时，再把每头局部矩与积分 value 成对编码为 query residual，是否提供增量几何收益。
- **证据**：UAV-ROD、稳定 O² epoch 67、seed 42、只训练 92,992 个 adapter 参数 12
  epochs。逐轮均为 427 张完整验证集、NMS-free evaluator。裁决报告位于
  `logs/research/head_conditional_measure_queries/uav_rod_stage1_analysis/report.json`。
- **结果**：最佳 epoch 10 的 mAP50-95 为 `0.896320048`，稳定 O² 为
  `0.896323180`，差值 `-0.000003132`；预注册要求至少 `+0.003`，12 轮没有一轮超过
  基线。已有对象日志上的 mean rIoU 诊断仅约 `+0.000190`，也未达到 `+0.001`。
- **排除“没学到”**：adapter 参数全部有限、全部非零，四层均有梯度；最佳轮末残差已达
  base attention 输出的约 `8.85%–35.41%`。冻结 O² 的非 EMA 共享状态逐字节未变。因此
  这是活跃但无益的 residual，不是漏接、零梯度或 base 被意外训练。
- **数值事实**：epoch 0 有 3 次 AMP skipped step，随后 scale 稳定为 8,192 且采样梯度
  均有限。它违反预注册稳定性 gate，但不能解释后续长期无任务收益，也不能作为换精度
  后重跑的理由。
- **否决原因**：Stage 0 的破坏性干预证明的是原注意力配对的必要性，不是新增 readout
  的充分性。原 MSDA 已经把每头采样 value 分别积分并拼接，新增 adapter 很可能重复编码
  已在 query 中可用的信息；任务梯度可以把 residual 做大，却没有新的约束使其方向与
  OBB 几何误差一致。
- **以后规则**：冻结模型中的“破坏它会掉点”只能证明现有机制重要，不能直接推出“显式
  再读一次会涨点”。下一项可学习命题必须先证明新增变量包含超出现有 query/value 的
  条件增量信息，或改变信息形成方式；不能靠更长训练、额外 loss 或局部结构修补救回这
  个 prototype。

## EXP-2026-006：坐标 chart 有 seam，不等于模型存在 seam 主导的失效

- **命题**：用矩形的 support function 代替 ADR 等局部坐标图，让 query state、旋转
  作用、逐层 refinement、matching 和监督都作用在同一个无 gauge 几何对象上；进入
  prototype 前，先验证冻结 O² 是否存在与 ADR chart seam 稳定相关的旋转非等变性。
- **证据**：UAV-ROD 的 8 张预注册图像、226 条对象轨迹、36 个受控旋转角和
  pre-box/四个 decoder layer；冻结 epoch 67 O² EMA checkpoint，共 40,680 条有限记录。
  正式报告位于
  `logs/research/support_space_refinement/uav_rod_o2_stage0a/report.json`。
- **结果**：Stage 0-A verdict 为 `NOT_SUPPORTED`。最终层 near-seam/control 的均值比为
  `1.19344`，但 paired difference 的 95% CI 为 `[-0.00978, 0.06725]`，没有满足预注册
  的正下界。只有 3/8 张图达到 `1.10` 的图内均值比；去掉 image 327 后整体 ratio 仅
  `1.03635`，说明点估计不具备场景稳定性。
- **数学事实与模型事实必须分开**：固定 1° 物理旋转时，ADR target 的
  max/median 位移比为 `66.63`，support target 为 `1.00084`。这严格显示坐标 chart
  的突刺，却不能推出训练好的检测器会在该处失效。真实 O² 从 pre-box 到 layer 3 的
  平均 normalized support max error 反而由 `0.15769` 降至 `0.13834`，commutator rIoU
  由 `0.85908` 升至 `0.90413`；逐层精炼没有表现出 seam 误差累积。
- **测量边界**：最终层有 `68.6%` 的对象 track 得到正的 near-control difference，
  中位差 `0.00792`，因此存在弱趋势，但不足以立项。普通均值又会被拥挤单类场景的
  Hungarian assignment 跳变放大：最大案例把同一 GT 的旋转预测匹配到远处另一辆车，
  单条 track difference 达 `3.11244`。这既是检测集合不稳定的一部分，也使该 assay
  不能把误差干净归因于表示 seam。
- **否决原因**：当前证据只支持“support 表示在数学上更平滑”，不支持“O² 的主要可改进
  瓶颈是 ADR seam”，更不支持为此启动新的端到端表示。按预注册联合决策，不再运行
  Stage 0-B，也不把 probe 结果事后改写成另一个故事。
- **以后规则**：表示层的拓扑优雅必须先对应稳定的模型级失效。涉及集合输出的干预应
  优先采用能保持物理对象身份的测量，并以 image 为统计 cluster；若效应在 leave-one-
  scene-out 下消失，就不能用更多训练去补。以后若研究一般 support-field readout，必须
  以新的 idea id、独立命题和独立预注册证据重启，不能复活本命题。

## EXP-2026-007：冻结分布的方差几乎没有改变下一层实际取证位置

- **命题**：O² 的 ADR 分布不应先压成均值框再决定下一层 attention；在固定采样预算
  下，让采样原子分布到六维 `mean±sigma` 框，应读出被点估计丢失的精炼证据。
- **证据**：UAV-ROD 完整验证集 427 张、11,461 个对象、34,383 个相邻层 transition；
  报告位于 `logs/research/geometry_marginalized_attention/uav_rod_o2_stage0/report.json`。
  mean replay、成对 sigma 日程、跨图 shuffle 和模型状态不变等实现不变量全部通过。
- **结果**：正式 verdict 为 `STAGE0_NOT_SUPPORTED`。sigma 相对每层最强 control 的
  grouped-OOF mean R² 增益依次为 `-0.004292/-0.002844/-0.002153`，三个 transition
  全部失败；六维 sigma frame 对实际采样位置的平均位移仅为归一化坐标
  `0.000162/0.000136/0.000128`。
- **否决原因**：当前训练好的 ADR 分布已经非常集中；在相同原子预算下边缘化它，几乎
  没有产生不同视觉观测。漂亮的 Jensen-gap 命题需要足够大的状态不确定性，这一必要
  条件在稳定 O² 上并不存在。
- **以后规则**：不要只从“非线性函数的期望不等于期望处函数值”推出可用效应；先测量
  分布在真实算子输入空间造成的位移和 observation change。若干预在算子输入层几乎为
  零，probe 或更复杂求积都不值得训练。

## EXP-2026-008：视频数据按 image 做 OOF 会制造相邻帧假阳性

- **命题**：用完整预测集合估计 query 成为阈值化一对一匹配唯一代表的 survival curve，
  能在框固定不变时改善 O² 排序。
- **证据**：UAV-ROD epoch-71 的 427 张、6 个视频、19,056 个 NMS-free 候选；报告位于
  `logs/research/metric_aligned_set_survival/uav_rod_o2_stage0/report.json`。
- **结果**：按 image 分组的开发预检曾给出约 `+0.00333 mAP50-95`，但相邻视频帧被拆到
  训练/验证 fold。改为整段 `video_id` 不可拆分后，absolute MASS 仅从
  `0.8966326` 到 `0.8967329`（`+0.0001003`）；把集合 probe 当作 O² 的 log-odds
  residual 更降至 `0.8826063`。正式 verdict 为 `STAGE0_NOT_SUPPORTED`。
- **关键区分**：集合特征能提高 unique-query AUC，不等于把它混入 detector score 会
  提高 AP。AP 还依赖原分数中类别、定位和跨图全局排序的强校准；一个 probe 的判别信息
  不能自动解释为可相加的似然比。
- **以后规则**：视频/航拍序列的所有 OOF、bootstrap 和 shuffle 必须以完整视频为最小
  独立单元；任何 image-level 正结果都只算泄漏风险预检。代理 AUC/R² 通过不能代替原框、
  原 evaluator 上的固定输出反事实 AP。

## EXP-2026-009：几何质量变好不等于置信度质量同时变好

- **命题**：coarse-to-fine decoder 的 metric fidelity 应随 state fidelity 共同演化；
  O² 四层以固定 `[0,1/3,2/3,1]` 从 HBox IoU 过渡到 rIoU，并让同一质量同时监督
  VFL target 与 FGL weight。
- **证据**：UAV-ROD、seed 42、稳定 O² 与隔离 MHR 各 72 epoch，模型/数据/优化器完全
  相同；完整裁决位于
  `logs/research/metric_homotopy_refinement/uav_rod_o2_stage1/report.json`。
- **正结果**：最佳 mAP50-95 为 `0.900812133`，比 O² 的 `0.896560818` 高
  `+0.004251315`。增益集中在 AP85/90/95（`+0.009318/+0.018053/+0.012740`）；
  11,461 个相同 GT 的 mean rIoU 增加 `+0.002796`，6 个视频各自的均值均非负，
  video-bootstrap 95% 区间为 `[+0.001443,+0.005112]`。
- **未通过项**：matched score-rIoU Pearson/Spearman 降低，MAE 增大；因此联合机制没有
  实现所声称的 confidence alignment。鲁棒 exact-rIoU 使同步 median training step
  耗时增加 `+7.61%`，也超过预注册 `5%`；DOTA-07 主指标微降 `0.000585`。
- **否决原因**：一次同时改变 VFL 与 FGL 的训练显示了可信几何效应，却不能支持“同一
  同伦统一改善置信度和定位”的更强命题。不能在看到结果后只保留成功的 FGL 解释，也
  不能把训练内核优化事后从 gate 中删去。
- **以后规则**：quality-aware detector 中，classification calibration 与 localization
  weighting 是两个可分离的因果通道；一个共同 scalar target 的形式统一不保证行为
  统一。若以后研究几何 weighting，必须以新 idea 预注册单一路径和归因消融，不得原地
  修改或复活已经失败的 MHR；单 seed 正结果只授权提出候选，不授权论文结论。

## EXP-2026-010：分离冲突角色能恢复校准，但不能自动保留联合干预的几何收益

- **命题**：classification target 与 localization credit weighting 不应共享一个
  universal quality metric；VFL 保持 HBox quality，只有 FGL 按 decoder 深度从 HBox
  过渡到 rIoU，应同时保留 MHR 的严格几何收益和稳定 O² 的分数校准。
- **证据**：UAV-ROD、seed 42，稳定 O²、shared-MHR 与 RFMR 三条各 72-epoch 受控训练；
  正式报告位于
  `logs/research/role_factored_metric_refinement/uav_rod_o2_stage1/report.json`。
- **机制验收**：206 个结构化 step 全部精确记录 VFL `[0,0,0,0]` 与 FGL
  `[0,1/3,2/3,1]`；3,914 条记录的最大恒等误差为 `1.413e-7`。参数、推理图和显存不变，
  因此不能用“实现没生效”解释结果。
- **正结果**：mAP50-95 相对 O² 为 `+0.002264`；score-rIoU Pearson/Spearman 提高
  `+0.032387/+0.015874`，MAE 降低 `0.001460`。角色分离确实消除了 shared-MHR 的
  score-alignment 退化。
- **否决原因**：预注册几何门槛未过：mAP 增益低于 `0.003`，AP80:95 平均增益仅
  `0.002444<0.005`，paired mean-rIoU 的 video-bootstrap 95% 区间
  `[-0.000372,+0.002772]` 跨零；training step 增加 `5.81%>5%`。RFMR 只保留 MHR
  dense 增益的 `53.3%`，不能支撑“FGL-only 是几何收益来源”的归因。
- **以后规则**：一次联合干预的成功效应不能靠删除失败通道完整继承；role conflict
  成立与 role-factorized mechanism 足够有效是两件事。三条件结果应画在同一
  geometry–calibration plane 上；若联合效应存在交互，不追加未预注册路径来事后补全
  因果故事。

## EXP-2026-011：局部几何排序正确，不等于残差分布是可最大化的对象后验

- **命题**：O² 的六个 ADR 边缘分布是同一合法旋转框的冗余测量；将六个累计 logit
  通过合法框编码器拉回并相加，可以恢复逐维期望丢失的联合证据。若这个对象能量确实
  是框后验，那么最大能量合法框应优于 O² 的逐维期望，并可逐层回灌继续精炼。
- **证据**：UAV-ROD 完整验证集 427/427 张、11,461 个匹配对象、6 个视频；冻结
  epoch 67 O² EMA checkpoint。正式报告位于
  `logs/research/pullback_distribution_refinement/uav_rod_o2_stage0/report.json`。
  零搜索 replay 与原 O² 的 11 类张量逐元素完全一致，模型状态哈希不变，所有候选框
  合法且搜索能量单调，因此不能用实现偏差解释结果。
- **表面支持**：最终层固定局部候选中，六因子能量与 GT rIoU 的对象内 Spearman
  中位数为 `0.84000`，video-cluster bootstrap 95% 区间为
  `[0.83427, 0.84274]`。这说明 ADR logit 的局部斜率通常指向更好的几何方向。
- **反事实结果**：最终层只有 `16.95%` 的对象能从局部搜索获得更高能量；在这些真正
  改变的对象上，post-hoc rIoU 平均反而下降约 `0.00277`，负例多于正例。完整 evaluator
  上 post-hoc mAP50-95 下降 `0.000717`。逐层回灌的 matched mean rIoU 下降
  `0.002006`，video-bootstrap 95% 区间为 `[-0.004944,-0.000653]`；mAP50-95
  下降 `0.003960`、AP75 下降 `0.001478`，推理耗时为 O² 的 `4.159` 倍。
- **根因**：训练只要求每个 ADR 边缘在当前 anchor 条件下拟合确定性目标残差，并没有
  把六因子乘积训练成相对于真实框校准的联合后验。最终层 GT 的对象能量在约 `93.7%`
  的有效匹配上低于 O² 输出；能量更像网络自身条件残差场的自洽度。局部 rank correlation
  只说明小扰动排序大体正确，不能保证其 mode、跨坐标尺度或跨层累加是正确决策规则。
- **容易误读的子集**：PDR 对少数低 rIoU 基线框有正均值，但这些对象只占很小比例；
  搜索能量增量与真实 rIoU 增量的相关仅约 `0.11`，没有不依赖 GT 的可靠门控证据，不能
  事后包装成“只修困难样本”的新方法。
- **以后规则**：不能把为确定性残差监督产生的 distribution logits 事后解释为结构化
  posterior。候选排序、posterior calibration 和最终 Bayes decision 是三个不同命题。
  若再研究分布式框预测，必须从训练时就定义对合法对象的 proper scoring rule、决策
  损失及跨层条件更新，并先证明它包含超出当前点框与 query 的条件增量信息；不得靠更
  复杂的 mode search 或 GT 不可见的事后门控复活 PDR。
