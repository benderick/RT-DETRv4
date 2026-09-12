# BCSR 代码验证记录

日期：2026-09-12。环境 `wyq-deim`，Python 3.10.16、PyTorch 2.3.0+cu121。
未安装新依赖，未进行完整数据训练，未获得正式 AP 或论文增益。

## 已完成

- 项目统一 CPU 测试（加入矩形输入与 benchmark 配方后）：172 项，169 通过，
  3 项因 CUDA 在沙箱中不可见而跳过。
  跳过项是既有的三项真实数据 1024 CUDA smoke；没有把断言失败改为 skip。
- 对改动前 `7b5f727` 的 decoder 源码做独立比对：direct-angle、O² 参数树严格加载，
  train/eval 的预测框和类别 logits 数值完全一致。
- 通用零残差接口测试：两种 decoder 的全部 loss、参数梯度完全一致。
- 数据：CWH→CHW、同步增强、mask、difficulty=0/1/2 有效、空标注、perfect-prediction
  AP、缺图失败、split hash、场景组不交叉等测试通过。
- BCSR：非方形画布像素采样、半周角和宽高交换等价、全无效区域零输出、GT 无关、
  padding 内容无关、消融参数形状一致、相同 seed 的共享基线参数一致均通过。
- 新模块的训练链路覆盖 regular/aux/encoder/pre/DN、空 GT、反向、有限梯度、连续
  四次更新后浅层分支与路由参数梯度非零；eval 输出和 query 诊断落盘通过。
- 两进程 CPU/Gloo：三种消融均完成四次优化，`find_unused_parameters=False`，所有
  adapter 参数在两进程间严格一致、路由梯度非零。尚未在两张 3090 上验证 NCCL。

## 真实 MODA 小样本检查

使用显式 16 图 debug 清单，非正式开发/测试协议。全模型 HGNetv2 B2 的 CPU
preflight 完成真实数据变换、四次优化和推理；尺寸 128、64 个普通 queries。
另以该清单完成真实训练器的一轮 CPU 检查（8 个 batch、每 batch 2 图），验证不构造
验证集的训练入口、训练数据 provenance、梯度诊断、checkpoint 保存。随后使用保存
的 `last.pth` 和实际解析配置通过通用推理 CLI 严格加载，对一张 `.npy` 图完成
原图框恢复、DOTA 文本和灰度预览导出。该 checkpoint 仅用于代码 smoke。

A100 40GB 单卡 AMP 使用相同样本与 seed 对照：基线与 BCSR 均从 scale=65536
经历六次溢出跳步，降至 1024 后连续完成四次有限梯度更新。主要残余溢出位于基线
backbone 首层卷积；没有新增局部强制 FP32、nan_to_num 或降低默认初始尺度来掩盖。
preflight 保留每次 scale、skipped、非有限梯度计数，最多允许 12 次校准尝试。

| 项目 | 八通道 O² | O² + BCSR |
|---|---:|---:|
| 参数数 | 19,551,626 | 19,635,146 |
| 本次 128 尺度 AMP 峰值 allocated bytes | 495,617,024 | 497,313,792 |

新增 83,520 参数（约 0.43%）。这张表只记录上述小规模检查：不代表 1024 训练显存，
不代表 reserved 显存或系统全部 GPU 占用，也不能证明吞吐、训练时间或高分辨率额外
显存很小。目标双 3090 的 1024 密集 batch 和 12 小时训练预算仍需实测。

本地证据位于 Git 忽略路径：`logs/research/bcsr/preflight_cpu.json`、
`logs/research/bcsr/preflight_amp.json`、`logs/research/bcsr/ddp_cpu.json`，以及
`logs/moda/preflight_baseline_amp.json`。这些检查期间工作区尚有待提交代码，相关 dirty
记录不能作为正式训练证据。

## 尚不能据此宣称

没有 MODA 完整训练结果；没有边界路由胜过整框共享路由的证据；没有确认日期代理分割
已排除场景泄漏；没有本项目模型与 FressDet/OSSDet 的实际性能比较。历史 O² 验收配方
与当前仓库 `7b5f727` 的三项调参差异保持可见，没有为了通过测试修改科学验收标准。

## FressDet 论文设置配方验证

新增 [20 轮 benchmark 配方](../../datasets/moda/fressdet_alignment.md)，主输入为
宽 1216 × 高 928、从头训练、无增强、全局 batch8。当前三种 BCSR 消融均为
19,635,146 参数；与基线在相同 seed 下的所有共享初始参数完全一致。

- ProbIoU、阈值匹配、AP 插值积分、fast NMS 对照本地 FressDet 函数通过；覆盖其
  AP=0.995 边界和已被抑制候选仍可抑制其他候选的行为。
- 矩形 isotropic 坐标在 transform、ADR/angle decoder、旋转 attention、BCSR、
  postprocessor 与诊断恢复中一致；物理采样与等价框表示测试通过。
- 真实 16 图 debug 清单、宽96 × 高64、20 queries、FP32，用实际训练入口完成
  两轮、16 次更新，每轮完整评估该小清单；EMA、逐层 AP、最佳 AP50 checkpoint、
  最后 checkpoint、原图坐标推理导出均通过。warmup 仅在此 smoke 缩为 12 次迭代，
  以覆盖跨 epoch 与预热结束；所有运行覆盖保存在 runtime.yml。
- AdamW 以所属模块类型识别归一化参数，含 Sequential 内匿名 LayerNorm；bias
  和归一化权重免衰减。跨 epoch 预热保持当次迭代的线性插值。

最新链路证据位于 `logs/research/bcsr/paper_protocol_trainer_cpu_smoke_v2/`，
早期的主配方 preflight 位于 `logs/research/bcsr/paper_protocol_preflight_cpu.json`。
这是代码可运行性证据；未测量新配方全分辨率显存或双 3090 吞吐，不据此推算正式 AP。

## 配置清理与设计复审

删除 MODA/BCSR 的 10 个旧配置文件，将保留的 O²、BCSR edge/object/initial 直接
连接到 20 轮主配方。清理前后四份完整解析配置逐项相同，包括输入通道、优化器、
诊断预算及数据加载设置。direct-angle 同步提供该主配方下的独立入口。
清理记录位于 `logs/research/bcsr/cleanup_review/config_equivalence.json`。

新 [设计复审](design_review.md) 区分 v1 原型已有性质与 v2 候选。局部几何 probe
验证边上平均的角度信息抵消反例、保留一阶矩恢复秩、GLS 方差以及相关波段重复计数
的反例。结果在 `logs/research/bcsr/cleanup_review/theory_probe.json`；所有噪声和
方差数据为显式构造，不是 MODA 实测。没有修改现有 detector 算子，也没有获得
新模型精度结果。
