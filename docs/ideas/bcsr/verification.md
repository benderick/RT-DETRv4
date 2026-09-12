# BCSR 代码验证记录

日期：2026-09-12。环境 `wyq-deim`，Python 3.10.16、PyTorch 2.3.0+cu121。
未安装新依赖，未进行完整数据训练，未获得正式 AP 或论文增益。

## 已完成

- 项目统一 CPU 测试：161 项，158 通过，3 项因 CUDA 在沙箱中不可见而跳过。
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
已排除场景泄漏；没有与 FressDet/OSSDet 统一 evaluator 后的比较。历史 O² 验收配方
与当前仓库 `7b5f727` 的三项调参差异保持可见，没有为了通过测试修改科学验收标准。
