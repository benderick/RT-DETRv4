# OBB 框架验证状态

本文只记录当前受支持的 direct-angle、O² ADR 和共同 OBB 底座。

统一验证命令：

```bash
/icislab/volume1/liuxiaolong/anaconda3/envs/wyq-deim/bin/python test/run_all.py
```

验证范围包括：

- CODrone 原图/tiny/tile 读取、空标注、旋转/翻转/resize-pad 增强；
- DOTA OBB perfect prediction、Task1 导出、可视化和通用 evaluator；
- tile IOF、padding、回源图、完整性检查、rotated NMS 和原图 AP；
- direct-angle 与 O² 的 forward、loss、backward、inference 和 tiny convergence；
- O² ADR、OCD、旋转 attention、KLD/Chamfer 语义；
- O²-DFINE-M 结构字段和 19–20M 实际参数量契约；
- Hungarian、NMS 因果归因、query/layer 轨迹、时间/显存/梯度/AMP 日志；
- 四个公开配置和两个 `refinement_mode` 的集成契约。
- 文档分层、稳定方法路径、本地 research ledger 边界和 Markdown 本地链接。

最新实测结果在每次清理或公共接口变更后更新；CUDA/真实数据不可用造成的 skip 必须
保留明确原因，不能记作已通过。

## 当前结果

2026-08-30 完整入口实测：**139 tests run，136 passed，3 skipped，0 failures**。
三个 skip 分别是
CODrone 原图、CODrone tile 和 UAV-ROD 的 O² 1024 CUDA smoke；当前会话无法初始化
CUDA/NVML。随后用户 GPU 环境已经用相同代码完成 UAV-ROD 1024、AMP、72 轮从零
训练与逐轮完整验证，这一运行覆盖了 O² 真图 forward、loss、backward、optimizer、
EMA、checkpoint 和 inference，且无训练崩溃。CPU suite 的三个 skip 仍按原始执行事实
保留，不伪记为同一次测试已通过。随后 `best_stg1.pth` 在 GPU 上完成独立冻结验收：
来源配置、strict load、427 张完整图、固定 ADR anchor、层间 reference、六值 OBB
重建、LQE、model/pixel rIoU 一致性和逐层 AP 的全部必需 gate 均通过。O² 当前状态为
**PASS**；机器可读报告为 `logs/uav_rod/dfine_obb_o2/acceptance/acceptance.json`。
