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
- Hungarian、NMS 因果归因、query/layer 轨迹、时间/显存/梯度/AMP 日志；
- 四个公开配置和两个 `refinement_mode` 的集成契约。

最新实测结果在每次清理或公共接口变更后更新；CUDA/真实数据不可用造成的 skip 必须
保留明确原因，不能记作已通过。

## 当前结果

完整入口实测：**114 tests run，111 passed，3 skipped，0 failures**。三个 skip 分别是
CODrone 原图、CODrone tile 和 UAV-ROD 的 O² 1024 CUDA smoke；当前会话无法初始化
CUDA/NVML。现有 `logs/dfine_obb_angle/checkpoint0029.pth` 对 direct-angle 模型
strict load 为零 missing、零 unexpected。O² 等待正确实现从头训练，不使用旧权重。
