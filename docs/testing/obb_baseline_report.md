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

## 2026-08-24 清理后结果

完整入口实测：**98 tests run，96 passed，2 skipped，0 failures**。两个 skip 分别是
O² 原图 1024 CUDA smoke 和 O² tile 1024 CUDA smoke；当前会话无法初始化 CUDA/NVML。
此外，`logs/dfine_obb_angle/checkpoint0029.pth` 与
`logs/dfine_obb_o2/checkpoint0029.pth` 的 model/EMA state dict 均已严格加载成功。
