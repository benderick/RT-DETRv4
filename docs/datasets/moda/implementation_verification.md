# 实现与观测链路验证

2026-09-14，环境 `wyq-deim`，未安装新包。当前实验统一从头训练：原输出目录、top-300、无 NMS、主 ProbIoU AP、固定训练子集机制图。

| 检查 | 结果 | 记录 |
|---|---|---|
| 33 项单元／集成测试 | 通过；含模型、观测、评估、训练配方和图像归档 | [测试日志](../../../logs/moda/implementation/observability_regression.log) |
| 真实 MODA 自动源图采集 | 固定规则选出的 10 张图、8 类图例，输出 PNG／NPZ／JSON／HTML | [未训练检查说明](../../../logs/moda/implementation/source_real_untrained_smoke/README.md)、[日志](../../../logs/moda/implementation/source_real_smoke.log) |
| 无 NMS 启动入口 | 原输出目录，FP32，500 普通 query、top-300，主 AP；dry-run 不启动训练 | [运行指南](RUN_EXPERIMENTS.md) |

源图检查用随机初始化完整模型、64×96 调试画布和 20 个普通 query，验证自动采集及渲染。它保留未关联对象，不能作为检测或机制有效性证据。

新增检查验证：关闭额外几何 AP 不改变主指标；归档查询可精确重放无 NMS 后处理；轻量评估跳过全量逐候选记录；固定子集采集不改变模型参数、训练状态和随机数状态；GT 仅在前向后用于对象身份关联；训练诊断 detach 并区分普通查询与 DN。

模型检查覆盖矩形坐标、等价 OBB、无效区域、联合背景统计、低支持回退、零初始化与基线等价、分类和定位反向、消融初始化一致性，以及源图扩展字段的序列化。

复查全部相关模块：

```bash
conda activate wyq-deim
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest test.research.test_spectral_evidence test.research.test_source_evidence test.framework.diagnostics.test_engine_integration test.framework.diagnostics.test_obb_diagnostics test.framework.evaluation.test_benchmark_protocol test.framework.solver.test_paper_recipe test.framework.model.test_model_pipeline -v
```

这些检查不证明 AP 改善。新版全尺寸 GPU 评估耗时、机制采集开销和双 3090 峰值仍由正式运行测量；旧版本约 40 分钟的评估时间不代表新版速度。各阶段计时会保存在新运行的诊断目录。
