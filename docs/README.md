# 项目文档地图

本目录只保存能够随代码版本复现的稳定文档。本地研究经验、外部论文源码和训练日志
分别属于 `research/` 与 `logs/`，不混入这里。

## 稳定框架

- [框架文档入口](framework/README.md)
- [OBB 坐标与几何约定](framework/obb_conventions.md)
- [新增数据集、模型、损失和模块的集成契约](framework/INTEGRATION_CONTRACT.md)

## 稳定方法

- [O²-DFINE 复现说明](methods/o2/README.md)
- [O² 实现审计](methods/o2/implementation_audit.md)
- [O² 诊断与验收协议](methods/o2/diagnostic_protocol.md)

## 数据集与测试

- [CODrone](datasets/codrone/README.md)
- [UAV-ROD](datasets/uav_rod/README.md)
- [稳定 OBB 底座验证状态](testing/obb_baseline_report.md)

## 开发与研究管理

- [开发文档入口](development/README.md)
- [多 worktree 协作契约](development/WORKTREE_CONTRACT.md)
- [Idea 生命周期与清退规则](development/IDEA_LIFECYCLE.md)
- [活动 idea 文档边界](ideas/README.md)

文档不在项目根目录散落：通用规则放 `framework/`，稳定方法放 `methods/`，数据协议
放 `datasets/`，开发流程放 `development/`。活动 idea 只存在于自己的分支和
`docs/ideas/<idea_id>/`，失败后随 worktree 清退。
