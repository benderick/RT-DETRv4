# 开发与研究管理

- [Worktree 协作契约](WORKTREE_CONTRACT.md)：分支、目录所有权、共享资产和首次使用。
- [Idea 生命周期](IDEA_LIFECYCLE.md)：candidate、Stage 0、prototype、晋级与清退。
- [框架集成契约](../framework/INTEGRATION_CONTRACT.md)：新增稳定能力的目录和测试门槛。

稳定代码从 `main` 开发；每个研究 idea 使用独立 `idea/<idea_id>` 分支和 worktree。
失败经验写入主 worktree 被忽略的 `research/ledger/`，不会产生 `main` 提交。
