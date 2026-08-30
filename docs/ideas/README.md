# 活动 Idea 文档边界

本目录在稳定 `main` 中只保留本说明。每个活动 idea 在自己的分支中创建
`docs/ideas/<idea_id>/`，至少包含命题卡和 `manifest.json`；进入清退阶段时再添加
`experience.md`。

`candidate` 阶段只允许上述文档目录。进入 `feasibility` 后，按首个忠实实验的实际
需要创建审计工具、测试、隔离模型和配置；冻结审计不是强制步骤，训练也不必等到
`prototype`。

这些阶段可能使用的隔离路径分别为：

```text
engine/rtv4/obb/incubator/<idea_id>/
configs/incubator/<idea_id>/
tools/research/<idea_id>/
test/research/<idea_id>/
```

详细状态机、所有权和删除保护见
[Idea 生命周期](../development/IDEA_LIFECYCLE.md)与
[Worktree 契约](../development/WORKTREE_CONTRACT.md)。
