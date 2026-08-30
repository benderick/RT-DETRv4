# 活动 Idea 文档边界

本目录在稳定 `main` 中只保留本说明。每个活动 idea 在自己的分支中创建
`docs/ideas/<idea_id>/`，至少包含命题卡和 `manifest.json`；进入清退阶段时再添加
`experience.md`。

`candidate` 阶段只允许上述文档目录。通过 candidate gate、进入 Stage 0 后，才创建
审计工具和对应测试；需要训练的 `prototype` 才创建模型与配置目录。

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
