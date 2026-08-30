# Git worktree 研究协作契约

本契约用于同时推进多个研究 idea，并保证 direct-angle、O² 和已完成实验不被
其他分支的代码或日志污染。本文同时是第一次使用 `git worktree` 的操作手册。

## 1. 先理解 worktree

一个 Git 仓库可以同时拥有多个工作目录。它们：

- 共享同一套 Git 历史和对象，不会重复复制整个 `.git`；
- 各自拥有独立的文件、HEAD、索引和未提交修改；
- 一个分支同一时间只能被一个 worktree 检出；
- 不共享 Git 忽略的 `logs/`、`pretrain/` 等大文件目录；唯一例外是管理工具会让
  所有 worktree 共同使用主 worktree 的 `logs/research_ledger/` 本地账本。

本项目约定把 worktree 放在主仓库旁边：

```text
/icislab/volume1/liuxiaolong/futurama/
├── RT-DETRv4/                 main：稳定框架
├── RT-DETRv4-orbit_dn/        idea/orbit_dn
├── RT-DETRv4-<idea_id>/       其他 idea
└── data/                      共享数据集
```

这个平级布局会让配置中的 `../data/...` 在所有 worktree 里都指向同一数据集。

## 2. 不可破坏的规则

1. `main` 只保存稳定 OBB 框架、direct-angle、O²、通用诊断和研究管理设施；
   活动 idea 的模型、配置和专用测试不进入 `main`。
2. 每批受控实验在 manifest 中锁定不可移动的科学/模型 tag，当前 O² 基线为
   `obb-o2-baseline-v1`。代码分支从包含最新管理设施的干净 `main` 创建，但必须证明
   该 tag 是分支祖先；禁止重写或强制移动科学基线 tag。
3. 一个 idea 对应一个 `idea/<idea_id>` 分支和一个同名 worktree。
4. 训练进程运行时，不在它的 worktree 内切换分支、rebase 或修改模型源码。
5. 每次正式训练前 `git status --short` 必须为空。诊断 manifest 中
   `git.dirty` 必须为 `false`。
6. 不同 worktree 禁止写入同一个输出目录；不得把整个 `logs/` 链接到
   `main`。

## 3. 分支和目录所有权

| 类型 | 命名 | 用途 |
|---|---|---|
| 稳定主干 | `main` | 可以随时运行的 OBB 框架 |
| 框架能力 | `framework/<name>` | 方法无关、默认零影响的通用能力 |
| 研究 idea | `idea/<idea_id>` | 单一科学命题和它的证伪实验 |
| 组合实验 | `combo/<a>+<b>` | 只用于预注册的方法组合，不代替单项消融 |
| 晋级整理 | `promote/<idea_id>` | 从 `main` 重新整理已通过的实现 |

每个活动 idea 只能拥有：

```text
docs/research/<idea_id>/
tools/research/<idea_id>/
test/research/<idea_id>/
engine/rtv4/obb/incubator/<idea_id>/   # 只有 prototype 允许
configs/incubator/<idea_id>/           # 只有 prototype 允许
logs/research/<idea_id>/               # 本地证据，不进 Git
```

`docs/research/<idea_id>/manifest.json` 是该分支的活动 idea 真源，至少记录
`id/status/base_tag/base_commit/branch/owned_paths`。中央退役记录和经验卡位于主
worktree 被 Git 忽略的 `logs/research_ledger/`；所有 worktree 由管理工具自动发现
同一个账本，因此失败清退不会修改或产生 `main` 提交。
其中 `base_commit` 是 `base_tag` 解析到的精确科学/模型基线；分支可以额外继承
不改变模型的管理工具提交，正式训练仍会另行记录实际 HEAD。

## 4. 公共文件修改契约

idea 如果必须修改 decoder、criterion、evaluator 或诊断引擎，提交必须拆成：

```text
commit A: 通用扩展接口 + 稳定路径等价测试
commit B: idea 私有实现 + 配置 + 专用测试
```

`commit A` 必须同时满足：

- 不导入任何 idea 包；
- direct-angle/O² 默认配置和参数树不变；
- 默认 forward/inference 数值等价；
- 通过全项目稳定测试。

如果两个 idea 都需要同一公共能力，将 `commit A` 提升到
`framework/<name>` 审核；不得在两个 idea 分支复制两份实现。

## 5. 第一次创建 idea worktree

以 `new_idea` 为例。先进入主仓库：

```bash
cd /icislab/volume1/liuxiaolong/futurama/RT-DETRv4
git branch --show-current
git status --short
git worktree list
```

确认当前是 `main` 且状态为空，再创建：

```bash
git worktree add \
  ../RT-DETRv4-new_idea \
  -b idea/new_idea \
  main
```

进入新 worktree：

```bash
cd ../RT-DETRv4-new_idea
git branch --show-current
git status --short
```

共享只读预训练权重。不要把整个 `pretrain` 目录直接做成 symlink，
因为 Git 会把顶层目录链接视为未跟踪文件；先创建被忽略的真实目录，再链接其内的
权重子目录：

```bash
mkdir -p pretrain
ln -s ../../RT-DETRv4/pretrain/hgnetv2 pretrain/hgnetv2
git check-ignore -v pretrain/hgnetv2
```

如果需要读取稳定 O² 日志，只链接这一个基线目录：

```bash
mkdir -p logs/uav_rod
ln -s \
  ../../../RT-DETRv4/logs/uav_rod/dfine_obb_o2 \
  logs/uav_rod/dfine_obb_o2
```

不要链接整个 `logs/`：idea 的 `logs/research/<idea_id>/` 必须真正存在于自己的
worktree，以免多个训练互相覆盖。

然后只创建命题卡和 manifest，其中 `base_tag/base_commit` 仍填写
`obb-o2-baseline-v1` 及其解析提交；通过 candidate gate 后再添加实现目录。这样分支
拥有最新 worktree/ledger 工具，但科学模型基线没有被悄悄移动。

## 6. 日常使用

不需要反复 `git switch`，直接进入对应目录：

```bash
# 稳定框架
cd /icislab/volume1/liuxiaolong/futurama/RT-DETRv4

# Orbit-DN
cd /icislab/volume1/liuxiaolong/futurama/RT-DETRv4-orbit_dn
```

开始工作或训练前固定执行：

```bash
pwd
git branch --show-current
git status --short
```

提交时使用精确路径，避免把 checkpoint 或其他 idea 带入提交：

```bash
git add docs/research/new_idea tools/research/new_idea test/research/new_idea
git status --short
git commit -m "research(new_idea): freeze Stage 0 protocol"
```

查看所有 worktree：

```bash
git worktree list
```

## 7. 基线更新与公平对比

- idea 尚未开始正式实验时，可以在审核后 rebase 到新基线。
- idea 已有正式结果时，不能原地 rebase 后继续同一实验。新基线必须使用新 tag，
  并重新跑配对 baseline。
- 不把 idea A 直接 merge 进 idea B。需要联合时，从同一 baseline tag 创建
  `combo/a+b`，只 cherry-pick 两者已冻结的最小提交。

## 8. idea 成功或失败后

### 失败

1. 把 manifest 状态改为 `rejected`，填写 verdict 和 `experience_entry`。
2. 在 `docs/research/<idea_id>/experience.md` 写一张以
   `## <experience_entry>：...` 开头的可复用经验卡，然后原子写入本地总账：

```bash
python tools/research/manage_ideas.py record-experience new_idea
```

3. 把 manifest 声明的少量 `preserved_evidence` 复制到主 worktree 对应的
   `logs/...` 路径；checkpoint、逐 step 张量等大产物不复制。
4. 执行清退预演和应用：

```bash
python tools/research/manage_ideas.py retire new_idea
python tools/research/manage_ideas.py retire new_idea --apply
```

`--apply` 只更新被忽略的本地 ledger，不修改 `main`；失败实现不 merge、不提交清退
结果。若确需长期保留失败源码，可以另打 archive tag，否则直接删除分支。

5. 在主仓库确认目标路径后移除 worktree；因为其中通常包含被忽略的日志和清退产生的
   未提交删除，需要显式 `--force`：

```bash
git worktree remove --force ../RT-DETRv4-new_idea
git branch -D idea/new_idea
```

禁止直接 `rm -rf` worktree；`git worktree remove` 会同时正确更新 Git 元数据。

### 成功

不直接 merge 整个实验分支。从最新稳定 `main` 创建 `promote/<idea_id>`，
只 cherry-pick 最终实现、正式配置、测试和分析工具；调试版本和中间产物不进主干。

## 9. 常见问题

### `fatal: '<branch>' is already checked out`

该分支已经属于另一 worktree。用 `git worktree list` 找到对应目录，直接进入，
不要再次检出。

### 新 worktree 找不到预训练权重

Git 不复制被忽略的 `pretrain/`。按第 5 节只读链接主仓库的权重。

### 新 worktree 找不到数据

确认 worktree 与 `RT-DETRv4/` 平级放在 `futurama/` 下。不要放在
`RT-DETRv4/.worktrees/`，否则配置中的 `../data` 将指向错误位置。

### 手工删除了 worktree 目录

只在确认目录确实已不存在后执行：

```bash
git worktree prune
```

`prune` 只清理失效的 worktree 元数据，不是日常删除命令。

### 为什么主干里看不到失败经验

这是有意设计。实时退役表和经验总账位于主 worktree 的
`logs/research_ledger/registry.json` 与 `EXPERIENCE_LOG.md`，受 `.gitignore` 保护，
不会让 `main` 变脏。用下面命令查看实际路径和内容：

```bash
python tools/research/manage_ideas.py ledger-path
python tools/research/manage_ideas.py list
```

本地账本不随 clone/pull/push 迁移，必须纳入机器备份。管理工具拒绝任何 idea 把
`logs/research_ledger/` 声明为可清理 artifact；多个 worktree 的写入由共享文件锁
串行化，每次原子更新前保留 `.bak`。

## 10. 每次训练前的最短检查

```bash
pwd
git branch --show-current
git status --short
git describe --tags --always
```

只有当目录、分支和实验 idea 一致，且 `git status --short` 没有输出时，
才启动正式训练。
