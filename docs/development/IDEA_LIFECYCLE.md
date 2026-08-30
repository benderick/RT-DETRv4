# 研究想法生命周期

本文件用于保护探索，而不是尽早淘汰探索。它同时解决两个问题：让高风险 idea 获得与
其机制相匹配的验证机会；让多个 idea 在失败、暂停或成功后仍可追溯、可清理，并且不
污染 direct-angle 与 O² 两条稳定链路。

核心原则是：**选择成本最低的忠实实验，而不是一律选择成本最低的实验。** 冻结
checkpoint、合成实验和短训练都只是证据手段；是否训练以及训练多少，由科学命题决定。

## 1. 哪些约束是硬的

下面的约束保护代码、数据和结论，任何 idea 都必须遵守：

- 一个 idea 使用一个 `idea/<idea_id>` 分支和独立 worktree；
- `main` 中的 direct-angle/O² 默认 forward、配置和 checkpoint 契约不能被暗改；
- 数据、配置、Git 提交、checkpoint、随机种子、耗时和显存必须由代码记录；
- 正式 OBB 指标回到原图坐标计算，不能用 tile AP 或挑选样本替代；
- 探索性结果与确认性结果明确区分，修改 protocol 必须留下版本和原因；
- 清退只能作用于 manifest 精确声明的私有目录和日志，不能递归清理共享资产。

下面的事项不设统一硬规则：是否使用冻结模型、是否从零训练、训练轮数、样本规模、
代理指标、数值 gate 和需要尝试的实现数量。它们必须由命题的忠实性和测量噪声决定。

## 2. 状态不是单向淘汰赛

| 状态 | 含义 | 允许的证据与代码 |
|---|---|---|
| `candidate` | 问题与假设仍在形成 | 命题卡、来源、未知项和首个忠实实验设计 |
| `feasibility` | 判断机会是否存在、机制是否可实现 | 冻结审计、数学/合成实验、oracle、micro-train 或短程共同训练 |
| `pilot` | 小规模端到端验证 | 隔离实现、少量合理变体、训练与机制激活诊断 |
| `prototype` | 完整主数据集方法验证 | 完整 recipe、配对基线、必要消融、效率与稳定性 |
| `promoted` | 已满足并入稳定框架的证据 | 整理后的正式实现、配置、测试和方法文档 |
| `parked` | 因外部依赖、资源或时机暂停 | 保留当前证据，写明恢复条件；不等于失败 |
| `inconclusive` | 当前实验不能回答原命题 | 写明代理失真、统计不足或实现未激活；可以清退，也可以修订后继续 |
| `rejected` | 明确作用域内的主张已被反证 | 写明被否定的 claim、证据和适用边界 |
| `retired` | 活跃实现已安全离场 | 本地注册表、经验卡和少量裁决证据 |

典型路径是：

```text
candidate -> feasibility -> pilot -> prototype -> promoted
                 ^           |          |
                 +-----------+----------+  同一主张下修订实现或实验

任何活动状态 <-> parked
feasibility / pilot / prototype -> inconclusive 或 rejected -> retired
```

这不是强制逐级流水线。若低成本代理不能忠实表达机制，可以跳过它并说明原因；若一个
idea 必须完成较长的共同训练才可能出现目标能力，feasibility 可以包含一次完整训练，
但该训练只回答预先声明的可行性问题，不自动升级为论文结论。

同一核心科学主张下更换实现、修正数值错误或换用更忠实的 probe，继续使用同一 idea
id，并在命题卡中记录 attempt。只有核心问题或因果主张发生实质变化时才创建新 id。

## 3. Candidate：允许未知，不要求先讲完论文

candidate 至少回答以下问题：

1. **研究对象与重要性**：观察到什么现象、限制或机会，为什么值得研究？
2. **暂定主张**：当前认为哪个因果关系可能成立？也可以从尚无解释的反常现象出发。
3. **已知与未知**：哪些事实已有证据，哪些只是猜测，最大反例是什么？
4. **相邻工作**：先定位最近的直接冲突和明显重复；完整新颖性审计可以随命题收敛继续。
5. **首个忠实实验**：它实际能回答什么、不能回答什么？什么证据会改变下一步？
6. **本质测量**：需要记录哪些对象级、机制级或系统级信息，才能画出触及问题本身的图？

跨任务一般性不是进入研究的门槛。命题可以是数据集工程、领域方法或一般学习原理，
但必须诚实标注作用域，不能把 CODrone 专用技巧包装成普适理论。一般性可以在研究中
被发现，而不是在第一次写命题卡时强行编造。

candidate 也不要求表示、特征、监督和推理已经全部闭环。它需要一条最小因果链，并把
尚未闭合之处列为未知项。这样既避免堆补丁，也不会因为设计仍在形成就阻止实验。

candidate 只拥有 `docs/ideas/<idea>/`。此时不创建模型和训练配置；通过讨论确认首个
忠实实验后进入 `feasibility`。

## 4. Feasibility：按命题选择忠实 probe

feasibility 的问题不是“新方法最终能否提高 mAP”，而是“机会是否存在、关键机制是否
能被实现和学习”。可选择一种或多种 probe：

| Probe | 适用问题 | 不能据此推出 |
|---|---|---|
| 冻结 checkpoint 审计 | 基线是否已有某种信息、偏差或失败模式 | 新机制共同训练后仍无能力 |
| 数学、合成与反例 | 几何一致性、可辨识性、上下界 | 真实数据上的可学习性与最终精度 |
| oracle / 系统仿真 | 是否存在可利用的机会空间、理论效率上界 | 学习器一定能达到 oracle |
| micro-overfit | 梯度、容量、监督和实现是否能工作 | 泛化性能 |
| 小数据或短程训练 | 学习信号、早期动力学和模块共同适应 | 完整训练后的最终排名 |
| 完整共同训练 | 只有充分适应后才会出现的能力 | 未做配对与重复实验时的普适结论 |

冻结 checkpoint 不是默认优先级。如果命题改变表示、路由、注意力、匹配、损失或优化
轨迹，冻结模型通常只能验证一个代理命题；代理失败不得写成科学命题失败。

进入确认性 feasibility 实验前，应冻结：

- 本次测试的 claim 与作用域；
- intervention、对照和必须记录的机制激活证据；
- probe 能支持和不能支持的结论；
- 数据范围、训练范围和退出条件；
- 结果不确定时如何进入 `inconclusive`，而不是强行二分。

可以先做探索性校准来了解指标尺度和噪声，再冻结确认性 gate。禁止看完正式结果后
静默移动阈值；但也不要求在不知道方差和量纲时凭空写一个数值门槛。

feasibility 可以按需要拥有 `tools/research/<idea>/`、`test/research/<idea>/`，以及
隔离的 `engine/rtv4/obb/incubator/<idea>/`、`configs/incubator/<idea>/`。是否创建这些
目录由 manifest 的 `owned_paths` 明确声明，不要求为了状态完整而制造空目录。

## 5. Pilot 与 Prototype

### Pilot

pilot 是首次小规模端到端方法验证，目标是确认三件事：机制实际被调用、机制指标按预期
改变、任务指标或效率指标没有出现方向性矛盾。允许比较少量有科学理由的实现变体，
因为第一个实现失败不等于原命题失败；每个 attempt 必须说明它测试的是哪条因果链。

pilot 不要求达到论文最终指标，也不能把 micro-overfit 或短训练的最佳 epoch 当作正式
增益。若机制没有激活，结论是 implementation failure；先修复或进入 `inconclusive`，
不能据此否定主张。

### Prototype

prototype 使用完整主数据集和合理训练 recipe，至少包含公平的稳定基线、端到端任务
指标、命题对应的机制指标、效率与资源测量。需要多少 seed、消融和训练长度，由观测
方差、效果大小与论文主张决定，不使用全项目统一数量。

机制指标未变化但任务指标提高时，可以记录为有价值的工程结果，但它不能单独支撑原先
的机制故事；此时应修改 claim、补充解释或将结论限定为经验方法，而不是删除结果。

## 6. Verdict 必须有作用域

任何 STOP 或晋级决定都必须区分下面的对象：

| 结论对象 | 示例 | 对 idea 的影响 |
|---|---|---|
| `scientific_claim` | 因果主张被忠实实验或反例否定 | 可以进入 `rejected` |
| `opportunity` | oracle 上界或信息论下界表明几乎无空间 | 可以进入 `rejected` |
| `implementation` | 某个 head、router 或 loss 不工作 | 只否定该 attempt |
| `probe` | 冻结代理不能表达共同学习机制 | 进入 `inconclusive` 或换 probe |
| `training_recipe` | 当前轮数、优化器或尺度没有激活机制 | 修订 recipe，不自动否定 claim |
| `system_constraint` | 延迟/显存下界不符合目标设备 | 限定部署作用域或否定该系统方案 |

只有下列证据足以停止整个方向：

1. 数学矛盾或直接反例否定了主张；
2. 与命题一致的 oracle/上界表明机会不存在；
3. 忠实、已激活的实现经过与噪声相称的确认实验后仍系统性反驳主张；
4. 无法规避的计算或信息下界与目标根本冲突。

单个冻结代理无信号、一次训练没有涨点、一个 adapter 失败、一个相关性不显著或一次
数值崩溃，都不能单独否定整个科学命题。

进入 `rejected` 或 `inconclusive` 时，manifest 至少写明 `verdict` 与
`verdict_scope`。`parked` 写明 `pause_reason` 与 `resume_condition`。这些字段让未来的
研究者知道失败发生在哪一层，而不是只看到一个模糊的 `STOP`。

## 7. 可视化和日志

可视化必须接近研究问题，但探索阶段允许发现新的画法。确认性实验前冻结的是样本选择
原则、统计口径和禁止 cherry-pick 的规则，不是强行冻结所有图的版式。

每个 idea 至少考虑三类证据：

- **现象图**：基线问题在什么对象、尺度或场景中出现；
- **机制图**：intervention 如何改变内部证据、计算或学习轨迹；
- **结果/系统图**：任务质量、延迟、显存、处理像素/token 等 Pareto 关系。

训练和分析所需信息必须由代码写入 `logs/research/<idea>/`。用户负责启动实验，不负责
手工整理角度误差、中心误差、逐层框、显存或时间等分析数据。

## 8. 资源使用不是科学 gate

资源规划用于选择实验顺序，不用于替代科学判断。研究者应根据已有 baseline 日志估计
一次 probe 的时间、显存和输出规模，不要求用户先给出一个抽象“预算”。

通常先尝试便宜且忠实的证据；如果便宜实验不忠实，就跳过它。任何扩大训练的决定都应
说明新增算力将解决哪个尚未回答的问题。一个 idea 不能因为“已经花过算力”自动继续，
也不能因为“训练昂贵”在从未接受忠实测试时被宣判失败。

## 9. 暂停、清退与重启

`parked` 用于外部代码未就绪、数据缺失、设备不可用或优先级暂时下降。它必须写明恢复
条件，不进入清退流程，也不被记成负结果。

`rejected` 和 `inconclusive` 都可以清退。清退前在分支的
`docs/ideas/<idea>/experience.md` 写经验卡，并追加到主 worktree 的本地账本：

```bash
python tools/research/manage_ideas.py record-experience <idea>
python tools/research/manage_ideas.py retire <idea>
python tools/research/manage_ideas.py retire <idea> --apply
```

`retire` 不带 `--apply` 永远只是预演。工具只删除 manifest 精确声明的私有路径；只有
额外传入 `--prune-artifacts` 才删除明确列入 `discard_artifacts_on_retire` 的大型产物。
checkpoint 不属于自动清理范围。

清退后保留：

- `research/ledger/registry.json` 中的 claim、verdict、verdict scope、原状态和证据；
- `research/ledger/EXPERIENCE_LOG.md` 中的人可读经验卡；
- protocol、report 和少量能够支持裁决的关键图。

清退前，同一主张可以在原 idea 中继续新的 attempt。清退后若因新证据重启，创建带有
明确关联的新 id 并链接旧经验；这是为了避免活动 manifest 与已退役记录重名，不代表
旧主张被永久封禁。不得通过改名掩盖同一实验，也不得让旧 verdict 无法追溯。

## 10. 分支 manifest 与本地账本

活动 idea 的机器可读真源是分支中的 `docs/ideas/<idea>/manifest.json`。退役记录的真源
是主 worktree 被 Git 忽略的 `research/ledger/registry.json`，人可读经验位于同目录的
`EXPERIENCE_LOG.md`。所有 worktree 由管理工具定位并共享同一个账本。

常用检查：

```bash
python tools/research/manage_ideas.py init-ledger
python tools/research/manage_ideas.py ledger-path
python tools/research/manage_ideas.py list
python tools/research/manage_ideas.py validate
```

本地账本不会随 clone、pull 或 push 迁移，必须单独备份。管理工具使用文件锁、原子
替换和 `.bak` 防止多个 worktree 同时更新时互相覆盖。
