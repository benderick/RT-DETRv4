# 研究想法生命周期

本文件约束研究 idea 从提出到晋级或清退的全过程。目的不是减少探索，而是让大量
探索保持低成本、可证伪、可清理，并且不污染 direct-angle 与 O² 两条稳定链路。

## 1. 六种状态

| 状态 | 允许内容 | 进入下一状态的必要条件 |
|---|---|---|
| `candidate` | 一页命题卡；禁止训练代码 | 完成新颖性检索、机制闭环、最低成本证伪协议 |
| `stage0` | 冻结 checkpoint 的只读审计、纯数学实验 | 预注册 gate 在完整样本上通过 |
| `prototype` | 隔离的新模块和训练配置 | 至少一个主数据集有效，且机制诊断与指标方向一致 |
| `promoted` | 准备并入框架的候选方法 | 多数据集、消融、效率和稳定回归全部通过 |
| `rejected` | 只等待清退 | 已写明证据、否决原因和可复用经验 |
| `retired` | 注册表、经验卡和小型证据 | 研究实现和专用测试已经离开活跃源码树 |

合法的单向迁移是：

```text
candidate -> stage0 -> prototype -> promoted
    |           |           |
    +-----------+-----------+-> rejected -> retired
```

被清退的命题如果以后因新证据需要重启，必须创建新的 idea id，并链接旧经验卡；不得
把旧目录原地复活，从而避免新旧假设和结果混淆。

## 2. 写代码前的 candidate gate

每个候选命题必须先回答六个问题：

1. **科学对象是什么**：不能只写“增加一个 loss/module”；必须指出现有学习范式中的
   哪个假设可能错误。
2. **比 OBB 更一般在哪里**：说明它是否适用于 HBB、姿态、分割或一般集合预测；若只
   是遥感技巧，不能按 ICLR 主命题推进。
3. **现有工作边界在哪里**：至少检索表示、等变性、DETR query/匹配、概率建模和最相邻
   的最新工作，记录直接冲突的论文。
4. **机制如何闭环**：同一个抽象必须解释表示、特征获取、训练监督和推理输出；不允许
   四个彼此无关的补丁拼成方法。
5. **什么图能直接触及问题**：在训练前定义可视化对象和选择规则，禁止只展示“新框
   更接近 GT”。
6. **最低成本怎样否证**：优先使用冻结 checkpoint、完整验证集审计或合成可控实验，
   明确数值 gate 和 STOP 条件。

只有六项齐全才允许建立 `tools/research/<idea>/`。在此之前，候选只存在于
idea 分支的命题卡和 `manifest.json`，不创建模型、配置或测试目录。

## 3. 隔离边界

每个活动 idea 必须在独立 `idea/<idea_id>` 分支和 worktree 中进行，具体 Git
契约与初次使用方法见 `docs/framework/WORKTREE_CONTRACT.md`。

Stage 0 的所有权目录仅允许为：

```text
docs/research/<idea>/
tools/research/<idea>/
test/research/<idea>/
```

需要训练的 prototype 才可额外拥有：

```text
engine/rtv4/obb/incubator/<idea>/
configs/incubator/<idea>/
```

未经 `promoted`，不得新增稳定 `refinement_mode`，不得改变 direct-angle/O² 默认
forward，不得向通用 evaluator、dataset adapter 或 postprocessor 塞入私有逻辑。

## 4. 证据层级与算力闸门

探索按下面顺序花费资源，前一层失败就停止：

1. **纸面审查**：新颖性、数学一致性、反例和实现复杂度；不使用 GPU。
2. **Stage 0**：冻结模型或合成实验；原则上不训练，输出预注册 verdict。
3. **micro overfit**：极少样本验证梯度、可学习性和机制图；不得报告正式增益。
4. **单数据集 prototype**：只允许一组主配置和预注册的最少消融。
5. **论文实验**：只有 prototype 同时通过效果、机制和稳定性 gate 后才展开。

“结果略有提高”不是自动晋级条件。主指标必须和命题对应的机制指标同步改善；如果
机制指标不变，增益只能作为偶然工程结果，不能支撑原故事。

## 5. 清退与证据保留

idea 进入 `rejected` 后必须先在 `EXPERIENCE_LOG.md` 写经验卡，再运行：

```bash
python tools/research/manage_ideas.py retire <idea>
python tools/research/manage_ideas.py retire <idea> --apply
```

第一条命令永远只是预演。第二条只删除 manifest 中声明、且名称严格等于
idea id 的隔离目录；`o2` 是保留 id，工具会拒绝操作。清退工具会把分支
manifest 转成中央退役记录。清退后仍保留：

- `registry.json` 中的命题、verdict、证据路径和清退日期；
- `EXPERIENCE_LOG.md` 中的人可读经验卡；
- `report.json`、协议和少量关键图等小型裁决证据。

逐对象记录、特征张量等大型证据不随源码自动删除。注册表可声明
`discard_artifacts_on_retire`，但只有额外传入 `--prune-artifacts` 才会清除这些精确
文件；也可在退役后再次运行下面的独立预演/清理。训练 checkpoint 永远不属于该自动
清理范围。

```bash
python tools/research/manage_ideas.py prune-artifacts <idea>
python tools/research/manage_ideas.py prune-artifacts <idea> --apply
```

## 6. 分支 manifest 与中央退役表

活动 idea 的机器可读真源是它自己分支中的
`docs/research/<idea>/manifest.json`。`docs/research/registry.json` 只保存已退役
记录，因此多个 worktree 不会同时争写一个中央 JSON。

`manage_ideas.py` 会合并当前 worktree 可见的活动 manifest 和中央退役表，
并拒绝重复 id、未登记目录、越界所有权、缺失经验卡以及退役后仍残留的源码。

常用检查：

```bash
python tools/research/manage_ideas.py list
python tools/research/manage_ideas.py validate
```
