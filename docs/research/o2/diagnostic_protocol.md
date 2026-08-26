# O² 逐层精炼与论文证据协议

## 要回答的科学问题

这套协议不以“模型 loss 在下降”替代机制验收，而直接回答五个问题：

1. 第一层 traditional OBB head 产生的 pre-box 是否是 ADR 的固定几何锚？
2. 六组 distribution logits 是否逐层累计，`P(n)` 是否真的改变，积分后的框是否改变？
3. 上一层预测框是否成为下一层 query position 和 rotated cross-attention 的输入 reference？
4. 精炼对完整验证集 AP 和同一 matched query 的 rIoU 是改善、无效还是过修正？
5. 固定 weighting function 与 LQE MLP 分别做了什么，二者是否被错误混同？

## 三条不能混淆的路径

```text
几何路径:
cumulative logits -> P(n) -> A(n)P(n) -> sum A(n)P(n)
                  -> 4 external boundaries + epsilon + eta -> OBB

置信度路径:
P(n) top-k statistics -> LQE MLP -> class-logit correction

注意力路径:
encoder reference -> decoder 0
decoder 0 OBB     -> decoder 1
decoder 1 OBB     -> decoder 2 -> ...
```

`A(n)` 是固定解析码本，不是 MLP；LQE 不参与几何 residual 的积分。

## 训练时落盘的证据

`diagnostics/eval/epoch_XXXX/refinement_stages.json` 保存完整验证集上的：

```text
pre_box -> decoder_0 -> decoder_1 -> ... -> decoder_L
```

每个 stage 使用与正式推理相同的 postprocessor 和 evaluator。文件中的
`final_stage_matches_primary_evaluator` 必须为 `true`，否则逐层指标不具备可比性。

`queries.rankNNN.jsonl.gz` 对确定的验证图像子集保存：

- GT、pre-box、各 decoder 层预测框与同 query 的误差；
- 本层 `input_reference_box` 和固定 `initial_anchor_box`；
- 六组累计 logits、`P(n)`、`A(n)P(n)`、期望、熵、方差和 peak bin；
- ADR residual、raw six values、闭合前 orthogonality error；
- LQE 前后分数与 target-class logit correction；
- 旋转前后 sampling offsets、sampling locations 与 attention weights。

对象跨 epoch 以 `(image_id, gt_index)` 固定，而不是强行固定 query index。DETR 的
Hungarian assignment 可以让同一 GT 在训练过程中换 query；同一 epoch 内的逐层图则
固定最终 Hungarian 匹配的同一个 query，二者回答的是不同问题。

## 自动选例与图

运行：

```bash
/icislab/volume1/liuxiaolong/anaconda3/envs/wyq-deim/bin/python \
  tools/research/o2/visualize_mechanisms.py \
  logs/uav_rod/dfine_obb_o2 \
  --output logs/uav_rod/dfine_obb_o2/paper_evidence
```

默认按数值规则依次选择：跨 epoch rIoU 提升最大、最新 pre-to-final 提升最大、
过修正最大（若没有负值则为提升最小）、最终 rIoU 中位样本。程序不读取图像外观来
决定样本，并把选择规则和数值写入 `evidence_index.json`。

输出包括 PNG 与适合论文排版的 PDF：

- 固定 GT 随 epoch 的最终预测框；
- 同一 query 的 pre-box 到各 decoder 层预测框；
- 外接 HBox、epsilon、eta 与最终 OBB 的几何图；
- 六分量上下两排 `P(n)` / `A(n)P(n)` 图；
- 全验证集逐 stage AP 随 epoch 曲线；
- rotated cross-attention、CDC matching cost 和 OCD instability 图。

## 验收解释

单个对象不要求逐层单调变好；如果只展示改善样本会构成 cherry-picking。最低验收为：

- 最后一层逐 stage 指标与正式 evaluator 完全一致；
- `input_reference_box[k] == predicted_box[k-1]`，且所有 ADR stage 使用同一个 pre-box anchor；
- 累计 logits 和积分 residual 能数值重建每层最终框；
- 同时报告改善率、退化率、均值/中位数变化和完整验证集 AP，而非只给好看的案例。

UAV-ROD 只有 car 一类，适合低成本验收 ADR、reference 传递、旋转采样和 LQE；它不能
单独证明 CDC 的多类别匹配收益，也不能证明 label-noise 的类别扰动收益。后两者仍需在
多类别数据集上做最少但独立的对照。

## ADR metric-consistency 全验证集审计

常规 `queries.rankNNN.jsonl.gz` 故意只保存少量图像的完整分布与注意力，适合逐对象
画图，却不能用来估计坐标跳变在整个验证集上的效应。冻结 O² 基线后，使用独立工具：

```bash
CUDA_VISIBLE_DEVICES=1 \
/icislab/volume1/liuxiaolong/anaconda3/envs/wyq-deim/bin/python \
  tools/research/o2/run_metric_consistency_audit.py \
  --config configs/experiments/uav_rod/dfine_obb_o2.yml \
  --checkpoint logs/uav_rod/dfine_obb_o2/best_stg1.pth \
  --output logs/uav_rod/dfine_obb_o2/metric_consistency_audit \
  --device cuda --batch-size 4 --workers 0
```

该命令只做一次冻结推理，关闭大体积 attention trace，并对所有最终 Hungarian 匹配
保存紧凑记录。它不修改训练配置、O² decoder、后处理或 checkpoint。输出包括：

- `matched_records.jsonl.gz`：全部匹配目标的逐层几何、ADR target、六分布统计和 LQE；
- `report.json` / `report.md`：覆盖率、完整逐层 AP、预注册效应量与判定；
- `theoretical_chart_transition`：同一矩形跨图像轴时，rIoU 连续而 ε/η target 跳变；
- `empirical_metric_mismatch`：seam/target 分层收益、端点质量和退化风险；
- `case_locality_failure`：按“局部几何矛盾中 pre-to-final rIoU 下降最大”自动选择；
- `case_local_control_success`：相同好 pre-box 条件下的小 residual 数值对照；
- `case_layer0_failure`：完整验证集 Decoder 0 过修正最大案例。

“局部几何矛盾”在运行前固定定义为：pre-box rIoU ≥ 0.8、周期角误差 ≤ 5°，但
`max(|Δε|,|Δη|) ≥ 0.8`；数值对照把最后一项替换为 ≤ 0.2。只有以下五项同时成立，
报告才写 `SUPPORTED`：

1. 矛盾样本至少 30 个且不少于全部匹配的 1%；
2. 对照组减矛盾组的平均精炼收益至少 0.005 rIoU；
3. 矛盾组减对照组的退化率至少 10 个百分点；
4. 最终 ε/η 端点概率质量至少为四边界的 2 倍；
5. 最终 ε/η 分布方差至少为四边界的 4 倍。

这些阈值属于审计协议，不能在看到新模型结果后调整。UAV-ROD 上 `SUPPORTED` 只允许
得出“该数据集的冻结 O² 存在 ADR 局部度量不一致”的结论，不能替代跨数据集验证。
