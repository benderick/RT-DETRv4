# O² 失败 run 根因审计（2026-08-21）

## 最终结论

此前把 ADR 局部强制转为 FP32、套用 ai4rs midpoint-offset 的等对角径向投影，
以及在 GradScaler 小于 1 时中止训练，都不是根因修复。新的真实训练在 epoch 0、
step 84 仍发生非有限梯度，证明该结论错误，现已撤回相应代码和文档声明。

作者仓库截至本次审计仍未公开 O²-DFINE/ADR decoder；公开 issue #9 也仍在请求
O²-DFINE/O²-DEIM 支持。因此本项目只能称为 paper-derived reproduction，不能把
别的 coder 当作作者实现：

- https://github.com/wokaikaixinxin/ai4rs/issues/9
- https://github.com/wokaikaixinxin/ai4rs/issues/12

## 真实训练日志给出的故障链

`logs/dfine_obb_o2/diagnostics/train/steps.rank000.jsonl.gz` 显示：

1. step 0 的 scale 从 65536 降为 32768；之后多次跳过 optimizer step；
2. step 30--83 在 scale=1 时连续完成有效更新，loss 从约 43 降到约 33；
3. step 84 的所有已记录前向标量看似有限，但 `loss_kld_dn_2` 恰好为 0；
4. 同一步 backbone/encoder/decoder 分别有约 602 万、458 万、378 万个非有限梯度。

`loss_kld_dn_2=0` 不是正常零损失。criterion 末尾曾对所有 raw loss 调用
`torch.nan_to_num`，把这个 NaN 静默改成了 0，但无效反向链仍污染全网梯度。

## 两个可独立复现的根因

### 1. 把 midpoint-offset coder 错接到 ADR

ai4rs 的 `delta_midpointoffset_rbbox_coder.py` 属于另一种六参数框 coder，最终还会
调用 CPU/OpenCV `minAreaRect`。它不是 O²-DFINE 的可微 ADR 实现。此前借用其
等对角半径投影，会把彼此接近的 top/right 方向继续投到同一圆周上，产生极细的
OBB。

确定性压力测试使用 32,768 个普通 `[-1,1]` ADR 码本残差。旧径向路径在 CPU
FP32 下即可得到 1 个非有限 KLD 和 6 个非有限梯度；200,000 个样本时得到 11 个
非有限 KLD 和 66 个非有限梯度。问题与 AMP 精度无关。

当前解码只执行论文定义的 top/right/bottom/left 顶点构造，再调用框架统一的
ordered-quadrilateral 转 OBB：中心取四点均值，两条相邻边给出宽高，较长边给出
方向。合法 ADR 状态严格 round-trip；论文未定义的不一致六元组也不再引入外部
径向投影。

### 2. 协方差行列式的灾难性消去

旧 KLD 先构造旋转协方差，再用 `adjugate / det(Sigma)` 求逆。细长旋转框的
`det(Sigma)` 是两个几乎相等的浮点乘积之差，可能舍入为 0，即使输入框五个参数
全部有限且边长为正。

当前 KLD 在预测框的局部坐标系中使用数学等价闭式：中心项由局部中心差和两条边
直接计算，shape 项由相对角及边长比计算，log-det 项为
`log(wp*hp/(wt*ht))`。普通框与公开 MMRotate/O² KLD 逐项一致，同时不再制造虚假
奇异协方差；没有增加 jitter，也没有修改损失定义。

## 删除的掩盖措施

- 删除 ADR distribution/geometry 的强制 FP32 转换；AMP dtype 由正常张量提升规则
  决定；
- 删除 `amp_abort_min_scale`。GradScaler 的数值不是正确性判据，scale 小于 1
  本身不应被写成训练失败条件；
- 删除 rotated criterion 的全局 `nan_to_num`。任何 raw loss 非有限都会按具体
  loss 名称 fail loudly；
- 保留 scale before/after、optimizer skipped、分组梯度等日志；新增逐 decoder
  层及逐 DN 层的最小边长、宽高和长宽比分布。

## 昂贵训练前门禁

1. ADR 合法框 round-trip、轴边界、平移、square/near-square 测试通过；
2. 32,768 个 ADR residual -> OBB -> KLD 完整反向压力测试全部有限；
3. KLD 与公开实现普通框数值对齐，极细旋转框前向/反向有限；
4. criterion 对非有限 raw loss 必须抛出带 loss 名称的异常，禁止清零继续；
5. CUDA autocast full smoke 通过后，才启动新的 O² output directory；旧冻结
   checkpoint 不得 resume。
