# CODrone 数据适配

CODrone 是当前首个 DOTA-style OBB 数据集，但不是框架评价逻辑的所有者。
`CODroneDetection` 只负责类别、路径、DOTA 四点标注、ignore 和数据集特有
provenance；AP、Task1 导出和 tile 合并统一由 `engine/evaluation/obb/` 实现。

## 数据范围

- 正式划分：`train`、`val`、`test`；
- 流程验证划分：`train_t`、`val_t`、`test_t`；
- `crossview_patches`、`crossview_splits` 不属于本项目输入；
- 原图或 tile 都返回统一的 `(cx, cy, w, h, theta)` target，坐标约定见
  [OBB 约定](../../framework/obb_conventions.md)。

原图配置由 `configs/dataset/codrone_obb.yml` 提供，标准切片配置由
`configs/dataset/codrone_obb_tiles.yml` 提供。模型 recipe 不复制数据集逻辑。

## Adapter 额外信息

原图 target 会携带图像路径、原始尺寸和 CODrone 文件名中可解析的采集信息。
物化 tile 还会自动加载 sidecar，并提供：

- `partition_id`、`tile_id`、origin、size、overlap 和 step；
- `source_image_id`、源图尺寸和 `source_object_index`；
- object/window IOF、重复 tile 数、到边界的有符号距离；
- manifest、源 inventory hash 和切片协议 provenance。

通用 diagnostics 只消费这些字段，不硬编码 CODrone 文件名。跨 tile 连接同一物理
目标时使用 `source_image_id + source_object_index`。

## 相关协议

- [原图 OBB recipe、评估和诊断](baseline.md)
- [1180/200 标准切片、物化、合并和推理](tiling.md)
- [新增数据集/求值器的集成契约](../../framework/INTEGRATION_CONTRACT.md)
