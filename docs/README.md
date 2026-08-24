# 项目文档地图

- `framework/`：坐标规范、公共 API 和
  [集成契约](framework/INTEGRATION_CONTRACT.md)；
- `datasets/`：数据集 adapter、切片协议和基线 recipe；
- `research/o2/`：O² 论文资料、复现边界和实现审计；
- `testing/`：当前稳定底座的验证记录。

当前仓库只支持 direct-angle 与 O² ADR。新增文档不得在根目录散落：数据集文档
放入 `datasets/<name>/`，通用规则写入 `framework/`，方法私有资料按方法单独建目录。
