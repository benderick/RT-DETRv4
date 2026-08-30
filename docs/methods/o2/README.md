# O²-DFINE 复现

O² 是当前框架除 direct-angle 对照外唯一的稳定旋转框精炼方法。本目录保存随代码版本
演进的复现边界和验收协议；第三方论文源码、PDF 与图表原件属于本地参考资料，位于被
Git 忽略的 `research/references/o2/paper_source/`。

- [实现与来源语义审计](implementation_audit.md)
- [诊断、逐层精炼与可视化验收协议](diagnostic_protocol.md)
- 稳定实现：`engine/rtv4/obb/methods/o2/`
- 方法测试：`test/research/o2/`
- 分析工具：`tools/research/o2/`

本项目没有 O²-DFINE 官方源码可直接照搬。复现必须同时以论文、公开的 O²-RTDETR
实现、D-FINE 机制和本仓库的数值/训练验收为依据，不能把后续改进混入稳定复现。
