# Documentation Index · 文档导航

本文是 ResNet Ablation Factory 的文档入口。**请优先阅读当前代码与 docs/paper/，不要把历史设计稿中的待办直接当作当前缺陷。**

## 从哪里开始？

| 任务 | 文档 | 当前角色 |
| --- | --- | --- |
| 项目定位、安装与主要 CLI | [项目 README](../README.md) | **入口 / 当前** |
| Codex、OpenCode 等代理开发规范 | [AGENTS.md](../AGENTS.md) | **开发契约 / 当前** |
| 论文段落→实现→验证标准 | [01_PAPER_TO_CODE.md](paper/01_PAPER_TO_CODE.md) | **研究依据 / 当前** |
| Plain/Residual、深度、Shortcuts、扩展及混杂因素 | [02_ABLATION_MATRIX.md](paper/02_ABLATION_MATRIX.md) | **实验设计 / 当前** |
| 运行、日志、断点与最终测试 CLI | [03_RUNBOOK.md](paper/03_RUNBOOK.md) | **操作手册 / 当前** |
| 实验 YAML、论文配方与历史示例 | [configs/README.md](../configs/README.md) | **配置索引 / 当前** |
| 旧论文实现审计、早期讨论与长篇设计稿 | [历史归档索引](archive/README.md) | **历史参考 / 非当前状态** |

## 推荐阅读顺序

1. 初次使用：项目 README → 03_RUNBOOK → configs 索引 → Factory Mode dry-run。
2. 研究论文细节：01_PAPER_TO_CODE → 02_ABLATION_MATRIX → 对应模型源码和测试。
3. 新增消融：AGENTS.md → 02_ABLATION_MATRIX → 对照配置 → 测试 → 单种子 smoke → 独立多种子实验。
4. 查找早期实现背景：archive/README.md；历史材料不能覆盖当前源码和 CI 检查。

## 当前文档的研究边界

- 论文原始结果仅作为**引用**；仓库实际结果必须可由 manifest 和 metrics.jsonl 核验。
- CIFAR Factory Mode 使用官方训练集的独立验证划分，最终 test 单独调用。旧版训练脚本不具备此保护。
- 原论文 ImageNet 数据增强、LR 动态调度、10-crop/multi-scale 测试及 CIFAR per-pixel mean subtraction 尚未全量实现。
- 多种子统计、真实指标和计算成本是可靠消融所需证据；代码合入或单测通过不意味着论文复现成功。

## 目录政策

- **docs/paper/**：当前可以直接用于实验的理论—代码对应表、协议和执行说明。
- **docs/archive/**：历史长篇设计讨论和某一时间点的静态审查；保留原路径兼容说明。
- **docs/superpowers/**：较早期方案设计与规划资料，为避免破坏既有内部引用保留原路径，其内容按历史资料理解。
- 新增文档优先使用清晰、可链接的文件名，保持外部引文和现有 CLI 路径稳定。
