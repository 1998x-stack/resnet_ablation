# ResNet Ablation Factory

**论文驱动、配置可追溯的 ResNet 架构复现与受控消融实验平台。**

*Paper-grounded ResNet experiments with configurable architectures, deterministic splits, step-aware training, instrumentation and auditable reports.*

项目以 He et al. 的 [Deep Residual Learning for Image Recognition (CVPR 2016)](https://arxiv.org/abs/1512.03385) 为主线，研究深度网络的优化退化问题、Plain/Residual 对照、Shortcut A/B/C 与 Bottleneck 设计；SE/ECA、DropPath、Deep Stem 等作为**独立的后续工程扩展**。

> **研究状态 / Research status**：模型、Factory Mode、实验矩阵、合成数据端到端测试和日志/绘图工具已提交。**本仓库尚无经确认的 CIFAR/ImageNet 论文级基准复现结果。** 原始论文数据、合成数据测试和本仓库真实训练指标不得混用。详细差异见 [论文—代码映射](docs/paper/01_PAPER_TO_CODE.md)。

## 1. 研究问题与核心能力

ResNet 论文关注的是：增加网络深度后，Plain CNN 为什么可能出现**更高的训练误差**；残差连接能否缓解这一优化问题？因此本项目将 train loss/error 与 validation error **分别**记录和比较，不能只使用最终验证准确率证明优化退化。

| 研究维度 | 当前实现 | 边界 |
| --- | --- | --- |
| CIFAR 网络 | 20、32、44、56、110、1202 层；6n+2 结构；Plain/Residual | 未完成全部深度的大规模实验 |
| ImageNet 网络 | 18、34 BasicBlock；50、101、152 Bottleneck | 未复现原文完整 ImageNet 训练与评估协议 |
| Shortcut | A 零填充、B 仅维度变化时投影、C 每个 residual block 投影 | B/C 产生额外参数 |
| Bottleneck | 原始 v1（1×1 stride）与现代 v1.5（3×3 stride） | v1.5 是后续工程变体 |
| 扩展 | SE/ECA、DropPath、Deep Stem、avg-projection | 不是 CVPR 2016 原论文的消融 |
| Factory | 模型、数据集、优化器、损失函数、调度注册工厂 | 当前已验证训练配置以 SGD 为主 |
| 流程 | Seed 配对矩阵、日志、checkpoint、最终测试、统计图表 | 最终测试需单独运行，绝不自动当验证集使用 |

原论文的实验与后续扩展划分、混杂因素及科学验收门槛见 [消融矩阵](docs/paper/02_ABLATION_MATRIX.md)。

## 2. 两条执行入口

**Factory Mode：用于受控研究，推荐。** 模块入口在 resnet_ablation/research/，CIFAR 官方训练集固定拆分 train/val（默认 45k/5k），训练流程不会读取官方 test；显式 final-test 才可读取测试数据。输出实验 manifest、实际训练日志、TensorBoard 和可恢复 checkpoint。

**Legacy：保留兼容。** scripts/train.py、scripts/eval.py 和根目录 configs/*.yaml 维持历史命令。该路径的 CIFAR loader 会把官方 test 用作循环验证集，且已有恢复、学习率记录等已知限制。**不得把 Legacy 结果与 Factory Mode 结果混合统计或声称论文级复现。**

| 任务 | 入口 |
| --- | --- |
| 模型架构与训练控制 | resnet_ablation/models/、resnet_ablation/config.py |
| 研究数据/训练/矩阵/报告 | resnet_ablation/research/{factories,runner,sweep,report,profile}.py |
| 受控论文实验 | configs/paper/ |
| 历史脚本和示例 | scripts/、configs/*.yaml、run_all.sh |
| 论文与执行说明 | docs/paper/、[文档导航](docs/README.md) |
| 开发代理规约 | [AGENTS.md](AGENTS.md) |

## 3. 快速开始

在仓库根目录执行。建议 Python 3.11，以及相互兼容的 PyTorch/TorchVision 版本。

~~~bash
git clone https://github.com/1998x-stack/resnet_ablation.git
cd resnet_ablation

python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[research]"

# 不下载 CIFAR/ImageNet 的离线架构、协议和合成数据 E2E 测试
python -m pytest -q tests/test_paper_research.py tests/test_paper_e2e.py tests/test_docs_links.py
python -m compileall -q resnet_ablation scripts tests

# 仅显示实验计划：不下载、不训练
python -m resnet_ablation.research.sweep \
  --manifest configs/paper/cifar10_depth_matrix.yaml
~~~

**资源提醒**：CIFAR 深度矩阵目前为 6 条条件 × 3 个随机种子，每组 64,000 optimizer steps。只有显式加入 --execute 才会下载缺失数据并启动训练；运行前确认数据目录、GPU、存储、时间预算。

## 4. Factory Mode 端到端工作流

### 4.1 预览 / 执行配对实验

~~~bash
# dry-run：默认无训练副作用
python -m resnet_ablation.research.sweep \
  --manifest configs/paper/cifar10_depth_matrix.yaml

# 显式触发全矩阵训练
python -m resnet_ablation.research.sweep \
  --manifest configs/paper/cifar10_depth_matrix.yaml --execute
~~~

矩阵生成的配置保存在 runs/paper/矩阵名称/_configs/；每个 variant/seed 使用独立的 run 目录，避免 checkpoint 和指标覆盖。

### 4.2 单组训练、调试与成本分析

~~~bash
# CIFAR 论文 64k-step 学习率调度（但不等同精确复现原文预处理）
python -m resnet_ablation.research.runner run \
  --config configs/paper/cifar10_paper_control.yaml \
  --schedule paper_cifar_64k --steps 64000 \
  --run-dir runs/paper/single/resnet20_seed1

# 10-step CPU smoke：仅测试代码链路，不是 benchmark
python -m resnet_ablation.research.runner run \
  --config configs/paper/cifar10_paper_control.yaml \
  --schedule cosine_step --steps 10 \
  --run-dir runs/smoke/resnet20_seed1 --device cpu

# 统计参数、Conv2d/Linear MACs、输出形状，不下载数据
python -m resnet_ablation.research.profile \
  --config configs/paper/cifar10_paper_control.yaml
~~~

原论文 CIFAR 学习率在 32k 和 48k optimizer steps 衰减，ResNet-110 具有特殊 warmup；本工程使用 per-channel mean/std 而非原文 per-pixel mean subtraction，**只有学习率调度与原文相近不意味着全流程等价**。

### 4.3 汇总与独立最终测试

~~~bash
# 必须已存在真实 run；声明本次比较唯一允许改变的变量
python -m resnet_ablation.research.report \
  runs/paper/cifar10_depth/baseline/seed_1 \
  runs/paper/cifar10_depth/plain20/seed_1 \
  --allow-factor model.residual --out reports/plain_vs_residual

# 仅完成实验选择之后：冻结模型，独立读取 CIFAR 官方 test
python -m resnet_ablation.research.runner final-test \
  --config configs/paper/cifar10_paper_control.yaml \
  --run-dir runs/paper/single/resnet20_seed1
~~~

正式结论应比较**相同种子集合**，展示单种子数据、均值和样本标准差，并同时报告模型参数、训练时间、吞吐和计算口径。详见 [运行手册](docs/paper/03_RUNBOOK.md)。

## 5. 可复现实验矩阵

| 配置文件 | 设计 | 来源类别 |
| --- | --- | --- |
| [cifar10_depth_matrix.yaml](configs/paper/cifar10_depth_matrix.yaml) | Plain/Residual × CIFAR 深度 | 原论文核心 |
| [imagenet_degradation_matrix.yaml](configs/paper/imagenet_degradation_matrix.yaml) | ImageNet Plain/Residual × 18/34 层 | 原论文核心 |
| [imagenet_shortcut_matrix.yaml](configs/paper/imagenet_shortcut_matrix.yaml) | ResNet-34 Shortcut A/B/C | 原论文核心 |
| [imagenet_bottleneck_matrix.yaml](configs/paper/imagenet_bottleneck_matrix.yaml) | v1 与 v1.5 | 现代扩展 |
| [cifar10_attention_extension.yaml](configs/paper/cifar10_attention_extension.yaml) | SE/ECA/DropPath | 现代扩展 |
| [imagenet_stem_downsample_extension.yaml](configs/paper/imagenet_stem_downsample_extension.yaml) | Stem 与 shortcut downsample | 现代扩展 |

使用和组织约定见 [configs/README.md](configs/README.md)。单因素比较只允许变化声明的因素；深度改变也会改变参数规模，Shortcut 投影也有参数成本，必须明示混杂因素。

## 6. 日志、可视化与结果可信度

Factory Mode 每个 run 独立记录：

~~~text
runs/.../<variant>/seed_<n>/
├── manifest.json       # resolved config、code SHA、dataset split SHA、环境/设备
├── metrics.jsonl        # step/epoch 的原始事件与指标
├── train.log            # Loguru
├── tensorboard/         # TensorBoard scalars
├── last.pt              # 最后完整 epoch checkpoint
├── best.pt              # 验证最佳 checkpoint（如有）
├── status.json          # completed_steps / is_complete
└── final_test.json      # 显式 final-test 后才出现
~~~

报告脚本基于真实 JSONL 生成 runs.csv、summary.json、training_loss.png、training_error.png、validation_error.png、learning_rate.png、gradient_global.png、residual_rms_ratio.png、multi_seed_comparison.png。**未完成的 run 不进入成功组均值；不存在的图表或数字不得伪造。** 残差 RMS 比例是用于诊断的代理指标，不是论文 Figure 7 的严格复现。MACs 只统计 Conv/Linear 乘加，非完整 FLOPs。

Checkpoint 使用临时文件加原子替换；支持**完整 epoch 边界**恢复，而不是任意 mini-batch 精确恢复。仅加载可信来源的 PyTorch checkpoint。

## 7. 仓库结构与文档

~~~text
.
├── README.md / AGENTS.md             项目入口与开发代理规约
├── configs/
│   ├── README.md                    配置索引与协议界限
│   ├── paper/                       研究控制配置及实验矩阵
│   └── *.yaml                       历史兼容配置
├── resnet_ablation/
│   ├── models/                      ResNet 架构及 Shortcut
│   ├── research/                    工厂、训练、矩阵、报告、分析
│   ├── engine/                      Legacy 引擎
│   └── config.py, data.py, ...      通用配置与历史实现
├── scripts/                         Legacy train/eval
├── tests/                           架构、合成数据 E2E、文档质量
├── docs/
│   ├── README.md                    文档阅读地图
│   ├── paper/                       原始论文证据与执行手册
│   ├── archive/                     历史材料归档
│   └── superpowers/                 旧设计规划（原路径保留）
└── .github/workflows/               离线 CI 检查
~~~

请以 [docs/README.md](docs/README.md) 为文档导航入口。旧版长篇中文设计草稿归档于 docs/archive/，原根目录路径留兼容跳转说明。

## 8. 旧版兼容与开发规范

以下命令仅用于复现旧版执行方式，不建议用其结果进行严格论文归因：

~~~bash
python scripts/train.py --config configs/cifar10_resnet20.yaml
python scripts/eval.py \
  --config configs/cifar10_resnet20.yaml \
  --ckpt resnet_ablation/checkpoints/cifar10_resnet20/last.pt
~~~

关于 Legacy 的 CIFAR test 泄漏、resume、checkpoint 更新周期及 LR 日志问题，参见 [2026-09-21 历史审查](docs/REVIEW_2026-09-21.md)；该审查记录的是**当时的实现状态**，不应覆盖新 Factory Mode 的现状。

AI 编程代理与贡献者请遵循 [AGENTS.md](AGENTS.md)：模型/数据/调度变化必须补对照测试；优先离线测试和 matrix dry-run；未经实际长期训练不得宣称 SOTA、精确论文复现或硬件性能优势。

## 9. References

1. He, K. et al. (2016). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). CVPR. **主要原始依据**。
2. He, K. et al. (2016). [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027). ECCV. 后续 pre-activation 研究，**当前未实现完整对应架构消融**。
3. Hu, J. et al. (2018). [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507). CVPR. SE 扩展。
4. Wang, Q. et al. (2020). [ECA-Net](https://arxiv.org/abs/1910.03151). CVPR. ECA 扩展。
5. [TorchVision ResNet reference](https://docs.pytorch.org/vision/stable/models/resnet.html). v1.5 结构说明。

**Evidence labels**：paper-reported / repository-measured / synthetic-test / not-yet-validated。
