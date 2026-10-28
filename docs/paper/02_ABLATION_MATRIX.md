# 02｜消融问题、对照组、混杂因素与验收标准

## 原论文直接支持的消融

| 编号 | 受控比较 | 变量 | 数据 | 证据判据 |
| --- | --- | --- | --- | --- |
| P01 | plain20 vs ResNet20；plain56 vs ResNet56 | residual | CIFAR-10 | 优化步数对齐的 train loss/error |
| P02 | plain20 vs plain56；ResNet20 vs ResNet56 | depth | CIFAR-10 | 深层 Plain 是否出现更高训练错误 |
| P03 | 20/32/44/56/110/1202 | depth 及参数共同变化 | CIFAR-10 | 性能、计算成本、深度趋势 |
| P04 | ResNet34 A vs B vs C | projection policy | ImageNet | top1/top5 + 参数量 |
| P05 | plain18/34 与 ResNet18/34 的 2×2 设计 | depth×residual | ImageNet | 训练/验证曲线及交互作用 |
| P06 | ResNet50/101/152 | 模型规模 | ImageNet | top1/top5、参数、吞吐 |

## 工程扩展（不能称为 ResNet 2016 原文消融）

| 编号 | 对照 | 归因警戒 |
| --- | --- | --- |
| E01 | Bottleneck v1 vs v1.5 | 改变 stride 位置，可能改变感受野/速度 |
| E02 | none vs SE vs ECA | 通道注意力同时增加/改变容量 |
| E03 | drop_path_rate 0 vs >0 | 相当于训练正则变动 |
| E04 | 7×7 stem vs 3×3×3 deep stem | 改变 stem 参数 |
| E05 | proj vs avgproj | 改变短接下采样算子 |
| E06 | cosine vs multistep、增强/混合策略 | 属于训练 recipe 变化 |

## 统计与因果纪律

同一对照仅改变清单中声明的因素；固定 dataset、split_sha256、seed 集合、数据增强、训练总 step、optimizer、评价视角。单次效果不等于可信结论，最低准备 3 个独立 seed；报告平均/样本标准差和每个 seed 的原始结果，不为缺失 run 编造数值。各组同时列出模型参数量、训练时间、吞吐与硬件，不能仅展示最高准确率。

重要：深度改变时参数量不可避免改变，说明这是“深度/容量共同变化”。检测退化问题时必须报告**训练误差**，仅用验证集准确率不能断言优化退化。原始 CIFAR test 只能在选择模型后通过 final-test 使用。ImageNet 的本地 val 是官方验证集，不能标为官方 test。

## 五级验收

1. 架构：Block A/B/C 参数策略、v1/v1.5 stride、20/56/110 block 数与 finite backward。
2. 数据：45k/5k 划分无重叠，split SHA 可复算；训练阶段完全不访问官方 test。
3. 恢复：每完整 epoch 的 checkpoint 可原子替换，按 next_epoch 接续、学习率与 global_step 连贯、RNG 状态恢复。
4. 统计：图和 CSV 来自真实日志；拒绝非法 recipe 差异；未完成 run 不参与成功组平均值。
5. 基准：完整 CIFAR/ImageNet、独立 seed、机器资源和真实吞吐，冻结模型后评估最终测试。**本次代码提交不意味着已达到第 5 级。**
