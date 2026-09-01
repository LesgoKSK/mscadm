# Architecture-v1 family-v1.2 Temporal Utility Probe 正式结果

日期：2026-09-01

正式状态：**FAMILY_V1_2_ORDERED_RECURRENCE_NO_GO**

本轮结论：**不进入 family-v1.3 的 SNR/NWP 门控 recurrence；冻结并停止这一条小型顺序 GRU residual 分支。**

## 1. 一句话结论

在已经具有小时位置编码和全局时间注意力的 D0-v Diffusion 上，正确小时顺序的 GRU residual 确实比打乱顺序略好，而且这个微小差异可以被统计检验稳定识别；但是，八个 DDIM 阶段的改善量全部只有预注册实质门槛的大约 0.5%～13.7%，没有任何阶段达到“值得继续开发”的幅度。因此，本轮按冻结规则判定为 No-Go。

这不是说 Diffusion 不行，也不是说时间信息没有用。它只说明：

> 在当前 D0-v 已有时间建模能力之上，再叠加这个小型、有方向的顺序 GRU context residual，增量收益太小，不足以成为后续创新主线。

## 2. 这次到底比较了什么

三个冻结的 D0-v best EMA backbone 分别在八个 DDIM 阶段上增加同构 adapter，并训练两种版本：

- **Chronological**：GRU 按真实小时顺序 `0→1→…→23` 传播信息；
- **Shuffle**：GRU 访问顺序被打乱，但 NWP 数值、物理小时位置、参数量、训练预算和随机变量均保持一致。

正式问题是：

```text
正确时间顺序的模型   与   打乱时间顺序的模型
        │                         │
        └──── 唯一计划内差别：小时邻接关系 ────┘
                              │
                    ramp 与 lagged 指标是否
                    稳定且达到实际有用的幅度？
```

每个 Chronological checkpoint 还使用两个训练中从未出现的 Shuffle permutation 做同 checkpoint 反事实检查。这样可以区分“正确邻接本身的作用”和“多一组参数或一般平滑作用”。

## 3. 判定逻辑

所有主要分数都是越低越好。正式效应定义为：

```text
Shuffle 分数 − Chronological 分数
```

因此，效应为正表示正确时间顺序更好。一个 DDIM block 必须同时满足：

1. ramp CRPS 和 lagged variogram 的 max-T 同时置信带下界都高于 0；
2. ramp CRPS 点改善至少为 **0.001**；
3. lagged variogram 相对改善至少为 **5%**；
4. 同 checkpoint 打乱顺序的反事实检查通过；
5. 相对原始 D0-v 的 level、joint、coverage 和 width 安全门通过。

只有至少两个相邻 block 同时通过全部条件，才允许进入 family-v1.3 门控模型。这个顺序是实验前冻结的，不能在看到结果后降低门槛。

## 4. 八个阶段的主要结果

括号内为 16 个主要 endpoint 联合控制后的 95% max-T simultaneous lower bound。Lagged 数值已经换算为百分比。

| Block | DDIM 阶段 | Ramp 改善（下界） | Lagged 相对改善（下界） | 同 checkpoint 检查 | 分布安全 | 完整支持 |
|---|---|---:|---:|---|---|---|
| B1 | 起始高噪声 | 0.000128（0.000077） | 0.471%（0.248%） | 通过 | 通过 | **失败** |
| B2 | 高噪声 | 0.000136（0.000085） | 0.564%（0.350%） | 通过 | 通过 | **失败** |
| B3 | 低中段 | 0.000121（0.000080） | 0.580%（0.399%） | 通过 | 通过 | **失败** |
| B4 | 中段负 log-SNR | 0.000116（0.000081） | 0.661%（0.488%） | 通过 | 通过 | **失败** |
| B5 | 中段正 log-SNR | 0.000122（0.000084） | 0.685%（0.497%） | 通过 | 通过 | **失败** |
| B6 | 清晰化阶段 | 0.000112（0.000081） | 0.658%（0.491%） | 通过 | 通过 | **失败** |
| B7 | 低噪声阶段 | 0.000085（0.000062） | 0.473%（0.350%） | 通过 | 通过 | **失败** |
| B8 | 最终清晰阶段 | 0.000005（0.000003） | 0.025%（0.013%） | 失败 | 通过 | **失败** |
| **预注册实质门槛** | — | **至少 0.001** | **至少 5%** | 必须通过 | 必须通过 | — |

最强的 ramp 效应出现在 B2，只达到门槛的 **13.6%**；最强的 lagged 效应出现在 B5，只达到门槛的 **13.7%**。B8 的影响几乎消失。

所以这里不是“差一点通过”，而是效应量整体比目标小约一个数量级。八个 block 的 `primary_support` 和 `full_support` 均为 `False`，不存在可以合法挑出的相邻阶段组合。

## 5. 为什么“统计显著”仍然是 No-Go

两项主要指标在 B1～B8 的联合置信下界都大于 0，说明 Chronological 相对 Shuffle 的微小优势并不像纯采样噪声。50 个配对 calendar day、共同随机数和三种子平均让实验能够精确识别很小的差异。

但研究目标不是证明差异精确地大于零，而是找到足以支撑新模型复杂度的效果。预注册同时要求：

```text
能够检测到差异  +  差异达到实际有用的幅度
```

本轮只满足前半句，没有满足后半句。若因为 p 值显著就继续做门控，相当于看到结果后取消原先的效应量要求，会把一个很小的机制现象包装成模型创新。

## 6. 三个 omnibus 检验怎么解释

| 检验 | p 值 | 能说明什么 | 不能说明什么 |
|---|---:|---|---|
| DDIM 阶段同质性 | 0.000200 | 微小收益随去噪阶段变化 | 不能证明任一阶段达到实质门槛 |
| NWP regime 同质性 | 0.020596 | stable、moderate、dynamic 的微小收益不同 | 不能自动授权 NWP gate |
| Block × NWP interaction | 0.000600 | 阶段效应还会随 NWP regime 改变 | 不能绕过基础机制 No-Go |

Validation 中包含 13 个 stable 日、18 个 moderate 日和 19 个 dynamic 日；regime 只由 train-only NWP dynamicity registry 分配，未使用目标功率。

这些 omnibus 结果说明“微小效应具有结构”，但冻结决策树要求先出现至少两个相邻 block 的完整实质支持，之后才允许依据结构选择 SNR gate、NWP gate 或双门控。由于该前提为 0/8，本轮不能进入任何 family-v1.3 门控分支。

## 7. 相对原始 D0-v 的表现

Chronological adapter 没有造成持续的分布破坏，八个 block 的安全门全部通过。八个 block 中最不利的 simultaneous upper band 仍明显位于预注册容许范围内：

| 安全指标 | 最不利联合上界 | 容许上限 | 结论 |
|---|---:|---:|---|
| level CRPS 恶化 | 0.000140 | 0.0015 | 通过 |
| normalized joint ES 相对恶化 | 0.213% | 2% | 通过 |
| coverage90 绝对偏移 | 0.350 个百分点 | 2 个百分点 | 通过 |
| width90 相对增幅 | 1.205% | 10% | 通过 |

这说明该 adapter 是“安全但作用很小”，不是“有效但副作用太大”。

作为描述性参照，Chronological 相对原始 D0-v 的最好 ramp 改善约为 0.000136（约 0.26%），最好 lagged variogram 相对改善约为 1.19%；同样远低于注册的 0.001 和 5% 实质门槛。

## 8. 反事实与 Atom 审计

- 同 checkpoint held-out Shuffle 检查在 B1～B7 通过，说明正确小时邻接确实影响了连续场景；B8 未通过。
- 所有 atom metric day vectors 在各组间完全一致。
- 场景归档中的 atom states 和 probabilities 逐数组完全一致。
- 运行时 atom probability 相对复用基线的最大绝对差为 `4.0233e-7`，低于 `1e-6` 容差。
- Atom 总审计状态为 `passed=true`。

因此，观察到的差异来自 interior continuous trajectory 路径，不是零功率、满功率状态或 allocation 改变造成的。

## 9. 运行与数据完整性

- 3 个 backbone 训练种子；
- 8 个 DDIM blocks；
- 2 个训练 order；
- 2 个同 checkpoint held-out permutations；
- 3 个共同 sampling seeds；
- 50 个 validation calendar days；
- 每份归档 100 个 scenarios、10 个区域、24 小时；
- 新生成 288 份归档，复用 9 份 D0-v 归档，共 297 份；
- 24 个 seed×block checkpoint 汇总全部落盘；
- 5,000 次 calendar-day bootstrap/permutation；
- 317 个 SHA256 sidecar 全部复核通过，无失败标记。

本轮只 materialize 了 `train` 和 `validation` targets。`selection`、`calibration`、R-SEEN 和最终外部测试均未访问；它们继续封存。

## 10. 对研究主线的含义

当前证据支持以下表述：

1. D0-v 的生成轨迹中存在可检测的顺序邻接信号；
2. 这个信号随 DDIM 阶段和 NWP dynamicity 改变；
3. 但在已有时间注意力的 D0-v 上，用冻结 backbone 加小型 GRU context residual 提取该信号，实际收益太小；
4. 因此，不应继续在该 residual 上增加 SNR/NWP gate，也不应把它扩写成主要创新点；
5. 该结果不推翻 family-v1.1，也不改变“Flow 与 D0-v 尚无全面胜者”的正式结论；
6. 若继续以 Diffusion 为研究平台，下一项创新应直接作用于 ramp/increment 的学习目标或生成变量，而不是继续给现有 context 叠加顺序 recurrence。

建议把下一阶段定义为一个新的、独立预注册的 **transition-aware diffusion probe**：保持 D0-v 和数据协议不变，仅检验“直接建模 level 与 increment/ramp 一致性”是否能达到预注册的 ramp 与 lagged 实质门槛。该方案尚未冻结，也不应在本报告中预先宣称有效。

## 11. 可复核文件

- 正式结果：`outputs/architecture_v1_family_v1_2_probe/formal_evaluation/FAMILY_V1_2_PROBE_RESULT.json`
- 冻结记录：`outputs/architecture_v1_family_v1_2_probe/formal_evaluation/evaluation.freeze.json`
- 八阶段简表：`outputs/architecture_v1_family_v1_2_probe/formal_evaluation/block_summary.csv`
- calendar-day 数组：`outputs/architecture_v1_family_v1_2_probe/formal_evaluation/seed_averaged_per_day_metrics.npz`
- 24 个 checkpoint 汇总：`outputs/architecture_v1_family_v1_2_probe/formal_evaluation/per_checkpoint/`
- 288 份新场景：`outputs/architecture_v1_family_v1_2_probe/formal_evaluation/archives/`

正式结果 SHA256：`2e8e514f42e7f18fe58908df4a02ba0d6d2f498ade3d6f74aed937921a43666a`

评估冻结 SHA256：`07bfd29670182a43b5bdb86b7f8562ee939b9606211957ecbb5003cdf1faf083`

评估 amendment SHA256：`f171daa3190d1ac45805f500a0d43ece21005cf4a61b18bfd89cf2be9c9e34d9`
