# 当前研究进度（唯一维护入口）

最后更新：2026-09-02

本文档是仓库中“目前做到哪里、已经得到什么结论、下一步尚待决定什么”的唯一持续维护入口。

- `*_PROTOCOL.md` 和冻结配置记录实验开始前的规则，不随结果改写；
- `*_RESULT.md` 记录已经完成的正式实验，不作为滚动路线图；
- 保留下来的 HTML 和 `FUTURE_RESEARCH_ROADMAP*` 是特定时间的汇报或调研快照；
- 如果旧文档中的“下一步”与本文冲突，以本文为准。

## 1. 研究问题

本项目从 MS-CADM 论文复现出发，目前关注的问题是：

> 如何生成 10 个风电区域未来 24 小时的 100 条联合场景，使每个小时的概率分布、相邻小时的爬坡变化、跨区域与滞后依赖，以及 0/1 边界状态都尽可能真实？

当前主要缺口不是平均功率水平，而是 **ramp、总变差、最大跳变和 lagged dependence**。

## 2. 当前走到哪里

```text
MS-CADM 论文复现
        │  未复现论文 headline 概率结果
        ▼
跨模型诊断
        │  缺口集中在 ramp / lagged dynamics
        ▼
Architecture-v1 公平实验底座
        │  无泄漏数据协议、共同随机数、三种子
        ├───────────────┐
        ▼               ▼
Flow vs D0-v        时间机制 v3.2
无全面胜者          全局 recurrence No-Go
        │               │
        └───────┬───────┘
                ▼
family-v1.2 Temporal Utility Probe
正确顺序有微小效应，但 0/8 blocks 达到实质门槛
                │
                ▼
当前状态：停止 GRU/SNR/NWP gate 分支，下一种干预尚未冻结
```

## 3. 已完成实验与正式结论

| 阶段 | 回答的问题 | 正式结论 |
|---|---|---|
| MS-CADM reproduction | 本地实现能否重现论文 headline 结果 | 完整链路已重建，但 headline CRPS/coverage 未复现 |
| Cross-model diagnosis | 主要误差来自哪里 | level 接近，ramp、jump、lag-1 和动态天气误差更突出 |
| Architecture-v1 R0 | Flow 基线能否稳定训练 | 经过数据协议和稳定性修复后，可作为公平比较基础 |
| Temporal v3.2 | 直接加入 recurrence 是否有效 | Chronological 优于 Shuffle，但候选因非劣或稳定性失败，No-Go |
| Family-v1.1 | Flow 与 Joint DDPM 谁更好 | Flow 的 level/joint/coverage 更好；D0-v 的 lagged dependence 更好；无全面胜者 |
| Family-v1.2 | D0-v 的哪些去噪阶段真正需要顺序传播 | 顺序效应可检测但过小，ordered recurrence 正式 No-Go |

Family-v1.1 之后采用 D0-v 做 family-v1.2，只是因为它具有明确的 DDIM 阶段坐标，便于做机制干预；这不等于已经把 Diffusion 选为最终模型家族。

## 4. Family-v1.2 最新结果

实验冻结三个 D0-v backbone，只训练小型 GRU context residual，并比较正确小时顺序与严格破坏邻接的 Shuffle。正式评估使用三个训练种子、三个共同 sampling seeds、50 个 validation 日和八个真实 DDIM sampler blocks。

| 判定项 | 结果 |
|---|---:|
| 主要效应达到实质门槛 | 0/8 blocks |
| 同 checkpoint Shuffle 反事实 | 7/8 blocks 通过 |
| 相对 D0-v 的分布安全 | 8/8 blocks 通过 |
| 完整支持 | 0/8 blocks |
| 正式状态 | `FAMILY_V1_2_ORDERED_RECURRENCE_NO_GO` |

正确顺序相对 Shuffle 的差异在统计上可以识别，但最强 ramp 与 lagged 效应分别只达到预注册实质门槛的 13.6% 和 13.7%。因此它属于“机制现象存在，但实际收益太小”，不能据此继续开发 SNR gate、NWP gate 或双门控 recurrence。

详细证据见 [`ARCHITECTURE_V1_FAMILY_V1_2_FORMAL_EVALUATION_RESULT.md`](ARCHITECTURE_V1_FAMILY_V1_2_FORMAL_EVALUATION_RESULT.md)。

## 5. `transition-aware diffusion` 是什么

这是 **未冻结的候选方向**，不是正在执行的实验。

现有生成模型主要学习每个时刻的功率轨迹 `x`。所谓 transition-aware，是让模型还直接关注相邻小时变化：

```text
功率水平：       x[t]
相邻小时变化：   Δx[t] = x[t] - x[t-1]
```

直观上，它要求生成结果同时做到：

1. 每个小时的功率值合理；
2. 从一个小时跳到下一个小时的幅度也合理。

它与刚刚失败的 GRU residual 不同：

- GRU residual 是在 NWP context 上再传播一次时间信息；
- transition-aware 方法会直接修改 diffusion 的训练目标、输出表示或 sampler-level 评分，使 ramp/increment 本身进入学习目标。

可能实现包括 increment-aware denoising loss、level–increment 一致性参数化或 sampler-level proper-score adaptation。三者目前都只是候选，尚未选择，也没有配置、代码或训练任务。

## 6. 当前真正未解决的路线选择

Temporal Utility Probe 已经先做完，因此不再需要讨论“Temporal Probe 与 proper-score adaptation 谁先做”。现在剩下的是两个新的候选：

| 候选 | 核心做法 | 主要优点 | 主要风险 |
|---|---|---|---|
| Transition-aware D0-v | 在 Diffusion 中直接约束 level 与 increment/ramp | 与已稳定的 D0-v 和现有公平协议衔接紧 | 需要证明不是普通辅助损失，且必须改善 ramp 而不伤 level/joint |
| Sampler-matched Flow adaptation | 在混合测度 Flow 上对最终场景做 level–ramp proper-score 后训练 | 直接针对最终概率评价，Flow 已有较好的 level/joint 表现 | sampler 反传、有限 ensemble 估计和数值稳定风险较高 |

**目前没有冻结下一模型，也没有启动新训练。** 下一项正式动作应是在本文档内完成一次二选一判定，然后只为胜出的方向建立一份不可改写的实验协议。

## 7. 当前不会做什么

- 不继续 family-v1.3 SNR/NWP recurrence gate；
- 不把 family-v1.2 的统计显著小效应包装成模型创新；
- 不因为 D0-v 被用作 Probe 平台就宣布 Diffusion 已经战胜 Flow；
- 不访问仍封存的 selection、calibration 或最终外部测试；
- 不同时启动两个重型方向。

## 8. 数据与证据边界

- 当前架构结论仍属于 validation-stage mechanism discovery；
- 历史诊断使用过的日期保持 `R-SEEN`；
- selection、calibration 和最终外部测试仍未用于模型选择；
- 数据、论文 PDF、checkpoint 和大型场景归档保留在本地，不进入公开 Git 仓库；
- 公开仓库保存代码、冻结配置、测试和可阅读的正式结论。

## 9. 文档维护规则

从现在开始：

1. 每完成一个正式阶段，只更新本文档中的状态表、最新结果和下一决策；
2. 新实验仍各自保留一份 frozen protocol 和一份 formal result，以保证预注册与结果不可被滚动叙事覆盖；
3. 不再为每次路线讨论新增新的 roadmap、HTML 或 DOCX；
4. 需要组会材料时，从本文档生成一次性演示版本，并标明生成日期；
5. README 只提供项目简介、当前一句话状态和本文档入口。

## 10. 关键证据入口

- [`ARCHITECTURE_V1_FAMILY_V1_2_FORMAL_EVALUATION_RESULT.md`](ARCHITECTURE_V1_FAMILY_V1_2_FORMAL_EVALUATION_RESULT.md)
- [`ARCHITECTURE_V1_FAMILY_V1_2_PROBE_PROTOCOL.md`](ARCHITECTURE_V1_FAMILY_V1_2_PROBE_PROTOCOL.md)
- [`ARCHITECTURE_V1_FAMILY_V1_1_FORMAL_COMPARISON_RESULT.md`](ARCHITECTURE_V1_FAMILY_V1_1_FORMAL_COMPARISON_RESULT.md)
- [`ARCHITECTURE_V1_TEMPORAL_MECHANISM_V3_2_RESULT.md`](ARCHITECTURE_V1_TEMPORAL_MECHANISM_V3_2_RESULT.md)
- [`CROSS_MODEL_ARCHITECTURE_DIAGNOSTIC.md`](CROSS_MODEL_ARCHITECTURE_DIAGNOSTIC.md)
- [`MSCADM_PAPER_REPRODUCTION_REPORT.md`](MSCADM_PAPER_REPRODUCTION_REPORT.md)
