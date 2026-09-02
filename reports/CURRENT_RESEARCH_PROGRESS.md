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
文献碰撞与等价性复审
固定条件协方差、通用多维条件 schedule 均不能作为首创
                │
                ▼
G0-A NWP 模式不确定性审计
6/6 folds、6/6 mode groups 改善，全部冻结门通过
                │
                ▼
当前状态：G0-A Go；只获准设计 G0-B，尚未形成新模型
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
| G0-A | NWP 能否预测 mask-aware 六类时空 residual 的条件不确定性 | 全部预注册门通过；必要前提 Go，但不是 diffusion 收益证据 |

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

## 5. 当前冻结的研究问题

经过 CW-Diff、MuLAN、MA-TSD 等近邻复审，以下表述已经放弃：

- “NWP 条件协方差是一种新的 forward process”；
- “由条件决定不同 mode 的 diffusion clock 是通用方法首创”；
- “标准 iid diffusion 在表达能力上无法学习时间依赖”。

当前只冻结一个可证伪的风电问题：

> 在总 corruption budget 匹配的条件下，利用 NWP 可预测的模式级条件不确定性来分配 diffusion SNR，是否能比标准 IID、条件白化、固定非各向同性日程和通用 learned adaptive schedule 更有效地降低风电联合条件分布的有限样本去噪复杂度？

这里的候选贡献不是发明 conditional multidimensional schedule，而是检验：

```text
NWP
  ↓
模式级条件可预测性 / 不确定性
  ↓
显式、受约束的 corruption allocation 原则
```

完整问题目前仍只是候选论文假说。G0-A 只为其中第一个必要前提提供了数据支持，尚未验证改变 diffusion path 是否有效。

## 6. 当前执行边界：G0-A

完整问题被拆成两个先后门：

```text
G0-A：NWP 能否预测六类时空成分的条件不确定性？
  ├─ No-Go：结束整条 predictability-aligned diffusion 假说
  └─ Go：只允许另行设计并冻结 G0-B

G0-B：这种信息是否值得控制 diffusion path？
  ├─ 尚未冻结
  └─ 未来至少比较 IID / CW / Fixed / MuLAN-lite / 候选方法 / Shuffle
```

G0-A 已于 2026-09-02 冻结并完成。它具有以下边界：

- 只使用 267 个 train calendar days；
- 不读取 validation、calibration、selection、R-SEEN 或 final targets；
- 只审计条件不确定性，不训练 diffusion；
- 使用固定的 `3 temporal bands × 2 spatial groups`，不估计 240 条 schedule；
- 精确 0/1 atom 与缺失坐标不进入 continuous residual；
- 条件均值和条件方差均采用嵌套 out-of-fold 估计；
- 主要比较 static、mask-only、NWP 和 inference-only Shuffled-NWP；
- 全部 Go/No-Go 条件必须同时通过。

正式结果为 `G0_A_NWP_MODE_UNCERTAINTY_GO`：

- NWP 相对 mask-only 的 macro log-score 改善为 0.04512；
- calendar-month cluster bootstrap 95% CI 为 `[0.03237, 0.06055]`；
- 可约 deviance explained 为 22.00%；
- 6/6 outer folds 与 6/6 mode groups 均为正；
- 正确 NWP 改善为正，而 32 个 wrong-day NWP 对照全部为负。

这说明 NWP 确实包含模式级条件不确定性信息，但尚未说明用它控制 diffusion 会改善最终场景。冻结协议见 [`ARCHITECTURE_V1_G0_A_PREDICTABILITY_PROTOCOL.md`](ARCHITECTURE_V1_G0_A_PREDICTABILITY_PROTOCOL.md)，完整结果见 [`ARCHITECTURE_V1_G0_A_PREDICTABILITY_RESULT.md`](ARCHITECTURE_V1_G0_A_PREDICTABILITY_RESULT.md)。

## 7. 当前不会做什么

- 不继续 family-v1.3 SNR/NWP recurrence gate；
- 不把 family-v1.2 的统计显著小效应包装成模型创新；
- 不因为 D0-v 被用作 Probe 平台就宣布 Diffusion 已经战胜 Flow；
- 不把固定 `Σ_NWP`、条件白化或通用条件多维 schedule 包装成方法首创；
- 不把 G0-A Go 解释成 G0-B 或完整 diffusion 已经有效；
- 不在新的 G0-B 协议冻结前实现完整 diffusion；
- 不同时恢复 transition-aware loss、Flow proper-score adapter 等备用工程线；
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

- [`ARCHITECTURE_V1_G0_A_PREDICTABILITY_RESULT.md`](ARCHITECTURE_V1_G0_A_PREDICTABILITY_RESULT.md)
- [`ARCHITECTURE_V1_G0_A_PREDICTABILITY_PROTOCOL.md`](ARCHITECTURE_V1_G0_A_PREDICTABILITY_PROTOCOL.md)
- [`ARCHITECTURE_V1_FAMILY_V1_2_FORMAL_EVALUATION_RESULT.md`](ARCHITECTURE_V1_FAMILY_V1_2_FORMAL_EVALUATION_RESULT.md)
- [`ARCHITECTURE_V1_FAMILY_V1_2_PROBE_PROTOCOL.md`](ARCHITECTURE_V1_FAMILY_V1_2_PROBE_PROTOCOL.md)
- [`ARCHITECTURE_V1_FAMILY_V1_1_FORMAL_COMPARISON_RESULT.md`](ARCHITECTURE_V1_FAMILY_V1_1_FORMAL_COMPARISON_RESULT.md)
- [`ARCHITECTURE_V1_TEMPORAL_MECHANISM_V3_2_RESULT.md`](ARCHITECTURE_V1_TEMPORAL_MECHANISM_V3_2_RESULT.md)
- [`CROSS_MODEL_ARCHITECTURE_DIAGNOSTIC.md`](CROSS_MODEL_ARCHITECTURE_DIAGNOSTIC.md)
- [`MSCADM_PAPER_REPRODUCTION_REPORT.md`](MSCADM_PAPER_REPRODUCTION_REPORT.md)
