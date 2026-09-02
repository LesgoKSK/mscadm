# Architecture-v1 G0-A：NWP 模式不确定性审计结果

完成日期：2026-09-02

正式状态：`G0_A_NWP_MODE_UNCERTAINTY_GO`

证据范围：仅使用 267 个 train calendar days 的嵌套交叉拟合前提审计。本文不是 validation 模型比较，不包含 diffusion 训练或生成场景结果。

冻结协议：[`ARCHITECTURE_V1_G0_A_PREDICTABILITY_PROTOCOL.md`](ARCHITECTURE_V1_G0_A_PREDICTABILITY_PROTOCOL.md)

## 1. 一句话结论

在先用 out-of-fold NWP 模型移除条件均值、排除精确 0/1 atom 与缺失位置，并控制 active-mask pattern 后，**NWP 仍能稳定预测六类风电时空 residual 的条件不确定性**。

因此，`predictability-aligned diffusion` 的第一个必要前提通过；下一步只获准设计和冻结 G0-B。当前结果不能说明按该信息修改 diffusion 一定有效。

## 2. 数据与运行身份

| 项目 | 正式值 |
|---|---:|
| materialized target roles | `train` only |
| train days | 267 |
| train date SHA256 | `f67eb5fd4382b2d69d2d22d23488ec618cd3b3b9547a9bbcc71bdd39587c8cf5` |
| train array SHA256 | `ec756d175edc4b265311e7a3b642a58d84aebd4e266392932ee0a6bb3dd56e3a` |
| train-only bundle SHA256 | `953ffccc9f0241c19757d0aeaf9de98cda2a268efef89b7cf977957ef5c0d7aa` |
| frozen G0-A config SHA256 | `5d1d5b7942cbb42f1b5a1af70f09e0bfdc99f2d29d9935b9b9c68968e20c5ead` |
| result JSON SHA256 | `37a55921016bd701ffb2f5a66f56b6abbaa28d027cfdb9b46e0b3f01f8d9b7e7` |
| runner SHA256 | `3f2655ab602121a02c10408ba31e6b390dd4355651ef52b8ca31d6cc5342ae9b` |
| analysis module SHA256 | `e03656c99a787171de359b8e93511bb7ebcb56d17a4911c27589b12c9a0e7797` |

Validation bank 未构造；validation、calibration、selection、R-SEEN 和 final target arrays 均未 materialize。没有保留任何拟合权重。

## 3. 样本与 mask 审计

| 项目 | 结果 |
|---|---:|
| raw observed fraction | 0.998112 |
| observed 中 continuous interior 比例 | 0.918432 |
| 每日 active cells 最小/中位/最大 | 123 / 230 / 240 |

六个 group 的最小有效秩全部超过冻结下限 1.5：

| Group | 最小有效秩 |
|---|---:|
| low_common | 2.0449 |
| low_local | 18.4039 |
| mid_common | 4.0924 |
| mid_local | 36.8319 |
| high_common | 6.1627 |
| high_local | 55.4642 |

这说明结果不是通过把 atom 数值填成连续轨迹得到的，也没有某个频带因 mask 过稀而失去定义。

## 4. 主要结果

分数为六个 group 等权平均的 Gaussian variance log score，越低越好。

| 模型 | Macro score ↓ |
|---|---:|
| S0：static | 1.680991 |
| M0：mask-only | 1.662227 |
| N1：mask + NWP | **1.617105** |

主要对比：

\[
S_{M0}-S_{N1}=0.045123.
\]

Calendar-month cluster bootstrap：

\[
95\%\ CI=[0.032366,\ 0.060555],
\]

且 10,000 次 bootstrap 中改善为正的比例为 1.0。

相对于 M0 与逐日 saturated score 之间的可约 deviance，N1 解释：

\[
22.00\%.
\]

这不是 22% 的最终预测误差下降，而是本次条件方差 quasi-score 中可约部分的解释比例。

## 5. 跨 fold 与跨 mode 稳定性

六个 outer folds 的 `M0−N1` 全部为正：

```text
0.04517, 0.03621, 0.04175,
0.03128, 0.02285, 0.09375
```

六个 mode groups 也全部改善：

| Group | M0 − N1 ↑ |
|---|---:|
| low_common | 0.03048 |
| low_local | 0.02104 |
| mid_common | 0.04844 |
| mid_local | 0.05293 |
| high_common | 0.06359 |
| high_local | 0.05426 |

因此正式结果同时覆盖：

- spatial common 与 local；
- temporal low、mid 与 high；
- 6/6 outer date folds。

高频改善更大，但结论不依赖挑选高频 group；low-frequency groups 也保持正向。

## 6. Shuffled-NWP 反事实

正确 NWP 的主要改善为：

```text
+0.045123
```

32 个 inference-only wrong-day NWP 的改善范围为：

```text
-0.040286 ～ -0.013094
```

第 95 百分位为：

```text
-0.017335
```

也就是说，同一个已经拟合的 N1 在保留正确 mask、只错配当天 NWP 后，不仅失去收益，而且全部劣于 M0。这支持“天气与当天 residual uncertainty 的对应关系”是真实信号，而不只是增加了 34 个特征。

## 7. 校准诊断

| Group | log-energy calibration slope | Spearman |
|---|---:|---:|
| low_common | 0.922 | 0.223 |
| low_local | 1.054 | 0.430 |
| mid_common | 0.948 | 0.362 |
| mid_local | 1.118 | 0.741 |
| high_common | 1.113 | 0.645 |
| high_local | 1.131 | 0.769 |

所有 rank association 均为正，calibration slopes 约处于 0.92–1.13。Low-common 的排序信号最弱，因此后续不能声称所有 mode 的可预测性同样强。

## 8. Go/No-Go 判定

| 冻结门 | 结果 |
|---|---:|
| primary bootstrap lower bound > 0 | Pass |
| deviance explained ≥ 5% | Pass：22.00% |
| positive outer folds ≥ 5/6 | Pass：6/6 |
| positive groups ≥ 4/6 | Pass：6/6 |
| common/local 均覆盖 | Pass |
| 至少两个 temporal bands | Pass：3/3 |
| N1 macro 优于 S0 | Pass |
| 正确 NWP 超过 Shuffle 第 95 百分位 | Pass |
| provenance、有效秩与有限值 | Pass |

全部冻结门通过，故状态为：

```text
G0_A_NWP_MODE_UNCERTAINTY_GO
```

## 9. 工程事件与可复现性

第一次 target-aware 执行在产生任何结果文件前 fail-closed：log-variance Newton solver 在数值平坦点无法接受一个机器精度级步长。随后只补充了“梯度或步长低于冻结 tolerance 时视为收敛”的数值判据；目标函数、数据、特征、folds、模型、alpha grid、统计方法和所有阈值均未改变。

修正后的正式运行再次执行到独立临时目录。移除 `started_utc/finished_utc` 后，两次完成运行的整个 scientific JSON payload 逐字段完全一致。

## 10. 可以说什么，不能说什么

当前可以说：

> 在本地 GEFCom2014 的 267 个冻结 train days 上，NWP 包含可跨日期 fold 泛化的模式级条件不确定性信息；该信息超过 static、mask-only 与 wrong-day NWP 解释。

当前不能说：

- predictability-aligned diffusion 已经优于 D0-v；
- NWP schedule 已经优于 CW 或 MuLAN-lite；
- ramp CRPS、joint ES 或 lagged variogram 已改善；
- 该现象已经通过 validation 或外部数据确认；
- conditional multidimensional schedule 是本项目首创。

下一步是单独设计 G0-B 的 corruption-budget matching、CW、Fixed-band、MuLAN-lite、Predictability-aligned 和 Shuffled-schedule 对照；在新协议冻结前不实现完整模型。
