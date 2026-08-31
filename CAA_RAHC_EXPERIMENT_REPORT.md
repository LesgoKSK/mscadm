# CAA-RAHC 冻结实验报告

## 结论摘要（结果先行）

本结果属于 **post-freeze internal nested outer-split confirmation**（冻结后的内部 nested outer-split 确认），**不是外部确认**。A4 未回退，锁定候选为 `A4_atom1_tail1_d0p25_w0p02`。预注册成功门槛通过 **9/10**；全部门槛是否通过：**否**。

下表的 A0/A4 点值来自 150 个互斥 sealed outer-test 日期上的三 seed pooled 结果。95% CI 来自 5,000 次按日历日成簇的 paired bootstrap，CI 对应表中明确写出的 A4−A0 或 A0−A4 差值定义，不能误读为单个方法点值的置信区间。

| 指标 | A0 点值 | A4 点值 | 配对差值点估计 | 95% CI |
| --- | --- | --- | --- | --- |
| CRPS（A0−A4 为正表示改善） | 0.082321 | 0.081404 | 0.000917 | [0.000619, 0.001242] |
| 90% coverage（CI 对应绝对误差改善） | 0.822287 | 0.849972 | 0.027685 | [0.021861, 0.034028] |
| W90（差值为 A4−A0） | 0.407945 | 0.409933 | 0.001987 | [0.001086, 0.002869] |
| Winkler-90（A0−A4 为正表示改善） | 0.677668 | 0.668876 | 0.008792 | [0.004462, 0.014372] |
| conditional ACE90（A0−A4 为正表示改善） | 0.07737 | 0.051045 | 0.026325 | [0.021208, 0.031408] |

## 1. 证据等级与数据使用边界

实验设计及阈值以 [CAA_RAHC_FROZEN_PROTOCOL.md](CAA_RAHC_FROZEN_PROTOCOL.md) 为准。三个 outer 各含 50 个 sealed test 日期，合计 150 日；本轮每个 outer 都重新训练三个底模 seed。旧研究中的 test 划分在本协议中**仅作为 calibration/候选选择数据**，没有充当本轮 sealed outer test。selection audit 明确记录选择阶段 test access 为 none，lock 记录 `test_archives_accessed=false`。

尽管方法、网格和成功门槛已冻结，本实验仍只是在同一 GEFCom2014 数据域内做内部确认。历史研究已接触过该数据集的全部年份，因此这里不能称为独立外部确认，也不能替代新年份、新场站或新公开数据集上的前瞻验证。

## 2. 方法与锁定流程

- **A0**：按小时 empirical PIT calibration 与 finite-ensemble linear tails，是所有约束和差值的基线。
- **A1**：旧 full RAHC 的 linear-tail 消融。
- **A2**：atom-only，用于分离边界原子修正。
- **A3**：regularized gate-only，不改变 atom。
- **A4**：atom + regularized local gate，预注册主方法；若约束集为空则精确回退 A0。
- **A5**：与 A4 候选相同，但选择时忽略非劣约束，仅作 unconstrained 消融。
- **A6**：atom + no-shrink gate，用于检验 shrinkage。

所有 calibration 决策先写入 `calibration/selection.audit.json`，其 SHA-256 再写入 `selection.lock.json`。同一锁定配置跨三个 outer 和三个 model seeds 使用；每个 outer 只重新拟合其 calibration-only 参数模型。

## 3. Calibration 选择结果

| 族 | 锁定候选 | 回退 A0 | 关键配置 |
| --- | --- | --- | --- |
| A0 | A0 | 否 | definition=locked A0 baseline |
| A1 | A0 | 是 | strength=1.0<br>tail_rule=linear<br>definition=exact A0 |
| A2 | A2_atom1 | 否 | atom_strength=1.0<br>tail_strength=0.0<br>dmax=0.0<br>width_delta_cap=0.0<br>width_cap_reference=atom_only |
| A3 | A3_tail1_d0p25_w0p05 | 否 | atom_strength=0.0<br>tail_strength=1.0<br>dmax=0.25<br>width_delta_cap=0.05<br>width_cap_reference=atom_only |
| A4 | A4_atom1_tail1_d0p25_w0p02 | 否 | atom_strength=1.0<br>tail_strength=1.0<br>dmax=0.25<br>width_delta_cap=0.02<br>width_cap_reference=atom_only |
| A5 | A5_atom1_tail1_d0p25_w0p02 | 否 | atom_strength=1.0<br>tail_strength=1.0<br>dmax=0.25<br>width_delta_cap=0.02<br>width_cap_reference=atom_only |
| A6 | A6_atom1_tail1_d0p5_w0p05 | 否 | atom_strength=1.0<br>tail_strength=1.0<br>dmax=0.5<br>width_delta_cap=0.05<br>width_cap_reference=atom_only |

主路径结论：A4 未回退，锁定候选为 `A4_atom1_tail1_d0p25_w0p02`。

## 4. Sealed test 总体结果与 A0–A6 消融

以下均为三个 pooled seed 的算术平均；conditional ACE90 采用 family-equal 聚合。

| 方法 | CRPS | coverage90 | W90 | Winkler90 | conditional ACE90 |
| --- | --- | --- | --- | --- | --- |
| A0 | 0.082321 | 0.822287 | 0.407945 | 0.677668 | 0.07737 |
| A1 | 0.082321 | 0.822287 | 0.407945 | 0.677668 | 0.07737 |
| A2 | 0.08141 | 0.848667 | 0.408191 | 0.669431 | 0.052323 |
| A3 | 0.082312 | 0.823694 | 0.409772 | 0.676952 | 0.075966 |
| A4 | 0.081404 | 0.849972 | 0.409933 | 0.668876 | 0.051045 |
| A5 | 0.081404 | 0.849972 | 0.409933 | 0.668876 | 0.051045 |
| A6 | 0.08133 | 0.860889 | 0.423325 | 0.661063 | 0.040644 |

### A4 相对 A0 的 paired day-bootstrap

| 指标 | 差值方向 | 点估计 | 95% CI |
| --- | --- | --- | --- |
| CRPS | baseline minus method; positive is better | 0.000917 | [0.000619, 0.001242] |
| coverage_90 | absolute baseline coverage error minus absolute method coverage error; positive is better | 0.027685 | [0.021861, 0.034028] |
| width_90 | method minus baseline; positive means wider method intervals | 0.001987 | [0.001086, 0.002869] |
| winkler_90 | baseline minus method; positive is better | 0.008792 | [0.004462, 0.014372] |
| conditional_ACE90 | baseline family-equal ACE minus method family-equal ACE; positive is better | 0.026325 | [0.021208, 0.031408] |

## 5. 每个 outer/seed 的一致性

| outer | seed | CRPS 相对变化 | coverage90 变化 | W90 变化 | conditional ACE90 变化 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | -1.6261% | 3.2417 pp | 0.00028 | -0.03023 |
| 1 | 1 | -1.6085% | 2.825 pp | 0.002025 | -0.027381 |
| 1 | 2 | -1.5993% | 4.2833 pp | 0.002515 | -0.039613 |
| 2 | 0 | -0.9686% | 2.75 pp | 0.002646 | -0.026245 |
| 2 | 1 | -0.6917% | 2.7833 pp | 0.002987 | -0.026587 |
| 2 | 2 | -0.736% | 2.7833 pp | 0.002456 | -0.027067 |
| 3 | 0 | -1.2393% | 1.9333 pp | 0.000872 | -0.01681 |
| 3 | 1 | -0.9501% | 1.95 pp | 0.003113 | -0.016081 |
| 3 | 2 | -0.671% | 2.3667 pp | 0.00099 | -0.02131 |

冻结 outer-consistency 审计结论为 **通过**；三个 outer 中与主方向一致的数量为 **3/3**。该判断还应用了 artifact 中冻结的 catastrophic rule，不能用 pooled 均值替代。

## 6. 条件校准

`conditional_metrics.csv` 共含 7896 个 group-level 记录。pooled A4 中 ACE90 最大的记录属于 `zone_x_spread` / `17`，coverage90=0.753788，ACE90=0.146212，样本 cell 数 n=528。总体 family-equal conditional ACE90 的 A0/A4 点值及 paired CI 已在摘要表中报告；不能以单个最差 group 代替 family-equal 主统计量。

## 7. Boundary atoms 与 finite-M 量化

A4 的解析 atom 诊断按三个 seed 汇总：零事件 Brier=0.052951、log loss=0.187755、平均预测率=0.078388、观察率=0.076；一事件对应 Brier=0.000028、log loss=0.000342、平均预测率=0.00008、观察率=0.000028。可靠性诊断共读取 60 个 seed-bin 记录。

场景数固定为 M=100，所以概率分辨率为 0.01。解析 atom 到 finite ensemble 的零事件量化 MAE=0.002655、跨 seed 最大绝对误差=0.658526；一事件分别为 0.00008 与 0.00009。解析概率与场景中实际 0/1 频率不是同一个对象，尤其不能把稀有上边界事件的 1/M 离散化误差解释成结构概率估计误差。

## 8. Rank、ties 与数值安全

| 方法 | strict reversals | raw ties broken | strict pairs collapsed | stable ordinal rank fraction | central changed | nonfinite |
| --- | --- | --- | --- | --- | --- | --- |
| A0 | 0 | 1449327 | 3306435 | 0.960421 | 0 | 0 |
| A1 | 0 | 0 | 0 | 1 | 0 | 0 |
| A2 | 0 | 5221294 | 8731052 | 0.947408 | 0 | 0 |
| A3 | 0 | 0 | 741 | 0.999959 | 0 | 0 |
| A4 | 0 | 5221098 | 8731685 | 0.947375 | 0 | 0 |
| A5 | 0 | 5221098 | 8731685 | 0.947375 | 0 | 0 |
| A6 | 0 | 5221098 | 8731685 | 0.947375 | 0 | 0 |

全部方法的 strict reversals、tail 引起的 central-value changes 与 nonfinite 计数均为零。`strict pairs collapsed` 可以因 atom/ties 而非零，它表示严格次序被压成 tie，不等同于反序；因此本报告同时给出 ties 与 stable ordinal rank，而不只报一个“无反序”结论。

## 9. SUC 描述性敏感性

| 方法 | 成功求解/paired cases | 平均 realized total cost |
| --- | --- | --- |
| A0 | 9/9 | 386965.76 |
| A4 | 9/9 | 368002.505 |
| A2 | 9/9 | 369157.456 |

SUC 共读取 27 条 paired method-case 记录。其 protocol 明确标为 **descriptive sensitivity only**，`success_gate_role=none`：它不是主/次成功门槛，也没有用于方法选择；这里不据此给出显著性或优越性结论。

## 10. 预注册成功门槛

| 门槛 | 结果 |
| --- | --- |
| CRPS_noninferiority | 通过 |
| MAE_noninferiority | 通过 |
| VS_noninferiority | 通过 |
| conditional_ACE90_reduction | 通过 |
| coverage_90 | 未通过 |
| outer_consistency | 通过 |
| ramp_CRPS_noninferiority | 通过 |
| strict_rank_reversals | 通过 |
| width_90_increase | 通过 |
| winkler_90_noninferiority | 通过 |

合计通过 **9/10**。科学结果是否成功与实验是否完整是两件事；即使门槛未全部通过，也必须完整保留 fallback、消融、CI 与失败门槛。

## 11. 限制与下一步

1. 这是同一数据域内的内部 outer-split confirmation，不是外部确认；下一步应使用未被任何旧研究接触的新年份、新场站或另一公开数据集。
2. 三 outer × 三 seeds 改善了内部稳定性审计，但不能把九个模型重复当成九份独立数据；不确定性以日历日为 cluster。
3. 上边界事件极少，解析 one-atom 结果应保守解释；需要更多真实边界事件才能验证可迁移性。
4. M=100 带来 0.01 的离散概率分辨率；下一步应预注册更大 ensemble 的量化敏感性，而不是事后选择 M。
5. SUC 仅是固定 RTS-24 proxy 下的描述性下游敏感性；需要独立系统、需求和成本设定做外部运营验证。
6. 下一轮确认必须重新训练底模、重新执行 selection lock，并保持 outer test 在锁定前不可访问。

## 12. 可追溯性

- 冻结协议：[CAA_RAHC_FROZEN_PROTOCOL.md](CAA_RAHC_FROZEN_PROTOCOL.md)
- 选择审计：`outputs/caa_rahc/calibration/selection.audit.json`
- 选择锁：`outputs/caa_rahc/selection.lock.json`
- 总体与条件指标：`outputs/caa_rahc/metrics/`
- paired bootstrap、非劣与成功门槛：`outputs/caa_rahc/statistics/`
- 描述性 SUC：`outputs/caa_rahc/suc/`
- 完整文件哈希：`outputs/caa_rahc/experiment_manifest.json`
