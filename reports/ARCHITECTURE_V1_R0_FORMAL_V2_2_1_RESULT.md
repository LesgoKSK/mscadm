# Architecture-v1 R0 formal-v2.2.1 结果

运行完成时间：2026-08-19（Asia/Shanghai）  
最终状态：**G1_NO_GO**  
下一步：**停止在 R0，不进入 T0，不打开 calibration/selection。**

## 1. 为什么有 v2.2.1

formal-v2.2 在 seed 0 完成 E/A、尚未进行首个 retained flow optimizer update 时，runner 因读取不存在的门字段 `preclip_gradient_norm_max` 而中止。v2.2.1 只把该读取修正为已经冻结的字段 `preclip_gradient_norm_max_over_all_flow_updates`；数据、模型、优化器、随机种子、指标和所有阈值均未改变。三个种子随后全部从全新初始化重新运行。

## 2. 训练前门检

train-only P0 三种子全部通过，所有临时权重均丢弃：

| seed | 审计窗裁剪率 | p99/median | 全程最大梯度范数 | P0 |
|---:|---:|---:|---:|:---:|
| 0 | 5.88% | 2.377 | 29.230 | PASS |
| 1 | 7.65% | 2.237 | 32.927 | PASS |
| 2 | 2.94% | 2.095 | 32.976 | PASS |

门槛分别为裁剪率不超过 25%、p99/median 不超过 5、全程最大梯度范数不超过 50。

CUDA 容量预检也通过：batch=16，50 次丢弃权重更新约 7.01 秒，峰值显存 192,818,688 bytes（约 184 MiB），低于 4.5 GiB 上限。

## 3. 正式训练

三个种子的 E/A G0 均通过；flow 均正常早停并通过正式梯度审计。

| seed | flow epoch 数 | 最佳 epoch | 最佳固定 validation loss | 裁剪率 | p99/median | 全程最大范数 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 250 | 129 | 1.54534 | 10.03% | 1.954 | 29.161 |
| 1 | 270 | 149 | 1.57413 | 8.82% | 1.901 | 28.853 |
| 2 | 270 | 149 | 1.51487 | 13.55% | 1.917 | 30.317 |

因此，formal-v2.1 的“所有更新都被裁剪”问题已经被 gradient clip=5.0 修复。

## 4. Validation 质量指标

| seed | level CRPS | ramp CRPS | normalized joint ES | coverage90 | width90 | zero Brier | atom-state Brier |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.07925 | 0.05188 | 0.11075 | 0.85396 | 0.39134 | 0.05970 | 0.11974 |
| 1 | 0.08185 | 0.05276 | 0.11445 | 0.87591 | 0.41693 | 0.06165 | 0.12363 |
| 2 | 0.07820 | 0.05129 | 0.11031 | 0.86407 | 0.38519 | 0.05995 | 0.12023 |

每个种子的绝对质量门都通过，但三种子离散度门未全部通过：

| 指标 | 实际 spread | 门槛 | 结果 |
|---|---:|---:|:---:|
| level CRPS | 0.003645 | 0.002500 | FAIL |
| ramp CRPS | 0.001464 | 0.001200 | FAIL |
| normalized joint ES（相对 spread） | 3.737% | 3.000% | FAIL |
| coverage90 | 0.021946 | 0.035000 | PASS |
| zero Brier | 0.001949 | 0.006000 | PASS |
| atom-state Brier | 0.003898 | 0.012000 | PASS |

这说明 R0 的平均水平可用，但训练种子稳定性尚不足以成为后续架构比较的可靠共同基线。

## 5. Member-chunk 审计

对每个训练 seed、三个采样 seed、全部 50 个 validation 日，使用相同 atom allocation 和相同初始噪声，比较 member chunk=10 与 20。

- 9 组中 4 组通过、5 组失败。
- 所有组的 state、active mask、atom 边界值、解析概率与重复主采样均严格一致。
- 9 组的 proper-score 最大绝对差不超过 `1.32e-10`，远低于 `1e-7` 门槛。
- 连续值全局最大差为 `1.1920929e-6`；一组超过 `1e-6` 上限。
- 其余失败来自近零单元未满足预锁 `allclose(atol=5e-7, rtol=1e-6)`，即使对应全局最大差仍小于 `1e-6`。

科学上，这些差异是极小的 CUDA 浮点分块舍入误差，未改变 atom 语义或评分；但按预注册门，仍必须记为 sampling-semantics FAIL，不能事后移动阈值。

## 6. 最终裁决

最终 `G1_NO_GO` 有两个独立原因：

1. 三个训练 seed 的 sampling-semantics 均未做到三组采样种子全部过门；
2. 三种子在 level CRPS、ramp CRPS、normalized joint ES 上的离散度超出预锁上限。

因此：

- R0 训练产物已冻结，但“冻结”只表示不可继续改写，不表示模型通过；
- `selection_authorized=false`；
- calibration 与 selection 均保持 sealed，访问标志均为 false；
- 当前不得进入 T0、T1-feature、T1-source 或 T1-shuffle。

如果继续研究，需要另立新的前瞻性修订。优先解决的是三种子稳定性；member-chunk 审计应改为以 exact state/probability、全局绝对误差和 proper-score 等价为科学门，把逐单元 `allclose` 降为诊断项，但不能用于追溯改判本次结果。

## 7. 关键产物与完整性

- 最终结果：`outputs/architecture_v1_formal_v2_2_1/R0_FORMAL_RESULT.json`
- 训练冻结：`outputs/architecture_v1_formal_v2_2_1/r0_training.freeze.json`
- P0 结果：`outputs/architecture_v1_formal_v2_2_1/P0_gradient_scale_preflight/P0_RESULT.json`
- CUDA 预检：`outputs/architecture_v1_formal_v2_2_1/preflight/preflight.json`
- 修订记录：`reports/ARCHITECTURE_V1_FORMAL_V2_2_1_AMENDMENT.md`

最终结果 SHA-256：`f9b6532ba0a784330f352570f5bd6a1ea65df9ce1c73b6c62d07a6f92b2da671`  
训练冻结 SHA-256：`422fa57c63c6852d094bca36d4936a8f2b33835329d34cd166f84b9456d00b8f`

输出目录内 68 个 payload 均有对应 SHA-256 sidecar，逐项复核 68/68 通过，无缺失、孤儿或哈希不一致。

