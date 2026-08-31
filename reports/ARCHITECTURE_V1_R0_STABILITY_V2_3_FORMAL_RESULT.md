# Architecture v1：R0 训练种子稳定性 v2.3 正式结果

## 一句话结论

R0 的训练种子稳定性问题已经在当前预注册验证范围内解决。正式候选 **R0-D（共享固定 E/A + 共同训练顺序/路径随机数，只保留 flow 参数初始化差异）** 获得 `D_G1_GO`：三个训练种子的单种子质量门、梯度门、采样语义门和三种子离散度门全部通过。

这一结果只授权下一步实现和冻结 T0 公共骨架；**不授权打开 selection 或 calibration**，也不代表已经选定 flow、diffusion 或状态空间模型中的最终架构。

## 1. 为什么旧 R0 不稳定

旧 R0 的“训练种子”一次改变了多种因素：

1. NWP encoder 与 atom 模块（合称 E/A）的训练结果；
2. flow 参数初始化；
3. 每个 epoch 的日期排列；
4. rectified-flow 训练时使用的时间点与随机路径。

因此，旧三种子差异并不等于单纯的 flow 优化不稳定，而是上游表示、边界状态和连续输运随机性叠加后的总差异。

v2.3 归因 pilot 将这些因素逐层固定：

| 组别 | E/A | flow 初始化 | 日期顺序与路径随机数 | 结果 |
|---|---|---|---|---|
| A：历史方案 | 各 seed 独立 | 独立 | 独立 | level、ramp、joint-ES 均未过稳定门 |
| B | 固定共享 | 独立 | 独立 | level 与 joint-ES 通过，ramp 仅超门槛 `0.00000933` |
| C | 固定共享 | 独立 | 三 seed 共用 | 三项全部通过 |

归因结果表明：

- 独立 E/A 是 level 波动的主要来源；
- 不同日期顺序与随机路径是 ramp 波动的重要来源；
- 在固定上述混杂后，仅保留 flow 初始化差异，R0 本身可以达到预设稳定性。

pilot 不是正式结论，它只用于确定最小修复方案。正式 D 候选随后从头训练三个 flow seed，并独立执行完整验证。

## 2. 正式候选 R0-D

R0-D 采用以下规则：

- 三个 flow seed 共用一个已冻结且哈希锁定的 E/A；
- 三个 seed 使用不同的 flow 参数初始化；
- 三个 seed 使用相同的日期排列计划和 rectified-flow 路径随机数库；
- 训练、早停、模型容量、优化器、batch size、梯度门和质量门保持不变；
- validation 只用于早停与 G1 判定；
- calibration、selection、R-SEEN 和 final 均未载入目标值。

“共同随机数”不是让三个模型变成完全相同的模型，而是把训练数据顺序与 Monte Carlo 路径固定为公平实验条件，留下参数初始化作为被检验的训练种子因素。

## 3. 三个种子的正式结果

| seed | level-CRPS | ramp-CRPS | normalized joint-ES | 梯度裁剪率 | 单种子 G1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.079252 | 0.051883 | 0.110749 | 10.03% | PASS |
| 1 | 0.080583 | 0.052306 | 0.112446 | 8.47% | PASS |
| 2 | 0.080562 | 0.052117 | 0.112344 | 13.63% | PASS |

三个种子均满足：

- level-CRPS、ramp-CRPS、joint-ES 的质量上限；
- coverage 与区间宽度边界；
- atom/zero Brier 要求；
- flow 相对初始验证损失的改善要求；
- 共享 E/A 在 flow 训练期间保持不变；
- 梯度全部有限，裁剪率不超过 25%，梯度尾部比率与最大值均合格；
- 正式采样语义检查通过。

各 seed 的最佳 validation loss 分别为 1.54534、1.55275、1.57518；最佳 epoch 分别为 130、140、150。三个训练均由同一预注册 early-stopping 规则结束。

## 4. 真正关键的三种子稳定性门

| 指标 | 正式 spread | 预注册上限 | 判定 |
|---|---:|---:|---:|
| level-CRPS absolute spread | 0.001331 | 0.002500 | PASS |
| ramp-CRPS absolute spread | 0.000422 | 0.001200 | PASS |
| normalized joint-ES relative spread | 1.510% | 3.000% | PASS |
| coverage90 absolute spread | 0.008921 | 配置上限内 | PASS |
| atom-state Brier spread | 0 | 配置上限内 | PASS |
| zero Brier spread | 0 | 配置上限内 | PASS |

与旧 R0 相比：

- level spread 从 0.003645 降至 0.001331，下降约 63.5%；
- ramp spread 从 0.001464 降至 0.000422，下降约 71.1%；
- joint-ES relative spread 从 3.737% 降至 1.510%，下降约 59.6%。

因此，结论不是“挑到了一个碰巧好的 seed”，而是三个不同 flow 初始化在同一公平随机实验条件下都达到质量门，并且种子间离散度整体通过。

## 5. 采样一致性门

正式审计对每个训练 seed 使用 3 个采样 seed，并在全部 50 个 validation day 上比较 member chunk 为 10 和 20 的结果，共 9 组配对：

- state、active mask、atom probability、zero/one probability、atom 边界值严格一致；
- atom latent 严格为零；
- coverage、zero/one Brier 等离散语义严格一致；
- 连续值全局最大绝对差不超过 `1.1921e-6`，低于预注册上限 `2e-6`；
- level-CRPS、ramp-CRPS、joint-ES 的任一聚合差最大不超过 `1.3160e-10`，远低于 `1e-7` 上限；
- 9/9 正式采样语义门通过。

旧的逐 cell `allclose(rtol=5e-7, atol=1e-6)` 仅作为诊断，7/9 通过。它不再作为正式判据，因为分块导致的浮点归约顺序差异可能使个别 cell 超过极严相对容差，却不改变状态、边界事件或 proper score。这个变更在正式 D 训练前已前瞻性冻结，并没有追溯修改旧 v2.2.1 的 No-Go 判定。

## 6. 数据与证据边界

- `selection_state = sealed`，`selection_target_accessed = false`；
- `calibration_state = sealed`，`calibration_target_accessed = false`；
- 正式结果只使用 train 与 validation；
- 旧诊断数据和 smoke 接触过的日期仍处于 quarantine；
- 本轮结论是“R0 具备进入下一阶段公平架构比较的稳定性”，不是最终论文性能结论。

## 7. 下一步

正式状态允许执行：

1. 冻结 T0 公共外壳及训练计划；
2. 实现参数量匹配的 memoryless T0；
3. 在同一 E/A、mask、采样成员数、训练预算和评价协议下实现 T1-feature、T1-source 与 T1-shuffle；
4. 所有候选代码、配置、分支规则和 Go/No-Go 哈希全部冻结后，才可一次性开启 selection。

当前不需要继续用 EMA、增大 batch 或改 cosine schedule“治疗”R0。那些措施会增加变量和计算量，而最小修复已经正式过门。若 T0/T1 阶段重新出现稳定性问题，再按预注册分支启用额外措施。

## 8. 可复核产物

- 正式结果：`outputs/architecture_v1_r0_stability_v2_3_formal/FORMAL_D_RESULT.json`
- 正式训练冻结文件：`outputs/architecture_v1_r0_stability_v2_3_formal/training.freeze.json`
- 正式配置：`repro_configs/architecture_v1_r0_stability_v2_3_formal.json`
- 正式 runner：`repro_scripts/run_architecture_v1_r0_stability_v2_3_formal.py`
- pilot 结果说明：`reports/ARCHITECTURE_V1_R0_STABILITY_V2_3_PILOT_RESULT.md`

关键 SHA-256：

- `FORMAL_D_RESULT.json`: `c7a614ccc46a7bee1f357bed4113257fee860dc29d7f598aaacee82e1f06529b`
- `training.freeze.json`: `30215d4ca10d30e30cf9c72028d311bf3c5adabb12304cbe3f94147d9be16aca`
- formal config: `8d22badaa1dd76ed98c393dc33c97df5d6c7f49cac8b28a9c7483df19f1928e0`

正式输出目录共有 44 个 payload 与 44 个 SHA sidecar；本轮逐文件重算校验为 44/44 通过，无缺失或哈希不匹配。
