# Architecture-v1 family-v1.2 Temporal Utility Probe 协议

冻结日期：2026-08-31

文档状态：本文件记录实验启动前冻结的协议；48 次 retained adapter training 与正式 validation evaluation 已于 2026-09-01 完成，正式结论见 `ARCHITECTURE_V1_FAMILY_V1_2_FORMAL_EVALUATION_RESULT.md`。

证据范围：train/validation 阶段的机制发现；selection、calibration 和最终外部测试继续封存。

## 1. 一句话说明

本实验冻结三个已经训练完成的 D0-v Diffusion，只增加一个很小的时间 residual adapter，检验：

> 在已经具有小时位置编码和全局时间注意力的 D0-v 上，按照正确小时顺序传播信息所产生的增量价值，是否取决于 DDIM 去噪阶段和 NWP 天气动态程度？

这不是新的 Flow-versus-Diffusion 胜负实验。family-v1.1 的正式结论仍然是两者各有优势、没有全面胜者。D0-v 仅因为具有明确的去噪阶段坐标而被用作机制研究平台。

## 2. 为什么需要这个 Probe

已有结果形成了连续证据链：

1. 跨模型诊断表明，主要缺口集中在 ramp、总变差和 lag-1 时间依赖，而不是总体功率水平。
2. v3.2 表明 Chronological recurrence 明显优于 Shuffle，说明正确的前后小时邻接确实有可利用信号。
3. 同一批候选又因为 level/joint 非劣性或跨种子稳定性失败，说明全天气、全生成阶段持续启用时间机制并不可靠。
4. family-v1.1 中 Flow 与 D0-v 没有正式胜者，但 D0-v 提供了可直接干预的 31-step DDIM 去噪轨迹。

因此，下一步不是直接实现完整双门控模型，而是先定位“有用的时间干预”出现在哪里。

## 3. Probe 测量什么，不测量什么

### 3.1 正式 estimand

每个阶段块测量的是：

> 只在该 DDIM 阶段块启用正确顺序 residual 后，最终生成场景相对于 Shuffle 和原始 D0-v 的变化。

### 3.2 解释边界

- D0-v 已经包含时间注意力和小时位置编码，因此本实验测量的是“有方向的顺序传播”的增量价值，不是笼统的“时间信息是否有用”。
- DDIM 是顺序系统。早期干预会改变后续所有 latent，因此一个块的结果是该干预的最终下游效应，不是可独立相加的局部贡献。
- 八个块的结果不能求和，也不能自动外推到其他采样步数或其他 diffusion schedule。
- Validation 用于机制发现；它不能替代仍封存的 selection、calibration 或最终外部确认。

## 4. 模型结构

```text
                                ┌── 冻结 Atom ──> 状态概率与共同 allocation
NWP ──> 冻结 Encoder ──> encoded ┤
                                └── 冻结 baseline context c0
                                              │
                            ┌─────────────────┴─────────────────┐
                            │ TemporalContextResidualAdapter    │
                            │ 正确顺序或注册的 Shuffle 顺序       │
                            └─────────────────┬─────────────────┘
                                              │ Δc
                  非目标阶段：c(t)=c0          │         目标阶段：c(t)=c0+Δc
                                              ↓
                       冻结 D0-v flow(z_t, t, c(t), mask)
                                              ↓
                                      31-step DDIM 场景
```

形式化定义为：

\[
c_0=C_{\mathrm{frozen}}(E_{\mathrm{frozen}}(NWP)),
\]

\[
\Delta c_\phi=\tanh\!\left(W_{out}\,h_{\mathrm{GRU}}(c_0;\,order)\right),
\]

\[
c_t=c_0+I_b(t)\Delta c_\phi,
\]

\[
\hat v=F_{\mathrm{frozen}}(z_t,t,c_t,mask).
\]

其中 `W_out` 零初始化。因此 P0 开始时 `Δc=0`，完整 Probe 必须严格退化为原 D0-v。

### 4.1 Adapter 固定结构

- 输入：冻结的 64 维 baseline context；
- `LayerNorm(64)`；
- 单向 `GRUCell(64→32)`；
- `LayerNorm(32)`；
- `Linear(32→64)`；
- `tanh` 有界 residual；
- 固定 residual scale 为 1；
- 总参数量 11,712；
- 不学习 SNR gate，不学习 NWP gate；
- 不修改 Atom 分支。

Adapter 只计算一次；DDIM 循环内仅根据当前 timestep 在 `c0` 与 `c0+Δc` 之间切换。

### 4.2 冻结与梯度语义

三个 D0-v best EMA 的全部参数和 buffer 都必须哈希锁定，且 `requires_grad=False`。但是 frozen flow 不能放进 `torch.no_grad()`：

- 对 flow 参数的梯度必须为零或不存在；
- 从 loss 经过 flow 的 context 输入到 adapter 的梯度必须存在。

这保证原模型不更新，同时 adapter 仍可学习如何轻微修改 context。

## 5. 八个 DDIM-stage blocks

正式 sampler 为 cosine-250、31-step deterministic DDIM。实验单位不是八个等宽 log-SNR 区间，而是沿真实 reverse grid 排列的八个连续阶段块。

| Block | Reverse timestep | Log-SNR 范围 | 解释 |
|---|---|---:|---|
| B1 | 249, 241, 232, 224 | -17.0633 ～ -3.7015 | 起始高噪声阶段 |
| B2 | 216, 208, 199, 191 | -3.1340 ～ -1.9447 | 高噪声阶段 |
| B3 | 183, 174, 166, 158 | -1.6587 ～ -0.9015 | 低中段 |
| B4 | 149, 141, 133, 124 | -0.6597 ～ -0.0246 | 中段负 log-SNR |
| B5 | 116, 108, 100, 91 | 0.1751 ～ 0.8204 | 中段正 log-SNR |
| B6 | 83, 75, 66, 58 | 1.0420 ～ 1.8353 | 清晰化阶段 |
| B7 | 50, 42, 33, 25 | 2.1419 ～ 3.4760 | 低噪声阶段 |
| B8 | 17, 8, 0 | 4.1639 ～ 8.5462 | 最终清晰阶段 |

每个 adapter 训练时只读取所属块中的真实 DDIM timestep。正式训练为 120 个完整 train epoch，共 2,040 次更新和 32,040 个 calendar-day timestep exposure；该总数同时被 3 和 4 整除，因此每个块内的 timestep 得到完全相同的 exposure 数。

## 6. Chronological 与 Shuffle

### 6.1 Chronological

GRU 按物理小时 `0→1→…→23` 扫描。

### 6.2 Shuffle

Shuffle 只改变 GRU 的访问顺序，输出随后 scatter 回原物理小时位置。NWP 数值、hour embedding、张量位置和目标都不改变。

协议注册八个 permutation：

- 前六个用于 Shuffle adapter 训练；
- 后两个从训练中留出，只用于同 checkpoint 推理干预；
- 每个 permutation 都是 derangement；
- 任意连续访问的两个小时都不是真实相邻小时，正向和反向相邻均被破坏；
- 每次训练更新使用由 backbone seed、block index 和 update index 决定的可复现 permutation。

### 6.3 Same-checkpoint control

每个 Chronological adapter 在评估时还要保持完全相同的参数，只把 GRU 顺序切换为两个 inference-only Shuffle permutation。这回答：

> 同一个已经学习好的 mechanism 在破坏真实邻接后，收益是否立即消失？

## 7. 48 次 retained adapter training

```text
3 个 D0-v best EMA backbone
× 8 个 DDIM-stage block
× 2 种 order mechanism
= 48 次 adapter-only training
```

Chronological/Shuffle 成对共享：

- 同一个 D0-v checkpoint；
- 完全相同的 adapter 初始参数；
- calendar-day minibatch 顺序；
- 每一天的 diffusion timestep；
- Gaussian forward noise；
- optimizer、EMA 与更新次数。

唯一计划内区别是 recurrent visitation order。

### 7.1 训练策略

- 只使用 train targets；
- AdamW，学习率 `1e-4`；
- batch 为 16 个 calendar days；
- gradient clip 为 5；
- adapter EMA decay 为 0.999；
- 每个 run 固定 2,040 次更新；
- 不进行 validation-based early stopping；
- 不挑选 best checkpoint；
- retained 权重是第 2,040 次更新后的 final adapter EMA。

这避免 48 个 run 分别利用相同 validation 选择不同 checkpoint。

## 8. NWP dynamicity

NWP 不增加训练轴。P0 只用 train NWP 建立并哈希冻结 dynamicity registry：

1. `rms_dws100`：100m 风速逐小时变化；
2. `weighted_turn100`：按风速加权的逐小时转向；
3. `spatial_dispersion`：区域间风速差异。

三个特征分别使用 train mean/std 标准化，再等权平均为 dynamicity score。train score 的三分位点定义：

- stable；
- moderate；
- dynamic。

整个过程不读取目标功率。Validation 的 regime label 只能在 48 次训练全部冻结后，由这个 train-only registry 分配。

## 9. 数据 provenance amendment

第一次 P0 在任何 optimizer update 发生前按预期 fail-closed：当前数据协议哈希与 D0-v 训练时的旧哈希不同。核查表明，变化来自一个 R-SEEN 来源清单后来补充了审计元数据；协议实际消费的 150 个 `outer_test_dates` 及其 union hash 均未改变。

不能简单忽略这个差异。仓库因此新增一个独立哈希冻结的 amendment，并要求运行器完成以下等价性证明：

1. 当前来源文件、schema、日期数量和日期 union hash 均与 amendment 一致；
2. 当前 train split 的完整数组指纹与 D0-v P0 冻结记录完全一致；
3. 在内存中的 manifest 副本里，只把这一条审计文件 SHA 投影回旧值，不修改任何日期、数组或标准化统计；
4. 重新计算后，必须同时精确复原旧 `protocol_sha256`、旧 train-only bundle SHA 和旧 fit bundle SHA；
5. 任一检查失败都停止 P0，并要求建立新的 amendment，不能自动放宽。

这意味着 adapter 看到的训练数据与 D0-v 原训练数据完全相同；被桥接的是 provenance 元数据身份，不是数据内容或实验角色。原始 `family-v1.2` 配置保持不变，amendment 单独留痕。

## 10. 正式比较

每个 block 必须同时报告三组对照：

| 对照 | 回答的问题 |
|---|---|
| Chronological − Shuffle | 收益是否来自正确时间邻接 |
| Chronological − D0-v | 时间干预是否产生实际净收益 |
| Shuffle − D0-v | 收益是否只是额外参数、平滑或一般 context perturbation |

主要指标：

- ramp CRPS；
- lagged increment variogram score。

分布安全指标：

- level CRPS；
- normalized joint ES；
- coverage90；
- width90。

Atom 指标继续报告，但按协议应保持完全一致，因为 E/A 和 allocation 均冻结。

## 11. 统计协议

推断单位为 calendar day。先在同一天内平均三个 backbone seed，再形成正式 contrast。

由于存在八个 block、多个 regime 和多个指标，不能逐格使用普通 95% CI 后挑选显著结果。正式顺序为：

1. 检验八个 block 的 Chronological−Shuffle 效应是否同质；
2. 检验 stable/moderate/dynamic 效应是否同质；
3. 检验 block×NWP dynamicity interaction；
4. 对八个 block 的两个主要指标使用 calendar-day paired joint bootstrap 和 max-T simultaneous bands；
5. 只有通过相应 omnibus 与同时推断控制后，才解释单个 block/regime。

实际门槛沿用已注册的 architecture-v1 量级：

- ramp CRPS 实际改善至少 0.001；
- lagged variogram 相对改善至少 5%；
- level CRPS 恶化不超过 0.0015；
- joint ES 相对恶化不超过 2%；
- coverage 绝对差不超过 0.02；
- width 相对增幅不超过 10%。

阶段结构至少需要两个相邻 block 得到支持，不能从八个结果里只挑一个孤立最好点。

## 12. P0 启动门

P0 使用 train targets，Chronological 与 Shuffle 各执行 50 次随后丢弃的更新。正式 48 次训练之前，必须同时通过：

- adapter 零初始化时，完整 31-step scenario 与原 D0-v 一致；
- 三个 D0-v checkpoint 文件哈希与 best EMA payload 均有效；
- backbone 训练前后 tensor hash 完全不变；
- optimizer 和 EMA 只包含 adapter 参数；
- frozen flow 参数无梯度，但 adapter 的 context 梯度存在；
- context residual 只在注册 block 的 timestep 启用；
- Atom probability、allocation 和 exact-boundary 语义不变；
- Chronological/Shuffle 随机变量成对一致；
- permutation bank 及其 adjacency destruction 通过；
- 两种 order 的有效更新和 timestep exposure 完全一致；
- checkpoint 恢复后下一次更新精确复现；
- same-checkpoint order switch 可执行且产生有限 residual；
- 峰值显存不超过 4.5 GiB；
- validation、calibration、selection、R-SEEN、final target 均未 materialize；
- P0 目录中不保留任何权重文件。
- provenance amendment 的当前身份、旧身份和 in-memory 等价投影全部通过。

## 13. P0 实际结果

2026-08-31 的正式 P0 结果为 `P0_GO`，结果文件 SHA256 为 `c8b5ab0684e4303328186fe9a0b65f73fcd2e18923d6cc395bdd222dc24e12d2`。主要审计结果如下：

- Chronological 与 Shuffle 各完成 50 次 train-only、随后丢弃的 adapter update；
- adapter 零初始化时，完整 31-step DDIM 的场景、state、interior latent 和 atom probabilities 与原 D0-v 逐张量一致；
- 三个 D0-v best EMA checkpoint 全部通过文件与 tensor hash 校验；
- 两个 probe 的 backbone 在训练前后 hash 不变；
- adapter context 梯度存在，optimizer 与 EMA 均只包含 adapter；
- checkpoint resume 后的下一次指标、adapter、EMA 与 optimizer state 精确复现；
- same-checkpoint held-out Shuffle 会改变 residual，且 residual 有限；
- 峰值 GPU memory 为 159,972,352 bytes，约 0.15 GiB，低于 4.5 GiB 门限；
- 只 materialize `train` targets，其他角色均未访问；
- P0 输出目录只包含结果、NWP registry 及 SHA256 sidecar，没有 `.pt` 或 `.pth` 权重。

因此，协议层面已经允许下一步启动 48 次 retained adapter training；该授权不等于已经训练，也不构成任何阶段、门控或模型优劣结论。

任一项失败都要求新 revision；不能修改当前阈值后原地继续。

## 12. 结果对应的下一步

| Probe 结果 | family-v1.3 决策 |
|---|---|
| Chronological 与 Shuffle 等效 | ordered recurrence No-Go |
| SNR 和 NWP 都无异质性 | temporal gating 路线 No-Go |
| 只有阶段异质性 | SNR/DDIM-stage gate |
| 只有天气异质性 | NWP gate |
| 两个主效应存在但 interaction 弱 | additive controls，不强调 dual gate |
| block×NWP interaction 明确 | dual-gated temporal context residual |
| 时间指标改善但分布损害持续超门 | No-Go，或建立新的固定干预强度 revision |

## 13. 执行边界

只读审计：

```bash
python repro_scripts/run_architecture_v1_family_v1_2_probe.py
```

执行 P0：

```bash
python repro_scripts/run_architecture_v1_family_v1_2_probe.py --execute-p0
```

只有 `P0_GO` 才授权 retained adapter training：

```bash
python repro_scripts/run_architecture_v1_family_v1_2_probe.py --execute-training
```

中断后只能从注册 checkpoint 更新边界恢复：

```bash
python repro_scripts/run_architecture_v1_family_v1_2_probe.py --execute-training --resume
```

本 runner 不生成 validation scenarios。48/48 run 完成、训练 gate 全部通过并写入 `training.freeze.json` 后，才能建立单独的正式评估 runner。

## 14. 权威文件

- `repro_configs/architecture_v1_family_v1_2_probe.json`：机器可执行的冻结协议；
- `architecture_v1/family_v1_2_probe.py`：adapter、stage-aware loss/sampler、EMA trainer 和 NWP registry；
- `repro_scripts/run_architecture_v1_family_v1_2_probe.py`：P0 与 48-run train-only runner；
- `tests/test_architecture_v1_family_v1_2_probe.py`：数值与语义测试；
- `tests/test_architecture_v1_family_v1_2_protocol.py`：协议、lineage 和封存边界测试。
