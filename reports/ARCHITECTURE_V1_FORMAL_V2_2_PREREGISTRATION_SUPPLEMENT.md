# Architecture v1 formal-v2.2：数值协议修订预注册补充

冻结日期：2026-08-15（Asia/Shanghai）

## 1. 状态与证据等级

formal-v2.1 的权威结论仍为 **`G1_NO_GO`**，不得追溯改判。它失败的两项预注册门是：flow warm-up 后梯度裁剪率为 100%，以及不同 `member_chunk` 下最终连续值没有逐比特一致。v2.2 不是对旧结果换一套标准后重新打分，而是一项只对未来运行生效的新协议。

本修订明确属于 **validation-informed engineering amendment**：`clip=5.0` 候选和跨 chunk 容差由 v2.1 已经观察到的 train/validation 工程证据启发。因此 v2.2 即使通过，也不能被称为独立 validation confirmation；它只能恢复 R0 共同基础的开发期工程可信度。

本版冻结后，如果 P0、G0 或 G1 失败，不得在 v2.2 内移动阈值、换候选或选择性重跑 seed。任何改变都必须形成新的、前瞻性冻结的协议版本，并保留 v2.2 No-Go 证据。

机器可读冻结文件：

- `repro_configs/architecture_v1_formal_v2_2.json`，SHA256 `78b88963e16a680b403afddba6f53dc069c8e2fa43b4f9e09a8e93e87784cf8c`
- `repro_configs/architecture_v1_go_no_go_v2_2.json`，SHA256 `72ba39982a97e3e5914e15c31223118c5f61288029a35f605b7eff2ba98d3bcb`

## 2. 不变的科学协议

除下面第 3–5 节列出的训练与数值完整性修订外，formal-v2 的科学协议全部保留：

- 日期仍为 R-SEEN 314 日、train 267 日、validation 50 日、calibration 50 日、selection 50 日；14 个 smoke 日仍包含在 R-SEEN 中。
- 只有 train 和 validation target 可被正式 R0 进程物化；calibration、selection、R-SEEN 和 final target 均为硬禁止角色。
- 数据日期及各角色 SHA256、缺失值处理、observed-only mask、train-only standardizer 全部不变。
- R0 网络、共享 E/A、atom NLL、interior location auxiliary、upper-one 稀疏支持规则和 E/A 冻结语义不变。
- AdamW、学习率、权重衰减、训练预算、早停、固定 validation flow bank 和 formal seeds `[0,1,2]` 不变；唯一优化器数值变化是 flow gradient clip 从 `1.0` 前瞻性改为候选 `5.0`。
- 正式采样仍为每集合 100 members、sampling seeds `[21000,21001,21002]`、Heun 16 steps / 31 flow NFE，登记的计分 chunk 仍为 10。
- G0、G1 的预测质量界、三 seed dispersion、G2–G6、选择端点、非劣界、负对照、效率界、mask、bootstrap 和 multiplicity 规则全部不变。
- T0、T1-feature、T1-source、T1-shuffle、T2 和 T3 的候选顺序不变；v2.2 通过只允许继续做 validation 开发，不授权打开 selection。

## 3. P0：train-only、三 seed、全部丢弃的梯度预检

P0 必须在任何 v2.2 retained-weight 训练前通过。它只用于验证已经冻结的 `flow clip=5.0` 是否与训练尺度相容，不负责调参。

每个 seed 采用完全独立的新初始化和预检专用随机数流：

1. 只加载 267 个 train 日，validation loader 和 validation bank 均不得构造。
2. 使用与正式 E/A 相同的 loss、AdamW 和 `clip=1.0`，固定训练 40 epochs；不早停、不选 checkpoint，直接取第 40 epoch 终点并冻结 E/A。
3. 新初始化 flow，使用正式 flow 的 loss、AdamW、LR `1e-4`、weight decay `0` 和候选 `clip=5.0`，固定训练 40 epochs。
4. batch 固定为 16 日，因此每 epoch 为 17 个 optimizer updates。flow 共 680 updates；warm-up 为前 30 epochs，即恰好 510 updates；审计窗为第 31–40 epoch，即 updates 511–680，边界均为 1-based inclusive。
5. 每个 optimizer update 在裁剪前计算所有可训练 flow 参数的 global total L2 norm，并保存精确逐更新记录。分位数必须采用 linear interpolation。
6. 三个 seed 的 E/A、flow、optimizer 和 RNG 状态全部丢弃。只保留身份清单、逐 update 梯度、汇总、判定和 SHA256 sidecar；这些权重不得成为正式初始化。

P0 对每个 seed 都要求：

| 条件 | 硬门 |
|---|---:|
| 完成比例 | `1.0` |
| nonfinite / skipped / rollback | 全部 `0` |
| updates 511–680 中 `preclip_norm > 5.0` 的比例 | `≤ 0.25` |
| updates 511–680 的 `p99 / median` | `≤ 5.0` |
| 680 个 flow updates 全程最大 preclip norm | `≤ 50.0` |
| validation/calibration/selection/R-SEEN target access | 全部 `0` |

三个 seed 必须全部通过。任一失败立即停止，不能启动 retained-weight v2.2。

## 4. 正式 flow 梯度门

P0 通过后，formal-v2.2 必须从 seed 0 开始完整重训三个 seed。正式 E/A 和 flow 都从新初始化开始；不得复用 formal-v2.1 的 E/A、flow、optimizer、checkpoint 或 RNG 状态。

正式训练保留原来的 validation checkpoint selection 和早停规则，但 flow 使用已冻结的 `clip=5.0`。逐 optimizer update 的 preclip global total L2 norm 必须持久化。正式 G1 对每个 seed 使用同一固定边界：

- warm-up 是前 510 optimizer updates，对应固定 batch 16 下的前 30 epochs；审计从 update 511 开始；
- post-warm-up clip fraction `≤0.25`；
- post-warm-up `p99/median≤5.0`，分位数方法为 linear；
- 从第一个到最后一个 flow update 的 preclip norm 最大值 `≤50.0`；
- nonfinite、skipped update、silent rollback 和 failure bundle 仍必须为 0。

若硬件 preflight 不能支持 batch 16，应 hard fail 并冻结证据，而不是切换 batch 8 后仍把 30 epochs 写成 510 updates。

## 5. 跨 member chunk 的新判定

v2.1 的“chunk 10 与 20 最终连续值必须逐比特相同”不再作为 v2.2 条件。v2.2 把真正应严格不变的离散语义和允许浮点归约尾差的连续运算分开。

审计覆盖完整 validation 面板：

```text
50 validation days × 3 model seeds × 3 sampling seeds = 450 个配对场景集合
```

每个配对均使用完全相同、显式传入并存档的 atom allocation 和 initial noise；只把 `member_chunk` 从登记值 10 改为审计值 20。两个 chunk 的原始输出及差异摘要都要保留。

严格逐比特门：

- state array 完全相同；
- active mask 完全相同；
- analytic zero/one probabilities 完全相同；
- atom 边界的最终值严格为 0 或 1，并且两种 chunk 完全相同；
- 相同 chunk、相同 seed 的重放仍必须逐比特相同；
- inactive atom latent 与 velocity 仍严格为 0。

连续值门同时要求：

```text
allclose(atol=5e-7, rtol=1e-6) == true
全 450 个配对、所有最终场景值的 global max absolute difference ≤ 1e-6
```

mean 和 p99 absolute difference 必须报告，但不是额外的隐藏门。所有冻结 validation evaluator 输出的 proper-score aggregate 都必须在 CPU float64 下重算；每个 `training seed × sampling seed` aggregate 以及最终 aggregate 的绝对差都必须 `≤1e-7`。coverage90 必须完全相同，atom 指标也必须完全相同。

所有正式计分仍固定使用 `member_chunk=10`。chunk 20 只用于数值审计，不能根据哪个 chunk 分数更好而选择结果。

## 6. 数据封存与允许的判定顺序

calibration 和 selection 在整个 P0、正式 E/A、G0、正式 flow 和 G1 期间保持 sealed：

```text
冻结 v2.2 配置、门、代码和环境
→ 50-update CUDA/显存 preflight，权重丢弃
→ P0 train-only 3-seed 梯度预检，全部权重丢弃
→ 若 P0 全过，从头训练正式 3-seed E/A + flow
→ validation-only G0/G1，包括完整 450 组 chunk 审计
→ G0/G1 全过才允许实现 T0；selection 仍 sealed
```

P0 不能读取 validation target。正式 E/A 的 checkpoint selection、G0 和正式 R0 的早停/G1 仍按原协议使用 validation；这不会授权 calibration 或 selection。calibration 不能用来改 margin，selection 不能因 v2.2 运行而提前打开。

## 7. 如何解释未来结果

- P0 失败：候选裁剪协议不稳定，v2.2 工程 No-Go；不得训练正式权重。
- P0 通过但 G0/G1 失败：R0 共同基础仍不可靠，停止在 R0；不得进入 T0/T1。
- G0/G1 全部通过：只说明 v2.2 在已经用于开发的 validation 上达到预注册工程稳定性，可继续构建公平控制组；不表示 flow 胜过 diffusion/SSM，也不表示外部确认成功。
- 无论结果如何，formal-v2.1 都继续显示原始 `G1_NO_GO`，不能用 v2.2 容差覆盖其历史记录。

最终架构结论仍需一次性 selection 和未来独立外部 final。v2.2 的作用只是把 R0 的训练尺度和采样数值契约修到可以公平比较下一批候选的程度。
