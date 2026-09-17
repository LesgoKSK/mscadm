# 当前研究进度（唯一维护入口）

最后更新：2026-09-17

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
G0-B 冻结前理论审计
平均 log-SNR 不是信息预算；裸 Bayes 最优解会产生硬 mode cutoff
                │
                ▼
G0-B0 schedule-only 可行性审计
reserve-then-water-fill 全部冻结门通过，可复现且不访问 raw target
                │
                ▼
G0-B tiny-denoiser utility protocol
六路径、三档数据量、共同网络、指标和 Go/No-Go 已冻结
                │
                ▼
G0-B tiny-denoiser P0
六路径 × 50 updates 全部工程门通过，权重与临时 checkpoint 已删除
                │
                ▼
G0-B retained training
324/324 final EMA 完成并冻结，0 failure、0 residual resume
                │
                ▼
G0-B common outer-test evaluation
PA-RWF 相对 IID 的 BRR 仅改善 0.64%，不胜 Fixed/Shuffle，oracle efficiency 更差
                │
                ▼
当前状态：G0_B_STRUCTURED_SCHEDULE_NO_GO；不授权完整 PA diffusion
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
| G0-B0 | G0-A 方差能否变成预算匹配、无硬 cutoff 的 mode-wise schedule | `eta=0.5` 全部工程门通过；只证明 schedule 可执行，不证明 denoiser utility |
| G0-B P0 | 六条路径能否在严格配对、可恢复且不留权重的条件下执行 | 6×50 updates 与全部 hard gates 通过；只授权正式训练，不是路径优劣证据 |
| G0-B formal | NWP predictability-aligned schedule 是否实质降低 held-out denoising 难度 | 324/324 训练与共同评估完成；PA 对 IID 只有 0.64% BRR 改善、未过 2% 门，且不胜 Fixed/Shuffle，正式 `G0_B_STRUCTURED_SCHEDULE_NO_GO` |

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

## 5. 已完成检验的研究问题

经过 CW-Diff、MuLAN、MA-TSD 等近邻复审，以下表述已经放弃：

- “NWP 条件协方差是一种新的 forward process”；
- “由条件决定不同 mode 的 diffusion clock 是通用方法首创”；
- “标准 iid diffusion 在表达能力上无法学习时间依赖”。

本轮冻结并完成了下面这个可证伪的风电问题：

> 在总 corruption budget 匹配的条件下，利用 NWP 可预测的模式级条件不确定性来分配 diffusion SNR，是否能比标准 IID、条件白化、固定非各向同性日程和通用 learned adaptive schedule 更有效地降低风电联合条件分布的有限样本去噪复杂度？

这里的候选贡献不是发明 conditional multidimensional schedule，而是检验：

```text
NWP
  ↓
模式级条件可预测性 / 不确定性
  ↓
显式、受约束的 corruption allocation 原则
```

完整证据链现已闭合：G0-A 支持“NWP 能预测模式级条件不确定性”，G0-B0 支持“该信息可被无退化地翻译成预算匹配 schedule”，但 G0-B 正式结果否定了“这种分配会带来预注册幅度的有限模型学习收益”。因此前两道必要条件成立，并没有推出最终方法有效；这条 predictability-aligned schedule 主线在 tiny-denoiser 层面正式停止。

## 6. G0-A → G0-B 证据链已经完成

完整问题被拆成两个先后门：

```text
G0-A：NWP 能否预测六类时空成分的条件不确定性？
  └─ 已完成：Go

G0-B0：能否构造预算匹配且无硬 cutoff 的 schedule？
  └─ 已完成：工程可行性 Go

G0-B：这种 schedule 是否真的降低有限样本 denoising 难度？
  ├─ train-only discarded-weight P0：Go
  ├─ retained training：324/324 完成并冻结
  └─ common outer-test evaluation：Structured Schedule No-Go
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

这说明 NWP 确实包含模式级条件不确定性信息；G0-A 本身不说明用它控制 diffusion 会改善最终场景，而后续 G0-B 已正式表明当前 schedule-allocation 实现没有达到实质收益门。冻结协议见 [`ARCHITECTURE_V1_G0_A_PREDICTABILITY_PROTOCOL.md`](ARCHITECTURE_V1_G0_A_PREDICTABILITY_PROTOCOL.md)，完整结果见 [`ARCHITECTURE_V1_G0_A_PREDICTABILITY_RESULT.md`](ARCHITECTURE_V1_G0_A_PREDICTABILITY_RESULT.md)。

## 7. G0-B0 理论与正式可行性审计

G0-A 之后提出的两段式逻辑现已全部检验：

```text
已完成：NWP → mode-specific conditional uncertainty
已完成但 No-Go：conditional uncertainty → 预注册幅度的 diffusion-path 学习收益
```

但不能直接冻结下面这个经验式：

\[
\gamma_k(t,c)=\gamma_0(t)+\lambda(t)
\left(\log v_k(c)-\sum_jw_j\log v_j(c)\right).
\]

中心化只能保证平均 nominal log-SNR 相同，不能说明该式是 Bayes-risk 最优；`lambda(t)` 仍是未解释的自由强度。更重要的是，平均 log-SNR 不是 Shannon information budget。

对简化条件 Gaussian mode：

\[
z_k\mid c\sim\mathcal N(0,v_k),\qquad
y_k=\alpha_k z_k+\sigma_k\epsilon,\qquad
\rho_k=\alpha_k^2/\sigma_k^2,
\]

其 posterior variance 与 mutual information 分别为：

\[
D_k=\operatorname{Var}(z_k\mid y_k,c)
=\frac{v_k}{1+v_k\rho_k},
\qquad
I_k=\frac12\log(1+v_k\rho_k).
\]

因此 `D_k=v_k exp(-2I_k)`。若把真正的信息预算定义为 `sum_k w_k I_k=B`，则下面的问题是凸的：

\[
\min_{I_k\ge0}\sum_kw_kv_k e^{-2I_k}
\quad\text{s.t.}\quad
\sum_kw_kI_k=B.
\]

KKT 解是经典 reverse water-filling：

\[
I_k^*=\frac12\left[\log\frac{v_k}{\theta}\right]_+,
\qquad
\rho_k^*=\left[\frac1\theta-\frac1{v_k}\right]_+,
\]

其中 `theta` 由预算等式唯一确定。活跃 mode 的 posterior variance 被压到共同水位 `theta`；`v_k <= theta` 的 mode 不再分配信息。若使用该理论参照，最自然的 `w_k` 是每组有效维数占比，而不是六组等权 macro 指标；换用其他权重等价于另行改变研究目标。

这条推导说明“高条件方差 mode 获得更多 signal”可以有明确依据，但也暴露出两个不能跳过的问题：

1. 它匹配的是 Gaussian mutual information，不是此前提出的平均 nominal log-SNR；两种公平合同不能混写。
2. 它本来就以最小化 Gaussian reconstruction MSE 为目标，因此若 G0-B 只比较 raw reconstruction MSE，会形成部分循环论证。至少还要报告 Gaussian oracle risk、模型相对 oracle 的 excess risk 和不同数据量下的 learning curve。

使用 G0-A 的 267 天 out-of-fold `N1_variance`、逐日有效秩权重和现有 cosine-250 网格进行纯 schedule 审计后，裸 reverse water-filling 出现明显硬 cutoff：

| 现有 cosine-250 位置 | 平均活跃 mode 数 |
|---|---:|
| log-SNR `8.55` | 6.00 |
| log-SNR `0.77` | 5.94 |
| log-SNR `-0.03` | 4.94 |
| log-SNR `-1.76` | 3.20 |
| log-SNR `-3.20` | 2.03 |
| log-SNR `-17.06` | 1.00 |

在全部 `day × diffusion-step` 上，`high_common` 和 `high_local` 仅分别有 52.9% 与 47.3% 的位置获得非零信息；末端只剩 `low_common`。这可能是合理的 coarse-to-fine 行为，也可能过早删除与 ramp 有关的高频证据，目前不能靠直觉裁决。

目前最合适的非退化候选是 **reserve-then-water-fill**。先为每个 mode 保留其 IID Gaussian information 的比例 `eta`：

\[
I_{k,\mathrm{res}}=\eta I_{k,\mathrm{IID}},
\]

再只对剩余预算做 reverse water-filling。令

\[
a_k=v_k\exp(-2I_{k,\mathrm{res}}),
\]

则剩余分配为：

\[
J_k^*=\frac12\left[\log\frac{a_k}{\theta}\right]_+,
\qquad
I_k^*=I_{k,\mathrm{res}}+J_k^*.
\]

`eta=1` 精确退化为 IID，`eta=0` 精确退化为裸 reverse water-filling；任意 `eta>0` 都为每个 mode 保留非零 baseline information。将 `eta=0.5` 作为“不调结果、保留一半并重分配一半”的 G0-B0 主候选后，冻结 runner 在 267 天 OOF 方差与 cosine-250 网格上的正式审计得到：

- 每个 `day × step` 的总 Gaussian mutual-information budget 与 IID 最大绝对误差为 `1.78e-15`；
- 398,898 个逐 mode 相邻步单调性检查为 `0` 次违规；
- 所有 SNR 有限且严格为正；
- 每个 mode 至少保留其 IID information 的 50%；
- 相对 IID 的 log-SNR shift 全范围为 `[-0.862, 2.288]`；
- `eta=1` 的 IID `alpha_bar` identity 最大误差为 `2.22e-16`；
- 当六个 `v_k` 相等时，各 `eta` 相对 IID 的 `alpha_bar` 最大误差为 `6.29e-15`。

全部冻结门通过，正式状态为 `G0_B0_SCHEDULE_FEASIBILITY_GO`。这说明 `eta=0.5` 通过了 schedule-level feasibility，并在当时获准进入下一道 Probe；后续 G0-B 已证明它没有通过真实 denoiser utility 的实质门，因此不能再被视为待实现的方法候选。冻结协议见 [`ARCHITECTURE_V1_G0_B0_SCHEDULE_PROTOCOL.md`](ARCHITECTURE_V1_G0_B0_SCHEDULE_PROTOCOL.md)，正式结果见 [`ARCHITECTURE_V1_G0_B0_SCHEDULE_RESULT.md`](ARCHITECTURE_V1_G0_B0_SCHEDULE_RESULT.md)。

因此本轮已冻结并执行以下边界，但未冻结最终模型：

- 拒绝无推导的 centered-log-variance 线性规则；
- 将 mutual-information-matched reverse water-filling 保留为理论 oracle/reference，而不是直接作为正式方法；
- G0-B0 以 `eta=0.5` reserve-then-water-fill 为主候选，并以 `eta=0/0.25/0.75/1` 作为只解释强度与退化边界的 schedule sensitivity；
- 所有约束强度只能通过 train 内层 fold 或纯数值安全规则确定，不能查看 validation；
- G0-B0 没有读取 raw target、训练 denoiser 或构造 validation bank；
- G0-B 的 oracle-adjusted 指标、六条路径、三档数据量和 Go/No-Go 已另行冻结；G0-B0 Go 本身仍不能替代 denoiser 证据。

新颖性边界也进一步收紧：MuLAN 已覆盖 input-conditional multivariate learned schedule，CW-Gen 已覆盖 conditional mean/covariance whitening，已有 spectral diffusion 工作从 Gaussian covariance eigenvalues 设计 schedule，MA-TSD 已覆盖 time-series non-isotropic frequency transition。因此未来可主张的只能是“风电 NWP 条件可预测性证据 + 明确受约束的 allocation principle + 因果反事实验证”，不能把 Gaussian water-filling 或 spectral scheduling 本身表述为首创。

## 8. G0-B tiny-denoiser：正式结果为 Structured Schedule No-Go

G0-B 不直接训练完整 D0-v，而是在六个 G0-A outer folds 内，用固定 56,058 参数 joint MLP 测试真实 residual 的 out-of-fold reconstruction：

| Path | 回答的问题 |
|---|---|
| IID | 标准统一 nominal schedule 的基线 |
| CW-group | 六组条件白化是否已经足够 |
| Fixed-band | 是否只需要不随天气变化的静态频带优先级 |
| MuLAN-lite | generic learned conditional schedule 是否已经足够 |
| PA-RWF | 正确 NWP variance 驱动的主候选 |
| PA-Shuffle | 只破坏当天 NWP variance 与 schedule 的对应关系 |

公平性合同为：

- 每条路径逐日逐 timestep 匹配同一个 Gaussian mutual-information budget；
- 所有 denoiser 都看到正确的 residual、mask、NWP summary 和 N1 variance；
- Fixed、MuLAN-lite、PA-RWF 与 Shuffle 共享 `eta=0.5` 安全 scaffold，只改变 schedule 使用的 proxy variance；
- 同 fold/fraction/seed 的网络初值、minibatch、timestep、Gaussian noise 和更新次数配对；
- CW-group 与 MuLAN-lite 是本 Probe 的容量受控版本，不冒充完整论文复现。

Sample-efficiency 设计固定为 25%/50%/100% 三档 denoiser 日期，nuisance estimators 在完整 outer-train 上拟合后跨三档冻结。正式矩阵为：

```text
6 outer folds × 3 data fractions × 6 paths × 3 seeds
= 324 retained runs
```

每个 run 固定 1,024 updates，只保留 final EMA，不 early-stop。正式评价必须等 324/324 training freeze 后才允许构造，使用 16 个固定 timestep、4 个共同 noise replicate 和 calendar-month cluster bootstrap。

主判断同时要求：

- PA-RWF 在 50% 数据下实质优于 IID、Fixed、CW 与 Shuffle；
- PA-RWF 对 MuLAN-lite 在 50% 数据下非劣，并在三档 learning-curve AULC 上实质优于它；
- raw reconstruction 和 Gaussian-reference-normalized efficiency 都改善；
- cell、increment 和 joint 三个端点不能靠相互抵消过门；
- 至少 5/6 outer folds、2/3 seeds 支持 PA 相对 IID 的方向。

完整协议见 [`ARCHITECTURE_V1_G0_B_TINY_DENOISER_PROTOCOL.md`](ARCHITECTURE_V1_G0_B_TINY_DENOISER_PROTOCOL.md)，冻结配置 SHA256 为 `58949f9c1729148146d428f3f237d997046e29e4b1e2990adf5574808693e2c9`。

P0 已在 outer fold 0、25% 数据、seed 3 上完成。六条路径各执行 50 次 train-only update，共 300 次；正式状态为 `G0_B_TINY_DENOISER_P0_GO`。主要工程证据为：

- G0-A 六折 residual energy、effective rank、N1 variance 与 nuisance hyperparameters 的最大重放误差均为 0；
- 每条路径的逐日逐 timestep information-budget 误差不超过浮点舍入量，PA-RWF 对 G0-B0 `eta=0.5` 的最大重放误差为 0；
- 51 次 MuLAN-lite schedule 审计（初始化加每次更新后）均通过有限值、正 SNR、端点和单调性检查；
- 六条路径的 base 初始化与 50 组 minibatch/timestep/noise 随机库完全配对；
- update 25 保存的临时 checkpoint 精确复现 update 26 的 loss、model、optimizer 与 EMA，随后已删除；
- 输出目录只保留 JSON 与 SHA256 sidecar，没有 `.pt` 或 `.pth` 权重。

正式 P0 artifact 位于本地 `outputs/architecture_v1_g0_b_tiny_denoiser/G0_B_TINY_P0_RESULT.json`，SHA256 为 `6f614dc6efcacb01d079f8052105722210ff8f48841a1604c9cf50f728a62b80`。首次 P0 因 Codex 受限沙箱没有暴露 GPU 而在 CPU 上执行；随后在同一 WSL 的 RTX 2060 上完成独立 CUDA capacity audit，300/300 updates、有限值和 checkpoint resume 全部通过，峰值分配约 20.7 MiB，结果 SHA256 为 `4c7ab9f539a1d0050f5beef8bd808cec572a08ac6aaf10a4c7804c56b2644052`。

P0 只说明 runner 可以按合同稳定运行。50-step train-subset loss 不是正式比较证据，不能据此排序六条路径。

正式 retained-training runner 为 `repro_scripts/run_architecture_v1_g0_b_tiny_denoiser_formal.py`。它具备：

- 每 128 updates 原子保存 model、optimizer、EMA 与 RNG，可从注册边界恢复；
- 每个 run 完成后只保留 update-1024 final EMA，删除 resume checkpoint；
- MuLAN-lite 在每个 128-update checkpoint 和 final EMA 上执行完整 schedule 安全审计；
- 支持按注册 `run_key` 分批执行，并持续写入只含完成状态的 progress manifest；
- 只有精确 324/324 完成、全部 checkpoint 哈希有效且同 fold/fraction/seed 的六路径初值、subset 与随机库严格配对时，才生成 `training.freeze.json`；
- 训练阶段不实现、不构造也不调用 outer-test reconstruction evaluation。

正式训练已经在 RTX 2060 上完成：

- 324/324 个 run、331,776 次 optimizer update 全部完成；
- 324 个 final EMA checkpoint 与 SHA256 登记一致；
- 0 failure、0 residual resume checkpoint；
- 54 个 `fold × fraction × seed` 配对组的初始化、subset 和共同随机库全部一致；
- 中途一个长进程在第 31 个 run 的 update 256 停滞，终止后从注册 checkpoint 精确恢复；此后按每 12 个 run 重启 CUDA 进程，未改变任何实验身份；
- training freeze SHA256 为 `f93a183ec0906b3553eaaf11f03de005e17dd04eeb35357b9cff9baf2252e385`。

训练冻结后，独立 evaluation runner 使用 267 个 outer-held-out day、16 个固定 timestep、每步 4 个共同 Gaussian noise replicate，以及 Shuffle 的 4 个固定 wrong-day derangement，对全部 324 个 EMA 做了一次共同评估。统计推断以 calendar day 为单位，先平均 noise、timestep 与三个 model seed，再进行 10,000 次 calendar-month cluster bootstrap，并对 13 个注册 contrast 使用 max-T simultaneous 95% bands。

50% 数据主结果如下；正值表示 PA-RWF 更好：

| 对比 | 相对改善 | simultaneous 95% band | 冻结要求 | 结果 |
|---|---:|---:|---:|---|
| PA-RWF vs IID，BRR | +0.640% | [+0.553%, +0.726%] | 点改善 ≥2%，下界 >0 | **失败：稳定但太小** |
| PA-RWF vs Fixed-band，BRR | -0.014% | [-0.026%, -0.002%] | 点改善 ≥1%，下界 >0 | **失败** |
| PA-RWF vs CW-group，BRR | +4.660% | [+4.180%, +5.140%] | 点改善 ≥1%，下界 >0 | 通过 |
| PA-RWF vs Shuffle，BRR | -0.016% | [-0.028%, -0.004%] | 点改善 ≥1%，下界 >0 | **失败** |
| PA-RWF vs MuLAN-lite，BRR | -0.086% | [-0.110%, -0.061%] | 非劣 margin 1% | 非劣通过，但没有优势 |
| PA-RWF vs MuLAN-lite，AULC | -0.087% | [-0.111%, -0.063%] | 点改善 ≥2%，下界 >0 | **失败** |
| PA-RWF vs IID，oracle efficiency | -3.739% | [-3.953%, -3.525%] | 点改善 ≥1%，下界 >0 | **失败** |

PA 相对 IID 的单项结果进一步说明它没有修复核心 transition 缺口：

- cell MSE 改善 1.039%；
- joint-day normalized SSE 改善 1.039%；
- increment MSE **恶化 0.159%**；
- 三个端点中达到 2% 实质改善的数量为 0/3；
- 方向在 6/6 outer folds 和 3/3 model seeds 上均为正，但幅度始终只有约 0.5%–0.8%。

三个控制相对 IID 的 BRR 变化为：Fixed-band `+0.654%`、MuLAN-lite `+0.725%`、CW-group `-4.217%`。Fixed 与 MuLAN 的小幅收益同样没有达到预注册 2% materiality 门。更关键的是，正确 NWP 对齐的 PA-RWF 略差于 wrong-day Shuffle，因此没有证据表明逐日 NWP-to-schedule 对齐贡献了收益。

正式状态按冻结决策顺序为：

```text
G0_B_STRUCTURED_SCHEDULE_NO_GO
```

这不是训练失败或统计功效不足：工程门全部通过，PA vs IID 的微小正效应甚至具有窄置信带；被否定的是“效应足够大、来自正确 NWP 对齐、并降低有限模型相对 Gaussian oracle 的学习难度”这一方法假说。因此不授权实现完整 predictability-aligned diffusion。

正式结果位于 [`G0_B_TINY_DENOISER_RESULT.json`](../outputs/architecture_v1_g0_b_tiny_denoiser/formal_evaluation/G0_B_TINY_DENOISER_RESULT.json)，SHA256 为 `2cff3adae29d572a50a2480453a834d873ac3286516eafa77db2ee8e9136a217`；267 日逐日指标 artifact SHA256 为 `af741aa9de3041029d708e4efeff744b72d4ee58560a9b410cb3b8f11dd0b2b9`。结果哈希、每日数组、13 个 contrast 和 10,000 次 bootstrap 已独立精确重放。

协议中的可选“PA-50% vs IID-100% data equivalence”不作为结论解释：BRR 的三个分母按各自数据 fraction 的 IID 均值分别归一化，因此跨 fraction 直接比较 BRR 会机械地令每档 IID 均值等于 1，不能单独证明 50% 数据等价于 100% 数据。该缺陷不影响任何必需 Go/No-Go 门。

## 9. 下一阶段建议：改变生成对象，而不是继续调 schedule

G0-B 给出了很明确的机制线索。PA-RWF 相对 IID 的六组 reconstruction MSE 改善主要集中在 `low_common`（约 5.06%）和 `low_local`（约 1.26%）；mid 两组略有恶化，high 两组几乎不变。它能够略微改善平滑的 level/joint 指标，却没有改善相邻小时 increment。这与项目最早发现的 ramp/lagged 缺口一致：问题更可能在“模型把什么对象当作基本随机变量”，而不是各频带在 forward corruption 中分到多少 SNR。

因此下一条建议主线不是继续调 `eta`、增加 schedule head 或扩大 tiny denoiser，而是先建立一个独立的 **transition-generative-object Probe**：

```text
标准对象：24 小时 level trajectory
候选对象：可逆的 anchor + transition/innovation trajectory
硬约束：通过累积算子精确还原 level，level 与 increment 不允许互相矛盾
```

候选 Probe 必须把“生成对象改变”与普通坐标缩放分开：至少包含原 level-space、train-only 标准化的可逆 transition-space、同预算正交变换控制，以及破坏真实相邻关系的负对照；共享网络容量、IID diffusion、训练预算和随机库。第一阶段仍只回答有限容量下的 held-out level/increment/joint reconstruction，不直接训练完整场景模型。

只有 transition-space 同时满足以下条件，才进入完整 mixed-measure generative protocol：

- increment/ramp 指标达到预注册实质改善；
- level 与 joint 指标非劣；
- 优势超过正交变换和时间破坏对照，而不是来自尺度重标定；
- 三种子和 outer folds 稳定；
- 新颖性复审确认贡献不是已有 differencing、wavelet diffusion 或普通状态空间重参数化的直接重复。

这是下一阶段的建议，尚未冻结协议、没有训练新模型。若该表示 Probe 也失败，再转向真正改变概率分解的 stochastic state-space/innovation model，而不是回到 adapter、额外 loss 权重或 schedule 雕花。

## 10. 当前不会做什么

- 不继续 family-v1.3 SNR/NWP recurrence gate；
- 不把 family-v1.2 的统计显著小效应包装成模型创新；
- 不因为 D0-v 被用作 Probe 平台就宣布 Diffusion 已经战胜 Flow；
- 不把固定 `Σ_NWP`、条件白化或通用条件多维 schedule 包装成方法首创；
- 不把 G0-A Go 解释成 G0-B 或完整 diffusion 已经有效；
- 不把 G0-B0 的解析 Gaussian proxy risk 下降解释成真实模型收益；
- 不把 tiny-denoiser reconstruction 结果直接解释成完整场景生成收益；
- 不因 PA 相对 IID 有统计显著的 0.64% 小改善而越过冻结的 2% 实质门；
- 不实现完整 predictability-aligned diffusion，不继续调 `eta` 或扩大 learned schedule；
- 不同时恢复 transition-aware loss、Flow proper-score adapter 等备用工程线；
- 不访问仍封存的 selection、calibration 或最终外部测试；
- 不同时启动两个重型方向。

## 11. 数据与证据边界

- 当前架构结论仍属于 validation-stage mechanism discovery；
- 历史诊断使用过的日期保持 `R-SEEN`；
- selection、calibration 和最终外部测试仍未用于模型选择；
- 数据、论文 PDF、checkpoint 和大型场景归档保留在本地，不进入公开 Git 仓库；
- 公开仓库保存代码、冻结配置、测试和可阅读的正式结论。

## 12. 文档维护规则

从现在开始：

1. 每完成一个正式阶段，只更新本文档中的状态表、最新结果和下一决策；
2. 新实验仍各自保留一份 frozen protocol 和一份 formal result，以保证预注册与结果不可被滚动叙事覆盖；
3. 不再为每次路线讨论新增新的 roadmap、HTML 或 DOCX；
4. 需要组会材料时，从本文档生成一次性演示版本，并标明生成日期；
5. README 只提供项目简介、当前一句话状态和本文档入口。

## 13. 关键证据入口

- [`ARCHITECTURE_V1_G0_B_TINY_DENOISER_PROTOCOL.md`](ARCHITECTURE_V1_G0_B_TINY_DENOISER_PROTOCOL.md)
- [`G0_B_TINY_DENOISER_RESULT.json`](../outputs/architecture_v1_g0_b_tiny_denoiser/formal_evaluation/G0_B_TINY_DENOISER_RESULT.json)
- [`ARCHITECTURE_V1_G0_B0_SCHEDULE_RESULT.md`](ARCHITECTURE_V1_G0_B0_SCHEDULE_RESULT.md)
- [`ARCHITECTURE_V1_G0_B0_SCHEDULE_PROTOCOL.md`](ARCHITECTURE_V1_G0_B0_SCHEDULE_PROTOCOL.md)
- [`ARCHITECTURE_V1_G0_A_PREDICTABILITY_RESULT.md`](ARCHITECTURE_V1_G0_A_PREDICTABILITY_RESULT.md)
- [`ARCHITECTURE_V1_G0_A_PREDICTABILITY_PROTOCOL.md`](ARCHITECTURE_V1_G0_A_PREDICTABILITY_PROTOCOL.md)
- [`ARCHITECTURE_V1_FAMILY_V1_2_FORMAL_EVALUATION_RESULT.md`](ARCHITECTURE_V1_FAMILY_V1_2_FORMAL_EVALUATION_RESULT.md)
- [`ARCHITECTURE_V1_FAMILY_V1_2_PROBE_PROTOCOL.md`](ARCHITECTURE_V1_FAMILY_V1_2_PROBE_PROTOCOL.md)
- [`ARCHITECTURE_V1_FAMILY_V1_1_FORMAL_COMPARISON_RESULT.md`](ARCHITECTURE_V1_FAMILY_V1_1_FORMAL_COMPARISON_RESULT.md)
- [`ARCHITECTURE_V1_TEMPORAL_MECHANISM_V3_2_RESULT.md`](ARCHITECTURE_V1_TEMPORAL_MECHANISM_V3_2_RESULT.md)
- [`CROSS_MODEL_ARCHITECTURE_DIAGNOSTIC.md`](CROSS_MODEL_ARCHITECTURE_DIAGNOSTIC.md)
- [`MSCADM_PAPER_REPRODUCTION_REPORT.md`](MSCADM_PAPER_REPRODUCTION_REPORT.md)
