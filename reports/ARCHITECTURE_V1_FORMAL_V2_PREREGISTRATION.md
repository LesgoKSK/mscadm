# Architecture-v1 formal-v2 正式预注册说明

> 状态：R0 尚未开始正式训练；calibration 与 selection 仍处于 sealed 状态。  
> 本文是配置的中文解释，不替代机器可读协议。若文字与 JSON 冲突，以 JSON 为准。

## 1. 这次到底要回答什么

formal-v2 的第一步不是直接宣布 flow、diffusion 或状态空间模型谁最好，而是先建立一个可信的新基线 **R0**，确认下面三件事：

1. 新的数据协议没有把已经看过的日期重新当作新证据；
2. 共同的 NWP 编码器和精确边界状态模块确实能够学到东西；
3. R0 在三个独立训练 seed 下能够稳定、完整地训练，而不是只靠某个幸运 seed 得到一个好数字。

只有 R0 通过 G0/G1，后续才实现参数匹配的 T0，以及 T1-feature、T1-source、T1-shuffle。即使进入后续阶段，仍然不能提前查看 selection；所有候选、指标和判定规则必须先冻结，selection 最后只打开一次。

机器可读的两份主文件是：

- [formal-v2 实验配置](../repro_configs/architecture_v1_formal_v2.json)，当前 SHA256 为 `b92f8647d7e1874c81a5927ff552e9ce03c9906b147ec389c71d55d06b317c9c`；
- [Go/No-Go 数值门](../repro_configs/architecture_v1_go_no_go_v1.json)，冻结 SHA256 为 `0c1cb8b87fd6d673dbb1a0fa352a3aa7de47c351cfc9d9ca9f436b220629fc4a`。

formal-v2 配置内也登记了 Go/No-Go 文件的路径、schema 和 SHA256。如果文件被无意修改，身份校验必须失败，不能悄悄沿用旧结论。

## 2. 数据证据如何重新隔离

本地 GEFCom 数据共有 731 个可用日。过去的两轮跨模型诊断已经看过 300 日；architecture-v1 smoke 又在工程验收过程中接触过 14 个日期。无论这些日期是否真正参与过梯度更新，只要其结果曾影响调试或路线判断，就不能再充当盲测证据。

因此 formal-v2 把这 314 日统一登记为 **R-SEEN**，只允许用于历史诊断，禁止拟合、校准、模型选择和最终检验。对应的 smoke 隔离清单在 [architecture_v1_smoke_quarantine.json](../repro_configs/architecture_v1_smoke_quarantine.json)。

剩余 417 日使用固定的 `PCG64`、固定 seed `20260816` 和固定分配顺序重新划分：

| 角色 | 日数 | formal-v2 中的用途 |
|---|---:|---|
| train | 267 | 拟合参数、拟合标准化器及 train-only 阈值 |
| validation | 50 | early stopping、R0 的 G0/G1、T0/T1 的开发期判断 |
| calibration | 50 | 将来对已经冻结的候选做后处理；当前保持密封 |
| selection | 50 | 所有候选冻结后执行一次正式架构门；当前保持密封 |
| R-SEEN | 314 | 只作历史诊断，不产生新证据 |
| local final | 0 | 本地数据不再伪装成最终测试集 |

每个角色的完整日期列表没有靠文件名或人为记忆维持，而是由日期集合 SHA256 锁定。formal R0 的数据入口必须使用 `architecture_v1.data.build_architecture_v1_fit_data`，它只允许物化 train 和 validation 的目标；calibration、selection、R-SEEN 与 final 都是 hard-forbidden。

最终论文级确认需要另行登记、冻结且此前未接触过的外部风电数据。任何本地 GEFCom 日期都不能在事后改名为 final。

## 3. 缺失值为什么不会再次造成泄漏

formal-v2 先确定日期角色，之后才处理目标缺失。填充值只能在同一个 `role × zone × calendar_day` 内先向前、再向后传播，禁止跨日期或跨角色传播。

填值只用于保持张量形状完整，并不把缺失值变成真值：

- level、atom、coverage 等逐格指标只使用 `raw_observed_cells`；
- ramp 只有相邻两个小时都原始可观测时才计分；
- lagged variogram 只有相应 pair 所需的全部端点都可观测时才计分；
- Joint ES 使用观测 mask，并用有效维数平方根归一化；
- H/K/L 和 any-atom event 只在 240 个格点完整观测的日期上计算；
- entry/exit Brier 只使用两端均观测的相邻小时；
- 如果某天对某项指标没有任何有效单元，程序必须报错，不能静默丢弃。

这意味着“为网络准备的有限数值”和“可以监督或计分的真实观测”是两套不同语义。

## 4. R0 是什么，又不是什么

R0 是一个新的 **common-shell joint time-domain Rectified Flow** 基线。它一次生成完整的 `10 zones × 24 hours` 场景，并采用与后续候选共享的条件编码和边界状态语义。

R0 当前会做：

1. 在 train 上训练共同的 NWP 编码器 `E` 和精确 atom 模块 `A`；
2. 用 validation 选择 E/A 的最佳 checkpoint；
3. 重新加载该 checkpoint，丢弃只为 E/A 训练服务的辅助头；
4. 对 E/A 做稳定哈希并冻结；
5. 只在 sampled interior active coordinates 上训练连续 flow；
6. 用 `[0,1,2]` 三个模型 seed 独立训练并完整报告；
7. 在 validation 上用 100 members、3 个预注册 sampling seeds、Heun 16 steps，即每条路径 31 次 flow NFE 做稳定性评价。

R0 当前不会做：

- 不复刻旧 STGF checkpoint，也不声称数值重现旧 Time-domain 结果；
- 不在 R0 flow 训练中继续改 E/A；
- 不读取 calibration 或 selection 目标；
- 不根据 selection 结果调 learning rate、epoch、loss 权重或阈值；
- 不证明 R0 是最终最佳模型；
- 不决定最终应选择 flow、diffusion 还是纯概率 SSM；
- 不产生本地 final-test 结论。

R0 的任务很朴素：提供一个数据干净、输出语义明确、能够掩码、能跨 seed 复核的共同参照点。

## 5. E/A 是什么，为什么先训练再冻结

### 5.1 E：共同的条件编码器

`E` 把 20 维条件输入编码为后续所有候选都能使用的共同表示。输入包含 predictor-only NWP 及协议定义的条件特征，不能偷看目标列。E 本身能够编码跨小时和跨场站信息，因此以后所谓 T0 “memoryless” 只表示新增 cell 不读 `h[t-1]`，并不表示整个网络看不到 24 小时结构。

### 5.2 A：精确的边界状态模块

`A` 为每个 zone-hour 预测三个互斥状态：

- `0`：功率精确为零；
- `1`：连续内部值，严格处于 `(0,1)`；
- `2`：功率精确为一。

连续 flow 只作用于状态 1。状态 0/2 的连续 latent 和 velocity 必须严格为零，最后直接重建为精确的 0/1，而不是用接近边界的小数冒充 atom。

### 5.3 为什么 E/A 不能只靠 atom 分类训练

如果 E 只接受 atom NLL 的梯度，它可能很擅长区分“零/内部/一”，却没有动力保存内部功率大小的信息。这会让后面的 R0 flow 和 T1 都拿到一个对连续轨迹不够有用的编码器。

formal-v2 因此给 E/A 增加一个架构中性的 interior-location 辅助任务：

- 仅在 `observed AND state==interior` 的格点上使用；
- 目标是 train-only 标准化后的 interior logit power；
- 使用 SmoothL1，权重固定为 `0.25`；
- 辅助头为 `LayerNorm + Linear`；
- 最佳 E/A checkpoint 选定后立即丢弃辅助头，后续候选不能使用它。

E/A 不与 R0 flow 联合预训练。原因是：如果先让 E 为 R0 的 velocity objective 专门优化，再把它称为所有架构的“共同基础”，会天然偏向 R0。现在的 atom 分类加 interior-location 任务更接近架构中性的条件表征。

### 5.4 上边界样本太少怎么办

精确为一的事件可能非常稀少。配置用 Jeffreys 平滑得到 train prior，并预先规定：train 中至少 20 个、validation 中至少 5 个 exact-one，才允许学习受限的 contextual deviation；其 logit 绝对偏移不超过 `2.0`。

只要任一支持量不足，`q1` 就固定为 train-only Jeffreys prior，同时禁止把 upper-atom calibration 或 duration 当作成功端点。不能因为样本少而删除这个分支，也不能在看过 selection 后改变最低样本数。

## 6. G0：先证明共同 E/A 真的学到了东西

G0 完全在 validation 上执行，而且三个训练 seed 必须逐个满足全部条件：

| 检查 | 数值门 | 直白解释 |
|---|---:|---|
| atom NLL 对 E 的梯度范数 | `> 1e-12` | atom 任务确实能更新编码器，不是断图 |
| location auxiliary 对 E 的梯度范数 | `> 1e-12` | 连续位置任务也确实能更新编码器 |
| validation atom NLL 相对 train-prior | 至少改善 `1%` | A 不能只复述先验频率 |
| validation location loss 相对 zero predictor | 至少改善 `5%` | E 确实保留了内部功率位置 |

G0 不是论文性能结论。它是一道“这个共同基础是否可学”的工程—科学接口门。只要一个 seed 失败，就不能继续把其 E/A 复制给后续模型。

## 7. G1：R0 的三 seed 稳定门

G1 同样只看 validation，不打开 calibration 或 selection。它分成四层。

### 7.1 数值完整性

三个 seed 必须全部完成；预期更新完成率必须为 `1.0`。允许的 nonfinite batch、跳步、静默 rollback、failure bundle、非法 archive cell 和 atom 边界违规数全部为 `0`。

warm-up 后被 gradient clip 的更新比例最多 `25%`。最佳 validation velocity MSE 必须比 zero-velocity 参照至少改善 `5%`，即相对比值不超过 `0.95`。E/A 的 SHA256 在整个 flow stage 中必须保持不变；同 seed validation replay 必须 bitwise 相同；member chunk 改变不能改变生成结果。

### 7.2 每个 seed 的灾难性质量上界

这些不是“达到论文 SOTA”的门，只用于阻止一个数值有限但明显失效的基线继续向下游传播：

| 指标 | 每个 seed 的要求 |
|---|---:|
| level-CRPS | `≤ 0.100` |
| ramp-CRPS | `≤ 0.065` |
| normalized Joint ES | `≤ 2.0` |
| coverage90 | `[0.70, 0.98]` |
| width90 | `[0.10, 0.70]` |

### 7.3 三 seed 的离散程度

| 指标 | 三 seed 的最大值减最小值 |
|---|---:|
| level-CRPS | `≤ 0.0025` |
| ramp-CRPS | `≤ 0.0012` |
| normalized Joint ES | 相对跨度 `≤ 3%` |
| coverage90 | `≤ 0.035` |
| zero Brier | `≤ 0.006` |
| three-state atom Brier | `≤ 0.012` |

这些门防止只汇报最好的一个 seed。即便三 seed 均值好看，只要其中一个 seed 明显漂移，R0 也不通过。

### 7.4 G1 失败后允许做什么

可以检查并修复数据、数值实现、断图、resume 或 checkpoint 问题，但不能为了“找一个能过门的结果”替换 seed，也不能查看 calibration/selection。任何改变实验语义或超参数的修订都应形成新版本、留下审计记录，再从未接触 selection 的状态重新运行。

## 8. 正式训练预算如何固定

GPU preflight 只运行 50 updates，所有权重必须丢弃。它唯一允许决定的是 batch size：默认 16；只有同步 CUDA 测得 50-update peak allocated memory 超过 `4.5 GiB` 或发生 OOM 时，才使用 8。这个决定必须在任何 retained-weight training 前冻结。

正式训练统一使用 FP32，关闭 AMP、TF32 和 cuDNN benchmark，启用 deterministic algorithms，worker 数为 0。三个 model seed 固定为 `[0,1,2]`，不同阶段的初始化、shuffle 和训练路径使用配置中登记的 seed offsets。

E/A stage 最多 4080 updates，至少运行 1360 updates；flow stage 最多 12240 updates，至少运行 2040 updates。两者每 170 updates validation 一次，patience 都是 12 次 validation。early-stop 的最小改善统一为：

```text
new < best - max(1e-5, 1e-3 * best)
```

validation flow 使用预先生成并持久化的 4 份固定 noise/time bank，所有 checkpoint、seed 和未来候选复用相同张量，避免每次 validation 因随机噪声不同而改变 early-stop 判断。

## 9. R0 通过后，T0/T1 怎么走

### 9.1 T0：参数匹配的无新增递归状态对照

T0 在共同 shell 上加入与未来 T1 参数形状和接口匹配的 temporal cell，但断开 `h[t-1]` 的递归通道。它回答的是：“T1 的改善是否只是多了一组参数或多了一层变换？”

T0 首先在 validation 上过 G2：相对 R0，level-CRPS 和 ramp-CRPS 的绝对非劣界均为 `0.0010`，normalized Joint ES 的相对非劣界为 `1.5%`；单侧 cluster CI 上界必须低于对应 margin。T0 还必须复用完全相同的 E/A SHA256，并满足 R0 的稳定性条件。

如果 T0 本身明显破坏 R0，那么以后看到“T1 优于 T0”可能只是 T1 修复了一个人为损坏的对照，不能解释成状态记忆有效。

### 9.2 T1-feature、T1-source 与 T1-shuffle

- `T1-feature`：显式状态模块只作为 velocity conditioner 的特征；随机 source 仍保持原方案；
- `T1-source`：有记忆的状态空间过程成为随机动态 source；
- `T1-shuffle`：打乱时间或 NWP—轨迹对齐，作为负对照。

首轮机制比较必须共享冻结的 E/A、训练预算、成员数、采样步数，并在可行处共享 state allocation 与初始 noise。否则 T1 与 T0 的差异可能来自 atom、调参预算或 Monte Carlo 噪声，而不是递归状态。

T1 的开发期调试和是否值得进入候选 roster 可以使用 validation。validation 可以反复承担开发角色，因此由它产生的 CI 不能包装成最终确认性结论。

## 10. selection 为什么只能打开一次

selection 的职责是对已经完成的候选做一次预注册架构判定，而不是边看结果边发明下一个模型。正式状态机为：

```text
sealed → authorized → in_progress → consumed
```

只有当候选代码、配置、checkpoint 身份、评价指标、calibration 规则和 Go/No-Go SHA256 全部冻结后，才能从 sealed 变为 authorized。

一个常见但无效的做法是：

```text
先看 selection 上 T1/T0
→ 根据结果决定如何实现 T2
→ 再用同一 selection 宣称 T2 更好
```

第二次比较已经受到第一次结果的指导，因此同一批 50 日不再是盲证据。formal-v2 明确禁止 selection 被后续 T2/T3 自适应复用。正确做法只有两种：

1. 在 selection 仍 sealed 时，利用 validation 完成开发并一次性冻结要比较的候选，然后统一打开 selection；
2. 如果 selection 已经 consumed，后来新增的 T2/T3 只能等待新的外部锁定数据，不能继续给旧 selection 赋予确认性含义。

calibration 也不能用来移动 selection margin。它只负责对冻结候选执行预先规定的后处理。

## 11. T1 对 T0 的正式数值门

所有 lower-is-better 指标统一定义：

```text
difference = candidate - comparator
```

因此负数表示候选更好。

### 11.1 主优效门 G3

主端点是 observed-only overall local ramp-CRPS。每个 T1 候选必须同时满足：

1. 点差 `≤ -0.0010`；
2. Holm 校正后的单侧 CI 上界 `< 0`；
3. 三个训练 seed 的逐 seed 平均差全部为负。

支持性动态端点固定为 p=1 的 increment-path lagged variogram composite：cross-zone 与 same-zone temporal 两类分别取 lag 1–6，再等权平均。它必须满足点差 `≤ -0.0002`、单侧 CI 上界 `<0`，且三 seed 方向全部为负。

Pearson cross-lag 仍然报告，但只作易解释的结构诊断，不作为唯一的选择端点，因为 correlation error 不是 proper score，并且容易混入边际和共同 NWP 影响。

### 11.2 保真非劣门 G4

动态变好不能以基础质量明显变差为代价。G4 是 intersection-union gate：下面所有端点都通过，整体非劣才通过。

| 端点 | 允许的最大恶化 |
|---|---:|
| level-CRPS | 绝对 `0.0010` |
| normalized Joint ES | 相对 `1.5%` |
| `|coverage90-0.90|` | 绝对 `0.01` |
| Winkler90 | 相对 `2%` |
| max-up / max-down ramp CRPS | 各相对 `2%` |
| three-state atom Brier | 绝对 `0.002` |
| atom H/K/L CRPS | 各相对 `2%` |
| atom entry/exit Brier | 各绝对 `0.001` |

每项单侧 CI 上界都必须小于 margin。为了防止模型仅靠无差别增宽区间获得表面改善，width90 点估计最多增加 `5%`，其单侧 CI 所允许的增加最多 `10%`。

首轮比较中，如果 T0/T1 的 E/A hash、sampled state array 和 analytic atom probabilities 在 common random numbers 下完全一致，atom 部分是结构性相等，而不是靠有限样本“没检出差异”。analytic atom probability 的最大允许数值差为 `1e-12`。

### 11.3 负对照 G5

真实 T1 相对 T1-shuffle 的 ramp-CRPS 点差必须 `≤ -0.0005`，单侧 CI 上界 `<0`，三 seed 全部同方向；shuffle 至少应消除真实 T1 收益的 `50%`。否则“模型使用了正确时间/NWP 对齐”的机制解释不成立。

对生成后的 joint members 再做 trajectory shuffle 时，cross-zone ramp variogram lag-0 至少应相对恶化 `2%`，且单侧 CI 下界大于 0。这个后处理负对照回答的是成员对齐是否真的携带跨场站信息，与训练期的 T1-shuffle 不是同一个操作。

## 12. 为什么采用这些数值

这些数值不是看过 formal-v2 selection 后倒推出来的，而是依据已经永久划入 R-SEEN 的历史 confirmation/diagnostic 尺度和预先定义的工程容忍度制定。历史依据见 [跨模型架构诊断](CROSS_MODEL_ARCHITECTURE_DIAGNOSTIC.md) 及其 [paired bootstrap 表](../outputs/cross_model_diagnostics/primary_paired_bootstrap.csv)。

### 12.1 `0.0010` 的 ramp 实际重要差异

历史模型的 ramp-CRPS 大致位于 `0.049–0.053`。因此 `0.0010` 约等于 2% 的相对改善。历史 MM–DDPM ramp 差为 `+0.002527`，所以改善 `0.0010` 相当于修复约 40% 的已观察缺口。

历史同一 outer 内三 seed 的最大 ramp 跨度约为 `0.00096`。把实际重要差异设为 `0.0010`，还可以避免把常见训练 seed 波动误写成机制突破。

### 12.2 为什么不能把主门降为 `0.0005`

历史 150 日 paired ramp CI 的半宽约为 `0.00042`。只按独立样本的平方根规律粗略缩放到 formal-v2 的 50 个 selection 日，半宽已约为：

```text
0.00042 × sqrt(150 / 50) ≈ 0.00072
```

真实的 month-cluster 推断通常只会更宽。因此 `0.0005` 虽然数值上看似改善，却很可能在 50 日设计中没有足够功效。formal-v2 把 `0.0010` 作为主端点的 minimum practically important improvement，并要求在 train-only synthetic perturbation power audit 中，对 `-0.0010` 达到至少 80% power。

如果 power 不足，处理方式是增加新的外部日期或把结果标记为 underpowered；不能降低 margin、放宽 alpha 或改用普通 iid member bootstrap。

### 12.3 R0 三 seed 稳定界的来源

历史 confirmation 中：

- 同一 outer 的 level-CRPS 最大三 seed 跨度约 `0.00198`，因此 R0 上界取更宽但仍有限的 `0.0025`；
- ramp-CRPS 最大跨度约 `0.00096`，对应取 `0.0012`；
- Joint ES 的最大相对跨度约 `2.15%`，对应取 `3%`；
- coverage90 最大跨度约 `0.0287`，对应取 `0.035`；
- zero Brier 最大跨度约 `0.00446`，对应取 `0.006`。

这些是稳定性容忍界，不是从旧模型继承的性能目标。

### 12.4 其他非劣界

level `0.0010`、Joint ES `1.5%`、coverage error `0.01`、Winkler `2%`、H/K/L `2%` 及 entry/exit Brier `0.001`，延续了复审阶段已经提出的第一版工程门。它们在 formal-v2 selection 打开前已经写入带 hash 的 JSON，因此之后只能执行，不能移动。

## 13. bootstrap、seed 和有限场景集怎么解释

每个训练 seed 使用三个固定 sampling seeds `[21000,21001,21002]`。每份场景集有 100 个成员，使用 Heun 16 steps / 31 flow NFE。三个 sampling replicate 先在同一个 `training-seed × day` 内平均；不能把 300 个成员拼成一个假想 IID ensemble。

主点估计先对 sampling replicate 平均，再对三个训练 seed 和 calendar day 等权平均。主 bootstrap：

- 单位是 `calendar month` cluster；
- 按气象季节分层；
- 运行 20,000 次；
- 固定 bootstrap seed `2026081501`；
- 至少需要 12 个独立 calendar-month clusters；
- ensemble members 绝不作为重采样单位。

主 CI 只量化“在这三个已经训练出的模型条件下”的日历不确定性。因为训练 seed 只有三个，不能声称一次日期 bootstrap 已经完整覆盖训练随机性。因此正式报告必须同时给出：

- 每个 seed 的点估计和 month-cluster CI；
- 跨 seed mean、SD、min、max；
- 三 seed 的方向；
- paired-day、leave-one-month-out、leave-one-seed-out 敏感性。

T1-feature 与 T1-source 共用同一 T0，会形成两个主假设，因此使用单侧 Holm FWER `0.05`。支持性 lagged-VS 只在该候选自己的主 ramp 门通过后按固定顺序检验。非劣门要求全部端点通过，采用 intersection-union 逻辑，不通过“挑一个最好看的 safety 指标”作结论。

## 14. 效率门何时算通过

效率测试只需要 validation conditions，不需要读取 selection target。比较必须在相同硬件、精度、compile 状态、成员数和 member chunk 下进行；先 warm-up 5 次，再做 30 次配对测量，随机化候选顺序，并在计时前后执行 CUDA synchronize。端到端时间包含 atom allocation 和 transport，但不含磁盘 I/O。

T1 相对 T0 必须满足：

- nominal parameter manifest 相同；
- flow NFE 相同；
- effective parameter ratio `≤1.15`；
- median sampling latency ratio `≤1.20`；
- latency ratio 的 paired-bootstrap 95% 上界 `≤1.25`；
- peak VRAM ratio `≤1.25`。

以后 flow 与 joint DDPM 做家族选择时，只有质量非劣且 flow/DDPM latency ratio 的 95% 上界 `≤0.80`，才能声称 flow 至少快 20%。不能把“16 integration steps”误写成“16 NFE”；Heun-16 明确是 31 次 flow velocity evaluation。

若以后 T3 概率 SSM 相对 T1 的 ramp-CRPS 落入 `±0.0005` TOST 等效界、通过其他非劣门，并且参数或 latency 至少降低 25%，才以简洁性选择 T3。

## 15. 执行顺序与停止规则

formal-v2 当前应按以下顺序执行：

```text
冻结配置、代码、环境和数据清单
→ 50-update GPU preflight，丢弃全部权重
→ train-only E/A，validation 选 checkpoint
→ G0：E/A 可学习性
→ hash 并冻结 E/A
→ 三 seed R0 flow
→ G1：数值、可学习性、质量 sanity 与 seed 稳定
→ 若通过，在 selection 仍 sealed 时实现 T0
→ validation 上执行 G2
→ 实现并用 validation 开发 T1-feature/source/shuffle
→ 冻结完整候选 roster、代码、指标、calibration 规则和 hashes
→ 只打开一次 selection，执行 G3–G5
→ calibration 仅按冻结规则后处理
→ 最终论文结论等待外部 locked final
```

如果 R0 没有通过 G0/G1，当前结论只能是“共同基础或训练协议尚不可靠”。此时应停下来修地基，不能通过添加 SSM、diffusion 或更多 loss 来掩盖问题。

如果 T1 在 validation 上没有稳定价值，可以停止实现更复杂的状态空间路线；这属于开发期 No-Go，不是外部最终否定。如果 T1 通过开发门，也不能立即宣布成功，仍需等一次性 selection 以及未来外部 final。

## 16. 一句话总结

formal-v2 的核心不是“先把 R0 跑出一个好分数”，而是：**用未被前期诊断污染的 train/validation，证明共同 E/A 可学、R0 三 seed 稳定；随后在完全不触碰 selection 的前提下准备公平的 T0/T1 候选，最后让一次性、预注册、可审计的 selection 决定动态机制是否真的值得继续。**
