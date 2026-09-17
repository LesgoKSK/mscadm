# Architecture-v1 G0-B：Tiny-denoiser Utility Probe 协议

冻结日期：2026-09-15

协议状态：在任何 G0-B target-aware P0、tiny-denoiser 训练或 held-out reconstruction 结果产生前冻结。

证据范围：267 个既有 train calendar days 内的嵌套外层交叉拟合机制实验；不是 validation、selection、calibration、外部确认或完整生成模型比较。

机器配置：[`architecture_v1_g0_b_tiny_denoiser.json`](../repro_configs/architecture_v1_g0_b_tiny_denoiser.json)

配置 SHA256：`58949f9c1729148146d428f3f237d997046e29e4b1e2990adf5574808693e2c9`

预执行说明：冻结后的只读文字审阅、任何 target access 或 P0 之前，将 P0 hard check 中含糊的 “one evaluation batch” 澄清为 “one train-subset diagnostic batch”。这只消除了 outer-test 提前评估的误读；路径、数据、模型、随机数、训练预算、指标和全部阈值均未改变。

## 1. 本轮只回答一个问题

G0-A 已经回答：

```text
NWP 能预测六类风电 residual 的条件不确定性。
```

G0-B0 已经回答：

```text
这些条件方差可以变成预算匹配、无硬 cutoff、数值稳定的 schedule。
```

G0-B 现在要回答仍然未知的一步：

> 在相同总 Gaussian information budget、相同真实条件信息、相同训练数据、相同网络容量和相同更新次数下，正确的 NWP predictability allocation 是否真的让一个有限容量 denoiser 更容易重建未参与训练的真实风电 residual？

这里测量的是 **denoising utility**，不是场景生成质量。只有本 Probe 通过，才有理由另行设计完整 predictability-aligned diffusion。

## 2. 为什么不能直接训练完整 D0-v

如果现在直接训练完整模型，结果会同时混入：

- 新 schedule；
- 大型 denoiser 的表达能力；
- atom 分支；
- checkpoint selection；
- 31-step sampler；
- 场景指标和训练稳定性。

这样即使分数变化，也很难知道原因。G0-B 因此使用一个固定的 56,058 参数 joint MLP，只研究：

```text
同一条真实 residual
        +
不同但预算匹配的 corruption path
        ↓
固定容量 denoiser 的 held-out reconstruction 难度
```

本轮不会生成 100 条场景，不训练 atom，不比较 Flow 与 Diffusion。

## 3. 前置信息与预注册属性

冻结前已经知道：

- G0-A 的 NWP variance 结果；
- G0-B0 的 `eta=0.5` 数值范围；
- reserve-then-water-fill 在 Gaussian proxy 下会解析地降低 posterior risk。

冻结前没有产生任何 G0-B tiny-denoiser loss、held-out reconstruction 或六路径比较结果。因此本协议对 denoiser utility 是前瞻冻结，但不能把已经看过的 G0-B0 解析结果包装成独立预注册证据。

## 4. 数据边界与外层交叉拟合

只允许加载原数据协议中的 267 个 `train` days。六个 G0-A chronological outer folds 原样复用：每次用五折建立 nuisance estimators 和训练 tiny denoiser，用剩余一折评估。

```text
267 个 train days
        │
        ├── outer-train：222 或 223 天
        │      ├── 只在这里拟合 conditional mean / variance
        │      └── 只在这里训练 tiny denoiser
        │
        └── outer-test：45 或 44 天
               └── 只做一次 out-of-fold reconstruction evaluation
```

validation、calibration、selection、R-SEEN 和 final targets 始终禁止访问。

### 4.1 必须精确重放 G0-A nuisance pipeline

每个 outer fold 必须先重放 G0-A：

1. 对 outer-train 内部做四折 conditional-mean cross-fitting；
2. 用 nested out-of-fold residual energy 选择并拟合 N1 variance model；
3. conditional mean 与 N1 都只能使用 outer-train target；
4. 在 outer-test 上生成真正 out-of-sample 的 residual 和六组 variance；
5. outer-test 的 nuisance hyperparameters、residual energy、effective rank 与 `N1_variance` 必须在 `1e-10` 内复现正式 G0-A artifact。

任何一项不匹配，P0 在 optimizer update 前停止。

### 4.2 为什么 nuisance model 不随 25%/50%/100% 重拟合

本轮要隔离的是 **denoiser 的样本效率**。因此，每个 outer fold 的 mean/variance nuisance estimators 都用完整 outer-train 拟合一次，然后在三档 denoiser 数据量之间冻结。

这意味着结论只能写成：

> 在同一套已建立的条件均值和不确定性信息下，某条 corruption path 是否更省 denoiser 数据。

不能写成端到端系统只需要更少的总标注数据。

## 5. Continuous residual 与六组正交结构

只处理 observed interior 功率：

\[
x=\operatorname{logit}(\operatorname{clip}(y,10^{-4},1-10^{-4})),
\qquad
r=M\bigl(x-\hat\mu_{\mathrm{NWP}}\bigr).
\]

精确 0、精确 1 和缺失位置在 residual target 中为 0，且从 loss 与所有正式指标排除。

六组仍为：

```text
low_common,  low_local,
mid_common,  mid_local,
high_common, high_local
```

对第 `k` 组使用冻结的空间与时间投影：

\[
Q_k(X)=P_{s,k}XP_{t,k}.
\]

六个 `Q_k` 两两正交并且和为恒等映射，所以：

\[
r=\sum_k Q_k(r).
\]

每日预算权重是 G0-A 的六组 effective rank 归一化值，而不是六组等权 macro score。

## 6. 所有路径共享的真正公平预算

对同一天、同一 timestep、同一真实条件方差 `v_k`：

\[
I^{\mathrm{IID}}_k
=\frac12\log(1+v_k\rho_0),
\qquad
B=\sum_kw_kI^{\mathrm{IID}}_k.
\]

六条路径都必须逐日逐步满足：

\[
\boxed{\sum_kw_kI^{(m)}_k=B}.
\]

因此任何路径都不能通过“总体留下更多原始信号”获得优势。配置冻结的容差为 `1e-12`。

对于 Fixed、MuLAN-lite、Predictability-aligned 和 Shuffle，统一保留：

\[
I_{k,\mathrm{res}}=0.5I^{\mathrm{IID}}_k,
\]

剩余一半预算再根据各自的 `v_proxy` 分配。四者唯一的计划内差别就是 `v_proxy` 来自哪里。

## 7. 六条路径的精确定义

| Path | `v_proxy` 或坐标定义 | 排除的替代解释 |
|---|---|---|
| `IID` | 六组共用原 cosine nominal SNR | 标准 diffusion reference |
| `CW_GROUP` | 用正确 `v_k` 做六组 whitening，再在 whitened space 使用预算匹配的 scalar SNR | 是否条件白化已经足够 |
| `FIXED_BAND` | 每个 outer-train 的 rank-weighted 平均 N1 variance，跨天气固定 | 是否只需静态频带差异 |
| `MULAN_LITE` | 小网络从正确条件学习六个 bounded proxy variances | generic learned conditional schedule 是否足够 |
| `PA_RWF` | 同一天正确的 N1 variance | 本项目的 predictability-aligned 假说 |
| `PA_SHUFFLE` | 同一 partition 内 wrong-day N1 variance | 正确 NWP 与 schedule 的对应关系是否重要 |

### 7.1 统一 proxy allocation

主候选以及三个对照共享：

\[
a_k^{\mathrm{proxy}}
=v_k^{\mathrm{proxy}}
\exp\{-2\eta I_k^{\mathrm{proxy,IID}}\},
\qquad \eta=0.5.
\]

然后只对真实当天的剩余预算 `C=(1-eta)B` 做 reverse water-filling：

\[
J_k=\frac12\left[\log\frac{a_k^{\mathrm{proxy}}}{\theta}\right]_+,
\qquad
\sum_kw_kJ_k=C,
\]

最终：

\[
I_k=I_{k,\mathrm{res}}+J_k,
\qquad
\rho_k=\frac{e^{2I_k}-1}{v_k}.
\]

`PA_RWF` 取 `v_proxy=v_true`，必须逐值重放 G0-B0。`FIXED_BAND`、`MULAN_LITE` 和 `PA_SHUFFLE` 仍使用真实当天 `v_true` 计算总预算与保留部分，因此它们也不能偷偷改变总信息量。

### 7.2 `CW_GROUP` 的边界

`CW_GROUP` 使用：

\[
u=\sum_k\frac{Q_k(r)}{\sqrt{v_k}}.
\]

令每个 whitened group 的 information 都等于当天总预算 `B`，所以加权和仍为 `B`。网络在 whitened coordinates 中预测 clean target，再固定 unwhiten 回 raw residual 后计算共同 loss。

它是六组对角近似下的 budget-matched CW control，不是完整条件协方差，也不声称复现 CW-Gen 全模型。

### 7.3 `MULAN_LITE` 的边界

MuLAN-lite schedule head 为：

```text
47 correct condition features
→ Linear(47,16) → SiLU → Linear(16,6)
```

共 870 个参数。它输出：

\[
v_k^{\phi}
=\bar v_k\exp\{\log(4)\tanh g_{\phi,k}(c)\},
\]

因此 proxy 被限制在 fixed variance 的 `[1/4,4]` 内。最后一层零初始化，所以 update 0 必须与 `FIXED_BAND` 完全相同。

该 head 与 denoiser 联合训练。它不直接为 250 个 timestep 输出自由参数；时间变化由共同 allocation 公式产生。因此它是容量受控的 MuLAN-lite，而不是完整 MuLAN 复现。

### 7.4 `PA_SHUFFLE` 真正打乱什么

所有 denoiser 始终看到正确的当天：

- residual target；
- NWP condition；
- active mask；
- 正确 N1 variance condition features。

Shuffle 只把 schedule 公式中的 `v_proxy` 换成 wrong-day variance；当天总预算、保留信息和 effective-rank weights 都保持正确。训练 subset 和 outer-test 各有四个固定 derangement，固定点必须为 0。

## 8. 共同 corruption 与 tiny denoiser

Raw-coordinate 路径使用同一个 Gaussian noise：

\[
x_t=\sum_kQ_k\left(
\sqrt{\bar\alpha_k}r+
\sqrt{1-\bar\alpha_k}\epsilon
\right).
\]

每个 path 的网络都看到：

- 240 个 noisy residual values；
- 240 个 active-mask bits；
- 47 个正确条件特征；
- 6 个 allocated log-SNR；
- 16 个 Fourier timestep features。

总输入为 549 维。网络固定为：

```text
LayerNorm(549)
→ Linear(549,64) → SiLU
→ Linear(64,64) → SiLU
→ Linear(64,240)
```

基础参数量精确为 56,058。所有路径在同一 fold/fraction/seed 下共享相同基础参数初值。MuLAN-lite 的 870 个 schedule 参数是唯一额外容量，占 1.552%，低于冻结上限 2%；这是给 generic learned baseline 的有利条件，不是候选方法的优势。

网络直接预测 clean residual。训练目标统一为 unwhiten 后的 active-cell raw-residual MSE。Raw training loss 不进入正式结论。

## 9. 三档数据量与 324 个 retained runs

每个 outer-train 使用确定性、嵌套的日期子集：

| 数据量 | 每折实际天数 |
|---:|---:|
| 25% | 56 |
| 50% | 111 或 112 |
| 100% | 222 或 223 |

日期由冻结 SHA256 排序规则选择，所有 18 个 subset 的日期哈希已写入机器配置。50% 是主数据量；25% 和 100% 用于预注册 learning curve。

正式训练总数：

```text
6 outer folds
× 3 data fractions
× 6 paths
× 3 model seeds
= 324 retained runs
```

每个 run 固定：

- 1,024 optimizer updates；
- batch 16 calendar days；
- 16,384 次 day-corruption exposure；
- AdamW，LR `3e-4`，weight decay `1e-4`；
- gradient clip 1；
- EMA decay 0.999；
- FP32、无 AMP、确定性算法；
- 不 early-stop，不选择 best checkpoint；
- 正式权重只能是 update 1,024 的 final EMA。

固定更新次数意味着 learning curve 比较的是“相同优化计算量下，增加独立日期是否改变 denoising risk”，而不是固定 epoch 曲线。

同一 fold/fraction/seed 的六条路径共享基础参数、day minibatches、timesteps、Gaussian noise 和更新次数。

## 10. 冻结 evaluation bank

所有 324 个训练 run 完成并写入 training freeze 后，才允许构造 outer-test evaluation bank。

固定 16 个 timestep：

```text
0, 17, 33, 50, 66, 83, 100, 116,
133, 149, 166, 183, 199, 216, 232, 249
```

每个 held-out day、每个 timestep 使用四个固定 Gaussian noise replicate。底层 noise tensor 在所有 path、fraction 和 model seed 间共同。Shuffle 额外平均四个固定 wrong-day schedules。

## 11. 正式指标

每个 held-out calendar day 报告：

1. `cell_MSE`：observed interior residual 的逐格 MSE；
2. `increment_MSE`：同一区域相邻两小时均为 active 时的 residual increment MSE；
3. `joint_day_normalized_SSE`：全天 10×24 联合误差，以 outer-train residual second moment 标准化；
4. 六组 projected reconstruction MSE；
5. Gaussian reference risk；
6. model group-risk / Gaussian reference-risk ratio。

前三个指标分别回答逐格、爬坡变化和整日联合轨迹。为避免某一个量纲支配，主综合量定义为：

\[
\mathrm{BRR}
=\frac13\left(
\frac{R_{cell}}{\bar R^{IID}_{cell}}+
\frac{R_{inc}}{\bar R^{IID}_{inc}}+
\frac{R_{joint}}{\bar R^{IID}_{joint}}
\right),
\]

其中 IID 分母按相同数据 fraction 使用全部 267 个 out-of-fold day 预先计算，并在 bootstrap 中固定。

三档 BRR 对 `log(data fraction)` 做 trapezoidal average，得到 learning-curve AULC；越低表示相同更新预算下越省数据。

## 12. 为什么必须同时看 Gaussian-reference-adjusted 指标

reserve-then-water-fill 本来就是按 Gaussian posterior MSE 推导的，所以它在同一 Gaussian 代理量上优于 IID 并不是新的实验证据。

因此 G0-B 同时要求：

- 真实 held-out residual 的 BRR 改善；
- model-to-Gaussian reference ratio 也改善。

如果只有 raw reconstruction 改善，而 reference-normalized efficiency 没有改善，正式状态是：

```text
G0_B_ORACLE_ONLY_NO_LEARNING_GAIN
```

这表示解析重分配生效，但“降低有限模型学习难度”的核心假说没有得到支持。

## 13. 推断单位与多重比较

推断单位只能是 calendar day：

1. 先在同一天平均 timestep、noise replicate 和 model seed；
2. 六折 outer-test 合并为 267 条 out-of-fold day records；
3. 按 calendar-month cluster 做 10,000 次 paired bootstrap；
4. 对全部注册的 superiority 与 non-inferiority contrasts 使用 max-T simultaneous 95% bands。

不得把 240 cells、16 timesteps、noise replicate、mode、fold 或 seed 当作独立样本扩大显著性。

## 14. `PA_RWF` 的全部 Go 门

所有工程与数据门先通过，然后必须同时满足：

| 比较或稳定性门 | 冻结要求 |
|---|---:|
| 相对 IID，50% BRR | 点改善 ≥2%，simultaneous lower bound >0 |
| 相对 Fixed-band，50% BRR | 点改善 ≥1%，lower bound >0 |
| 相对 CW-group，50% BRR | 点改善 ≥1%，lower bound >0 |
| 相对 Shuffle，50% BRR | 点改善 ≥1%，lower bound >0 |
| 相对 MuLAN-lite，50% BRR | 非劣 margin 1% |
| 相对 MuLAN-lite，AULC | 点改善 ≥2%，lower bound >0 |
| 相对 IID，oracle efficiency | 点改善 ≥1%，lower bound >0 |
| 相对 IID 的 cell/increment/joint | 每项非劣 margin 1% |
| 三个单项中实质改善 ≥2% | 至少 2 项 |
| PA vs IID 为正的 outer folds | 至少 5/6 |
| PA vs IID 为正的 model seeds | 至少 2/3 |

可选但不影响 Go 的强结果是：`PA_RWF-50%` 对 `IID-100%` 在 1% margin 内非劣。

全部通过才记为：

```text
G0_B_PREDICTABILITY_ALLOCATION_UTILITY_GO
```

该 Go 只授权另行设计完整模型协议，不授权 selection，也不等于 Diffusion 已战胜 Flow。

## 15. 失败结果也必须能回答问题

决策按冻结顺序执行：

1. lineage、泄漏、schedule、有限值、resume 或 run-count 失败：`G0_B_TECHNICAL_NO_GO`；
2. PA 未通过 IID 门时，若 MuLAN/CW/Fixed 中某条以 2% 门槛胜 IID，则分别记录 generic adaptivity、whitening 或 fixed-band sufficient；否则为 `G0_B_STRUCTURED_SCHEDULE_NO_GO`；
3. PA 不胜 Shuffle：`G0_B_NWP_ALIGNMENT_NO_GO`；
4. PA 不胜 Fixed：`G0_B_FIXED_BAND_SUFFICIENT`；
5. PA 不胜 CW：`G0_B_WHITENING_SUFFICIENT`；
6. PA 未在 AULC 上胜 MuLAN，且 MuLAN 实质胜 IID：`G0_B_GENERIC_ADAPTIVITY_SUFFICIENT`；否则 `G0_B_UNRESOLVED`；
7. raw 门通过但 oracle-efficiency 门失败：`G0_B_ORACLE_ONLY_NO_LEARNING_GAIN`；
8. 只有全部门通过才是正式 Go。

任何非 Go 状态都不授权完整 predictability-aligned diffusion。

## 16. P0 启动门

协议冻结后，下一步只能实现代码并执行 P0。P0 固定使用 outer fold 0、25% subset、seed 3，六条路径各做 50 次随后丢弃的 train-only update。

P0 必须验证：

- 所有 lineage、fold 和 18 个 subset hash；
- G0-A nuisance replay；
- 六组 projector 的正交完备性；
- 六条路径逐日逐步 budget equality、SNR 正值、有限值、endpoint 和单调性；
- PA 与 G0-B0 逐值一致；
- CW whiten/unwhiten identity；
- MuLAN update 0 与 Fixed 完全一致；
- Shuffle 没有固定点且只改变 schedule proxy；
- 六路径基础网络初值与随机张量配对；
- loss、gradient、optimizer、EMA、固定 eval batch 和 checkpoint resume；
- 峰值显存不超过 2 GiB；
- P0 目录不保留 `.pt` 或 `.pth` 权重；
- 除 train 外没有 materialize 其他 target role。

只有：

```text
G0_B_TINY_DENOISER_P0_GO
```

才授权 324 个 retained runs。P0 失败必须新建 revision，不能修改本协议后原地继续。

## 17. 当前执行边界

本次协议冻结只授权：

1. 实现 `g0b_tiny_denoiser.py`；
2. 编写 synthetic/protocol tests；
3. 建立 dry-run；
4. 执行随后丢弃权重的 P0。

当前没有授权 retained training，更没有授权 outer-test 正式评估。正式 outer-test evaluation 只能在 324/324 run 完成并写入不可变 training freeze 后另行启动。
