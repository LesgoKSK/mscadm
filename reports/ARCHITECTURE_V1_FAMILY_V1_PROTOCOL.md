# Architecture-v1 Family-v1：Flow vs Joint-DDPM 公平比较协议

冻结日期：2026-08-26  
配置：`repro_configs/architecture_v1_family_v1.json`  
配置 SHA-256：`d084852e74839b6aba0703cd6bc27388f73dad7ca52bf731a49ec89ac91c59e4`  
状态：在任何 family-v1 smoke、pilot 或正式训练前冻结。  
证据属性：validation-informed 模型家族开发，不是独立确认实验。selection/calibration 继续封存。

## 1. 本轮究竟比较什么

只比较两个没有新增 recurrence 的联合生成模型：

- **F0：Joint Rectified Flow**；
- **D0：Joint DDPM**。

二者都生成完整的 `10 zones × 24 hours` 日轨迹，一个 ensemble member 是一个 240 维联合样本。旧仓库中的 DDPM 是逐 zone-day 独立生成，不能作为 family-v1 的 D0。

family-v1 不重新讨论 v3.2 已经 No-Go 的 T1-feature、T1-source，也不把二者组合起来。若 F0/D0 都无法稳定通过，下一条路线才是概率状态空间模型或共同训练稳定性修订。

## 2. 公平性核心

| 项目 | F0 | D0 |
|---|---|---|
| E/A | 同一冻结 checkpoint | 同一冻结 checkpoint |
| 输出 | 10×24 联合轨迹 | 10×24 联合轨迹 |
| atom | 同一概率与 allocation | 同一概率与 allocation |
| 连续坐标 | interior raw logit | interior raw logit |
| 网络 | JointTimeDomainVelocity | 完全同一张量图作为 epsilon 网络 |
| 总参数 | 796,837 | 796,837 |
| 正式种子 | 3/4/5 | 3/4/5 |
| 每个种子初始张量 | 与 D0 完全一致 | 与 F0 完全一致 |
| 每次训练更新 | 1 次网络前向+反向 | 1 次网络前向+反向 |
| 主采样 NFE | 31 | 31 |
| 主采样器 | 16-step Heun | 31-step DDIM, eta=0 |
| 场景成员 | 100 | 100 |
| 评分与 bootstrap | calendar day | calendar day |

区别只保留在生成动力学：F0 学速度场，D0 学 Gaussian diffusion 的噪声。

## 3. 为什么不能直接复用旧 DDPM

旧 `ConditionalDDPMDenoiser` 和 `GaussianDiffusion` 存在四个不匹配：

1. 逐区域序列输出，不是 240 维联合生成；
2. denoiser 参数量和 Flow 网络不同；
3. 没有 exact atom/active-coordinate 语义；
4. DDIM 中会把预测 clean latent 截到 `[-5,5]`，改变尾部与边界行为。

family-v1 只允许复用 cosine beta schedule 的数学并做交叉测试；正式 runtime 在 `architecture_v1` 内重新实现，旧 DDPM runtime 不参与结果。

## 4. 共同连续潜变量语义

观测到的 interior 功率先变换为：

\[
x_0=\operatorname{logit}(\operatorname{clip}(y,10^{-4},1-10^{-4})).
\]

F0 和 D0 都使用这个未经额外标准化的 raw logit 坐标。train-only 的 logit mean/std 只属于 atom location auxiliary，不允许 D0 单独用作 transport 标准化。

在 0/1 atom 或缺失位置：

- 连续 latent 在训练、网络输出和每个采样步骤后都必须精确为 0；
- loss 不包含这些格点；
- 最终通过共同 atom reconstruction 精确恢复 0/1。

D0 不允许内部裁剪 predicted x0；出现非有限 latent 时直接失败并保存 bundle，不能用 clip 掩盖。

## 5. D0 的冻结定义

- 250 个训练 diffusion timesteps；
- cosine schedule，offset=0.008；
- epsilon prediction；
- fixed posterior variance，不学习 variance；
- timestep 按 calendar day 均匀采样 0–249；
- 网络时间输入为 `k/249`，因此复用 Flow 的 Fourier time embedding；
- masked epsilon MSE，只在 observed interior cells 上平均；
- 主采样为 deterministic DDIM，31 steps，eta=0；
- 31 个离散点来自包含 0 和 249 的 rounded linspace；
- 无 CFG、self-conditioning、learned variance 或 sampler guidance。

## 6. 为什么两边都重新训练

v3.2 的 T0 三个 seed 分别通过质量门，但 level/ramp 的种子跨度略超预锁上限。若直接拿旧 T0 对新 DDPM，会把“旧训练稳定性”与“模型家族”混在一起。

因此 family-v1 使用新的 model seeds `[3,4,5]`，F0/D0 在每个 seed 内共享完整初始张量。二者共同使用 EMA：

- decay=0.999；
- 初始化为 online transport 权重的精确副本；
- 每个 optimizer update 后更新；
- validation、early stopping 和 sampling 只使用 EMA；
- online 权重只在 seed3 做非选择性诊断；
- E/A 冻结，不做 EMA。

这是看到 v3.2 种子漂移后的共同工程修订，因此仍是 validation-informed，不伪装成独立验证。

## 7. 训练协议

- AdamW，LR=1e-4，weight decay=0；
- batch=16 calendar days，即每 epoch 17 updates；
- 最大 720 epoch / 12,240 updates；
- 至少 120 epoch；
- 每 10 epoch 在固定 K=4 validation bank 上验证；
- patience=12；
- gradient clip=5；
- FP32、无 AMP、确定性算法；
- 完整 epoch 边界断点恢复，必须恢复 online/EMA/optimizer/RNG 和全部 identity hashes。

F0 和 D0 的 validation loss 数值不可横向比较，只用于各自 checkpoint 早停。最终只能比较训练外的 proper scores 和可靠性。

## 8. 启动正式训练前的 P0

先执行 synthetic tests 和 train-only、丢弃权重的 50-update preflight。P0 不加载 validation targets，必须验证：

- F0/D0 参数键、形状、数量和初始哈希完全一致；
- masked q-sample 与闭式公式一致；
- DDIM 每一步 inactive latent 精确为 0；
- exact atom reconstruction；
- EMA 更新与 resume 可精确重放；
- 同 seed DDIM replay 完全一致；
- member chunk 数值等价；
- loss、gradient、optimizer、sampler 全 finite；
- 两边一次更新均只调用网络一次；
- GPU peak memory <4.5 GiB。

P0 失败时不得启动正式训练，必须新建协议 revision。

## 9. 采样预算和效率曲线

主比较固定为 31 NFE：

- F0：16-step Heun = 31 network forwards；
- D0：31-step DDIM = 31 network forwards。

次级、非选择性的质量—效率曲线：

| NFE | F0 | D0 |
|---:|---|---|
| 9 | 5-step Heun | 9-step DDIM |
| 17 | 9-step Heun | 17-step DDIM |
| 31 | 16-step Heun | 31-step DDIM |
| 65 | 33-step Heun | 65-step DDIM |

次级曲线不能覆盖主 31-NFE 结论或用于事后挑选 sampler。

## 10. 指标与推断

主要 proper scores：

- level CRPS；
- ramp CRPS；
- normalized joint ES；
- lagged increment variogram score。

可靠性与解释指标：coverage90、width90、zero/atom Brier、lag-1 correlation、cross-lag、NWP regime、member collapse。

效率指标：训练墙钟、同步采样墙钟、显存、实际 NFE。

三个 training seed 的同日分数先平均，再做 family paired contrast。bootstrap 单位只能是 calendar day，重复 5,000 次；成员、区域、小时都不能当独立样本。

## 11. family 决策门

每个 family 必须先通过：

- 三个 seed 各自的有限值、coverage、width、CRPS/ES 和 atom 语义硬门；
- level CRPS 种子跨度 ≤0.0025；
- ramp CRPS 种子跨度 ≤0.0012；
- joint ES 相对种子跨度 ≤3%；
- coverage 种子跨度 ≤0.04。

一个 family 只有在以下条件同时满足时才能支配另一个：

1. 自身 eligible；
2. 在 level、ramp、joint、lagged、coverage、width 全部非劣；
3. 至少一个 proper-score endpoint 达到实际重要改善且 paired CI 支持优效。

实际重要阈值为：level 0.001、ramp 0.001、joint ES 相对 2%、lagged VS 相对 5%。

若两者在所有端点等效，只能在 31 NFE 下采样墙钟快至少 20% 且 CI 支持时用效率打破平局；否则结论是 unresolved。

## 12. 可能结论

- D0 dominates：保留 Joint DDPM；
- F0 dominates：保留 Joint Rectified Flow；
- 两者等效且效率门通过：保留更快者；
- 只有一方稳定但无法对另一方非劣：unresolved；
- 两者都不稳定：转向概率 SSM 或共同训练稳定性 revision；
- 有明显分数权衡、没有 dominance：unresolved，不能用主观偏好选模型。

family-v1 runner 无权打开 selection。只有得到清晰 winner 后，才冻结 winner，并建立单独的一次性 selection 状态机。

## 13. 实现边界

计划新增：

- `architecture_v1/family_diffusion.py`
- family-aware fixed validation bank 与 common EMA 支持
- `repro_scripts/run_architecture_v1_family_v1.py`
- `tests/test_architecture_v1_family_diffusion.py`
- `tests/test_architecture_v1_family_v1_runner.py`

在这些实现、P0 和身份哈希通过前，不启动 6 个正式训练。
