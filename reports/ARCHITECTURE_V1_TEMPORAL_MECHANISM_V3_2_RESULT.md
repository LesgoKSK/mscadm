# Architecture-v1 时间机制 v3.2 最终结果

完成日期：2026-08-26  
冻结状态：`V3_2_T1_NO_GO`  
证据范围：train/validation 架构开发；selection/calibration 始终封存。  
正式结果：`outputs/architecture_v1_temporal_mechanisms_v3_2/V3_2_RESULT.json`

## 一句话结论

两种 chronological 时间机制都显著改善了 ramp 与 lagged dependence，也都显著优于对应 Shuffle；但是它们没有同时通过预锁的三种子稳定性和 level/joint 非劣门。因此 v3.2 的正式结论是：**不把 feature recurrence 或 AR source recurrence 纳入下一版主模型；以无新增 recurrence 的 T0 作为保守的 Flow 控制，进入 matched joint DDPM 模型家族比较。**

这不是“时间机制完全无效”。它表示目前的收益尚不足以在可靠性、联合分布质量和种子稳定性约束下成为可保留主张。

## 1. 完成与审计状态

- 12/12 个新训练完成；
- 3 个 v3.1 seed0 checkpoint 按冻结哈希复用；
- 15/15 个 candidate×seed 完成 M=100、3 个 sampling seed 的 validation 评估；
- 15/15 单种子硬门通过；
- 无 failure bundle、NaN 或 GPU 异常；
- 158 个 SHA-256 sidecar 全部匹配；
- selection/calibration 访问标志全部为 false；
- 最终 `training.freeze.json` 与结果哈希一致。

## 2. 三种子平均机制信号

相对 T0，T1-feature：

- ramp CRPS 改善 0.002162，95% paired CI [0.001593, 0.002796]；
- lagged increment variogram 相对改善 19.9%，CI [14.2%, 26.1%]；
- 对应 feature-shuffle 比 chronological 的 ramp 差 0.003250，CI [0.002537, 0.004048]；
- feature-shuffle 的 lagged score 比 chronological 差 0.004832，CI [0.003930, 0.005750]。

相对 T0，T1-source：

- ramp CRPS 改善 0.002697，95% paired CI [0.002122, 0.003271]；
- lagged increment variogram 相对改善 62.2%，CI [44.1%, 87.4%]；
- 对应 source-shuffle 比 chronological 的 ramp 差 0.001510，CI [0.001155, 0.001853]；
- source-shuffle 的 lagged score 比 chronological 差 0.000940，CI [0.000611, 0.001271]。

因此，“正常时间顺序携带有效信息”这一机制信号很清楚；No-Go 的原因不在主指标或 Shuffle 对照，而在非劣与跨种子稳定性。

## 3. 为什么 feature 没有通过

T1-feature 通过：

- ramp 实际与统计优效；
- lagged score 实际与统计优效；
- chronological 优于 feature-shuffle；
- coverage 与 width 非劣；
- level、ramp、coverage 的三种子稳定性。

但失败于：

1. level CRPS 非劣上界为 0.002667，高于预锁 margin 0.0015；
2. joint ES 相对非劣上界为 3.01%，高于预锁 margin 2%；
3. 三种子 normalized joint ES 相对跨度为 3.014%，略高于稳定性上限 3%。

feature recurrence 是最接近通过的候选，但协议禁止在看到结果后放宽 2%/3% 门槛，因此正式结论仍为 No-Go。

## 4. 为什么 source 没有通过

T1-source 通过：

- ramp 和 lagged score 优效；
- chronological 优于 source-shuffle；
- joint ES、coverage、width 的配对非劣；
- ramp 与 coverage 的三种子稳定性；
- source mean/variance 和 atom latent 语义硬门。

但失败于：

1. level CRPS 非劣 CI 上界为 0.001647，略高于 margin 0.0015；
2. 三种子 level CRPS 跨度 0.005689，高于 0.0025；
3. 三种子 normalized joint ES 相对跨度 7.02%，高于 3%；
4. 三个 chronological seed 的 `abs(rho)>=0.94` 比例约为 99.78%–99.88%，几乎全场饱和；source-shuffle 也有约 97.2%–97.4% 饱和。

虽然 chronological 明显优于 Shuffle，但几乎全场 `rho` 触顶意味着当前 source 模块更像“强制高度平滑的相关源”，条件自适应解释较弱；再加上约 3 倍训练成本，不适合作为下一阶段默认主干。

## 5. T0 本身的稳定性警告

T0 的三个 seed 均分别通过单种子质量硬门，但整体稳定性未完全通过：

- level CRPS 跨度 0.002595，略高于上限 0.0025；
- ramp CRPS 跨度 0.001474，高于上限 0.0012；
- coverage 与 joint ES 稳定性通过。

因此 T0 只能称为“无新增时间机制的保守控制”，不能称为已经稳定获胜的最终模型。后续 matched joint DDPM 必须沿用三种子与 calendar-day 配对协议；如果 DDPM 也出现类似种子漂移，应优先处理共同训练/选择噪声，而不是继续堆新模块。

## 6. 正式决策

按照冻结分支规则：

1. 不保留 T1-feature 作为下一版主模型；
2. 不保留 T1-source，也不增加 feature+source 组合；
3. 保留 T1-feature 的结果作为有价值的探索性证据和未来辅助消融；
4. 以 T0/common E/A 作为 matched Flow 控制；
5. 下一阶段实现相同联合输出语义、容量和训练协议的 joint DDPM；
6. selection/calibration 继续封存，直到 Flow/DDPM 候选、代码、门槛和分支规则全部冻结。

## 7. 下一步

下一阶段不再继续调 v3.2 阈值或重跑 seed。应建立 architecture-v1 family-v1 协议：

- F0：当前 T0 joint Rectified Flow 控制；
- D0：matched joint DDPM，无新增 recurrence；
- 可选 D1-feature 只在协议中预先注册为次级机制对照，不能因 v3.2 结果临时选择性加入；
- 共用 E/A、10×24 联合输出、active-coordinate/atom 语义、数据角色、模型种子和场景评分；
- 同时报告预测分数、可靠性、NFE、显存和墙钟时间。

只有完成模型家族比较后，才决定最终采用 Flow、Diffusion，或因两者共同不稳定而回到概率状态空间模型。
