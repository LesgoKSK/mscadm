# Architecture v1：T0/T1 时间机制公平比较协议

## 目标

本阶段不决定 Flow 与 Diffusion 的最终胜负，只回答一个更基础的问题：

> 在已有联合 Rectified Flow 与全轨迹注意力的基础上，显式按小时递推的状态是否能稳定改善 ramp 和相邻增量依赖？

R0-D 已证明公共 Flow 基线具备三种子稳定性。本阶段因此固定 R0-D 的 E/A、数据协议和随机实验条件，只改变新增时间机制。

## 候选模型

五个候选拥有完全相同的参数名称、形状和初始化张量：

| 候选 | 条件特征 cell | 随机源 cell | 作用 |
|---|---|---|---|
| T0 | memoryless | memoryless | 无新增递归状态的参数匹配对照 |
| T1-feature | chronological recurrent | memoryless | 检验时间状态作为 NWP 条件特征 |
| T1-source | memoryless | chronological recurrent | 检验时间状态作为随机动态源 |
| T1-feature-shuffle | shuffled recurrent | memoryless | T1-feature 的负对照 |
| T1-source-shuffle | memoryless | shuffled recurrent | T1-source 的负对照 |

“memoryless”只表示新增 cell 不读取 `h[t-1]`。Flow 主干仍然是 10 区域 × 24 小时的联合模型，并具有轴向全轨迹注意力。

正式尺寸下每个 T0/T1 模型总参数量为 830,117；共享且冻结的 E/A 为 141,763。所有候选总参数量完全相同。T0 中两个 recurrent projection 保留在 checkpoint 中但不进入 forward graph，因此其有效活跃参数量较小且被单独报告。

## 随机源的定义

每个候选都从相同的 IID Gaussian base noise 开始，并通过相同参数拓扑的条件 source adapter：

1. 将标量噪声投影至隐空间；
2. 加入冻结的 NWP 表示；
3. 经 source cell 扫描 24 小时；
4. 输出对基础噪声的 residual correction；
5. 只在 sampled interior active coordinates 上保留连续 source。

精确 0/1 atom 坐标的 source 和最终 continuous latent 始终严格为零。T1-source 只比 T0 多开启 source cell 的 `h[t-1]` 连接。

## Shuffle 负对照

冻结的零基小时顺序为：

`[17, 2, 12, 0, 15, 18, 1, 11, 16, 20, 13, 7, 5, 4, 9, 23, 22, 10, 3, 8, 6, 21, 19, 14]`

负对照按该顺序访问小时，然后把输出还原到原物理小时位置。这样保持参数、输入小时集合和输出布局不变，只破坏递归状态所利用的真实时间邻接。

## 共同实验条件

- 三个模型初始化种子：0、1、2；
- 每个 seed 内五个候选加载完全相同的初始张量 bank；
- 共享固定 E/A；
- 共享 epoch 日期排列 seed；
- 共享 rectified-flow path seed；
- 相同 AdamW、学习率、batch、early stop 与梯度审计；
- validation 使用相同的 4-replicate fixed noise/time bank；
- 场景评价使用 M=100、16 Heun steps、31 flow NFE；
- calendar day 是统计推断单位，member 不是独立重复。

## 主评价指标

主指标必须同时通过：

1. ramp-CRPS：T0 − T1 的逐日配对改善至少 0.001，且 95% bootstrap CI 下界大于 0；
2. lagged increment variogram score：相对改善至少 5%，且 95% bootstrap CI 下界大于 0。

第二项直接评价相邻功率增量的联合依赖，避免只用 Pearson correlation 作为模型选择目标。

## 非劣门

时间机制不能以破坏总体场景质量换取 ramp 改善：

- level-CRPS 上界 margin：0.0015；
- normalized joint-ES 相对上界 margin：2%；
- coverage90 绝对差上限：0.02；
- width90 相对增幅上限：10%。

所有 margin 均使用日级配对 bootstrap 的预注册上界判定。

## 负对照门

真实时间顺序的 T1 必须优于其匹配 shuffle：

- ramp-CRPS 实际改善至少 0.0005；
- ramp 与 lagged increment variogram 的配对 CI 下界均大于 0。

若 T1 与 shuffle 表现相近，则不能声称收益来自真实时间记忆。

## 种子稳定性

每个候选还必须维持 R0-D 已通过的三种子门：

- level-CRPS spread ≤ 0.0025；
- ramp-CRPS spread ≤ 0.0012；
- normalized joint-ES relative spread ≤ 3%。

## 数据边界

本阶段只允许读取 train 和 validation target。以下角色保持 sealed：

- calibration；
- selection；
- R-SEEN；
- final。

即使某个 T1 在 validation 上通过，也不会立即打开 selection。首先还需要冻结 matched joint DDPM 与概率 SSM 的代码、训练预算和分支规则，避免后来实现的模型利用已经查看过的 selection 结果。

## 当前工程状态

- 五个候选模型、共同随机源接口和两个 shuffle 对照已实现；
- 纯合成数据 smoke 为 `SMOKE_PASS`；
- 参数拓扑、初始张量、递归梯度开关、atom latent 和 scenario finite 检查全部通过；
- fixed validation bank 已增加 candidate-specific source transformation，并有回归测试覆盖；
- 正式训练 runner 默认为 dry-run，只有显式 `--execute-training` 才能启动；
- 正式 GPU 训练已按冻结配置启动，最终结果以 `TEMPORAL_MECHANISM_RESULT.json` 和 `training.freeze.json` 为准。

## 关键文件

- 配置：`repro_configs/architecture_v1_temporal_mechanisms_v3_0.json`
- 模型：`architecture_v1/model.py`
- proper-score 评价：`architecture_v1/formal_evaluation.py`
- 配对门控：`architecture_v1/mechanism_evaluation.py`
- 合成 smoke：`repro_scripts/run_architecture_v1_temporal_mechanism_smoke.py`
- 正式 runner：`repro_scripts/run_architecture_v1_temporal_mechanisms_v3_0.py`
- 定向测试：`tests/test_architecture_v1_temporal_mechanisms.py`
