# Architecture-v1 v3.1 时间机制小试结果

日期：2026-08-21  
状态：`PILOT_GO_EXPAND_SHUFFLES_AND_THREE_SEEDS`  
证据范围：只使用 architecture-v1 的 train/validation；selection 与 calibration 仍封存。

## 一句话结论

v3.1 已经解决 v3.0 的场景宽度塌缩问题。单种子结果同时支持两种可继续验证的时间机制：

- 把时间记忆放在 NWP 条件特征中，训练成本与 T0 基本相同，并明显改善 ramp 与 lagged dependence；
- 把时间相关性放在方差保持的 AR 随机源中，依赖分数改善更大、可靠性保持更好，但训练成本约为前者的 3.3 倍，而且学习到的相关系数触及预设上限。

因此现在还不能选最终模型。下一步必须完成三种子与 chronological-vs-shuffle 对照，判断改善究竟来自真实时间顺序，还是仅来自额外参数/平滑效应。

## 1. 为什么需要 v3.1

v3.0 使用可自由学习的残差随机源。它可以同时改变随机源的均值、相关性和尺度，结果虽然降低了 Flow 拟合损失，却把场景集合压得过窄：

- coverage90：0.507；
- width90：0.171；
- 梯度裁剪比例：47.2%。

这属于“通过牺牲不确定性来优化训练目标”，不能接受。

v3.1 将随机源改成：

\[
z_t=\rho_t z_{t-1}+\sqrt{1-\rho_t^2}\,\epsilon_t,
\qquad
\rho_t=0.95\tanh(g(c_t)).
\]

其中 `epsilon_t` 为标准高斯。模型只能学习相关系数 `rho_t`，没有随机源均值头，也没有尺度头。只要前后两个位置都属于连续 interior，边际方差理论上保持为 1；遇到 0/1 原子或缺失位置时，连续潜变量严格为 0，并切断 AR 传播。

## 2. 公平比较设置

三个候选共享：

- 同一 NWP encoder E；
- 同一完整 atom 模块 A；
- 同一 Flow velocity 网络；
- 完全相同的参数键、参数形状和初始张量；
- 同一训练日期顺序、Flow path 随机数和固定 validation bank；
- 相同学习率、batch、最大 epoch、早停和梯度裁剪；
- M=100、16-step Heun、31 次 Flow forward；
- 同一 atom allocation 与采样随机种子。

候选区别只有：

- T0：新增 cell 为 memoryless，随机源为 IID 标准高斯；
- T1-feature：按 1–24 小时顺序递推 NWP 条件特征，随机源仍为 IID 标准高斯；
- T1-source：NWP 特征不递推，只让随机源按有界、方差保持的条件 AR(1) 递推。

## 3. 单种子结果

| 指标 | T0 | T1-feature | T1-source |
|---|---:|---:|---:|
| level CRPS ↓ | 0.080835 | **0.080363** | 0.080492 |
| ramp CRPS ↓ | 0.051429 | **0.049307** | 0.049383 |
| lagged increment VS ↓ | 0.019119 | 0.014905 | **0.011510** |
| normalized joint ES ↓ | 0.112316 | **0.111816** | 0.112315 |
| coverage90 | 0.8526 | 0.8328 | 0.8503 |
| width90 | 0.3921 | 0.3689 | 0.3929 |
| 梯度裁剪比例 | 13.4% | 15.5% | **12.7%** |
| 完成 epoch | 270 | **260** | 720 |
| 同步训练耗时 | 12.5 min | **12.2 min** | 40.6 min |
| 最佳 epoch | 150 | 140 | 620 |

相对 T0：

- T1-feature 的 ramp CRPS 改善 0.002123，lagged increment VS 相对改善 22.0%；
- T1-source 的 ramp CRPS 改善 0.002046，lagged increment VS 相对改善 39.8%；
- 两者 level CRPS 与 joint ES 均未恶化；
- T1-feature 的区间略窄、coverage 下降约 0.020，但仍通过预锁安全门；
- T1-source 基本保持 T0 的 coverage 与 width。

这些是同一 validation 上的单种子工程证据，不是最终统计结论，也不能当作外部确认结果。

## 4. 方差保持审计

每个模型均在 1,094,220 个 active source 坐标上进行审计：

| 审计项 | T0 | T1-feature | T1-source |
|---|---:|---:|---:|
| source mean | 0.000806 | 0.000806 | 0.000113 |
| source variance | 1.000876 | 1.000876 | 0.999140 |
| inactive latent max abs | 0 | 0 | 0 |
| max abs rho | 0 | 0 | 0.94999999 |

这说明 v3.1 没有重演 v3.0 的尺度塌缩，atom 位置也没有被虚构连续潜变量。

需要特别注意：T1-source 的 `abs(rho)` 已触及 0.95 上限。它不是本轮硬门失败，因为上限是提前锁定且边际方差仍为 1；但它是一个必须由多种子和 shuffle 对照审查的风险信号。如果 chronological 与 shuffled source 都同样触顶并同样改善，就不能声称模型学到了真实物理时间顺序。

## 5. 当前能说和不能说什么

当前可以说：

- 显式时间动态是值得继续验证的；
- feature recurrence 是低成本、表现稳定的优先候选；
- 方差保持 AR source 能更强地修复 lagged dependence，同时避免场景带宽塌缩；
- Flow 仍是可行传输主干，没有证据要求立即换成 diffusion。

当前不能说：

- T1-feature 或 T1-source 已经成为最终模型；
- source recurrence 优于 feature recurrence；
- Flow 已经优于 diffusion；
- 单种子 validation 改善能够代表外部泛化；
- `rho` 触顶代表真实风电物理传播。

## 6. 下一阶段的固定顺序

1. 加入 T1-feature-shuffle 与 T1-source-shuffle，保持同一冻结的 24 小时排列。
2. 对 T0、T1-feature、T1-source 及两个 shuffle 控制运行 seeds 0/1/2。
3. 以 calendar day 为配对单位，检验：
   - T1 相对 T0 的 ramp CRPS 实际改善至少 0.001，且配对区间支持优效；
   - lagged increment VS 相对改善至少 5%；
   - chronological T1 必须优于对应 shuffle；
   - level CRPS、joint ES、coverage 与 width 通过预锁非劣门；
   - 三种子稳定性通过。
4. 只有 chronological 与 seed 门都通过，才保留该时间机制。
5. 完成机制选择后再实现 matched joint DDPM；在同一 E/A、联合输出语义、容量和采样预算下比较 Flow 与 diffusion。

selection/calibration 在上述候选、代码、门槛和分支规则全部冻结前保持封存。

## 7. 可复核产物

- 冻结配置：`repro_configs/architecture_v1_temporal_mechanisms_v3_1.json`
- 模型实现：`architecture_v1/model.py`
- 训练入口：`repro_scripts/run_architecture_v1_temporal_mechanisms_v3_1_pilot.py`
- 小试总结果：`outputs/architecture_v1_temporal_mechanisms_v3_1_pilot/PILOT_RESULT.json`
- 冻结记录：`outputs/architecture_v1_temporal_mechanisms_v3_1_pilot/training.freeze.json`
- 语义测试：`tests/test_architecture_v1_temporal_mechanisms_v3_1.py`
- synthetic smoke：`outputs/architecture_v1_temporal_mechanism_v3_1_smoke/SMOKE_RESULT.json`

所有正式小试 JSON、checkpoint、history 与场景 archive 均有 SHA-256 sidecar。最终 QA 校验 23 个 sidecar，0 个缺失或哈希错误；所有 selection/calibration 访问标志均为 false。
