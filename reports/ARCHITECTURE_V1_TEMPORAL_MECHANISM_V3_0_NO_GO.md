# Architecture v1 时间机制 v3.0：自由随机源 No-Go

## 结论

v3.0 的“可自由学习 residual correction 的条件随机源”在第一个正式 T0/seed0 上触发硬失败，因此整个五候选 panel 已提前停止并冻结为：

`V3_0_NO_GO_FREE_SOURCE_COLLAPSE`

这不是对 Flow、SSM 或 T1-source 总方向的否定，只是否定当前这一个没有约束随机源均值和尺度的实现。

## 为什么停止

source adapter 与 Flow 联合训练后，固定 validation transport loss 显著下降，但 M=100 场景集合的宽度和覆盖率同时崩塌：

| 指标 | T0/seed0 | 判定 |
|---|---:|---|
| coverage90 | 0.5065 | 严重不足 |
| width90 | 0.1713 | 场景过窄 |
| level-CRPS | 0.1000 | 越过质量门 |
| normalized joint-ES | 0.1328 | 越过质量门 |
| ramp-CRPS | 0.0546 | 未改善到可接受状态 |
| gradient clip fraction | 47.19% | 超过 25% 上限 |

训练 loss 变小并不等于概率分布变好。这里的 adapter 学会了改变基础随机源的条件均值和尺度，使 transport 任务变容易，却牺牲了 ensemble diversity 和 reliability。

## 两项工程审计修正

正式运行前两次早期尝试还暴露了两个实现问题，均在形成正式结果前停止并隔离：

1. fixed validation bank 最初没有调用 candidate-specific source transformation；
2. 旧 flow-stage optimizer 最初未纳入新增的 source cell/source adapter。

两个问题都已增加回归测试。相关无效权重只保留在 `/tmp` 审计目录，未进入当前 No-Go 指标，也未触碰 selection/calibration。

## v3.1 修正方向

下一版不再允许 source 网络自由学习均值和尺度，而采用有解析稳定性的条件 AR 状态源：

\[
z_t = \rho_t z_{t-1} + \sqrt{1-\rho_t^2}\,\epsilon_t,
\qquad \epsilon_t\sim\mathcal N(0,1),
\qquad |\rho_t|\le 0.95.
\]

只学习相关系数 `rho`：

- 不学习 source mean；
- 不学习 source marginal scale；
- 在连续 active path 内保持标准正态边缘方差；
- 遇到 atom/missing 位置重置递归，不在 atom 坐标虚构连续 latent；
- shuffle 对照只改变 AR 访问顺序；
- T0 固定 `rho=0`，即严格 IID Gaussian source。

这样 T1-source 只能学习“时间相关结构”，不能靠缩窄场景集合降低 loss。

## 数据边界

- selection 未访问；
- calibration 未访问；
- 本轮只使用 train/validation；
- v3.0 No-Go 不授权任何论文性能主张。

