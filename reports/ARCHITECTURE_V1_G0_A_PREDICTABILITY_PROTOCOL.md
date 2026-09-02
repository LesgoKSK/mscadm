# Architecture-v1 G0-A：NWP 模式不确定性审计协议

冻结日期：2026-09-02

文档状态：本文件记录任何 G0-A target-aware 运行之前冻结的协议。G0-A 只允许读取 267 个 `train` 日期；validation、calibration、selection、R-SEEN 和最终外部测试继续封存。

机器可执行配置：[`architecture_v1_g0_a_predictability.json`](../repro_configs/architecture_v1_g0_a_predictability.json)

## 1. 本阶段只回答一个问题

> 在先移除 NWP 可预测的条件均值、并正确排除 0/1 atom 与缺失坐标后，NWP 是否仍能在未参与拟合的 train folds 上稳定预测不同风电时空成分的条件不确定性？

G0-A 不训练 diffusion，也不比较生成场景。它是后续 `Predictability-aligned diffusion` 的前提审计：如果连模式级条件不确定性都不能被 NWP 稳定预测，改变 diffusion path 就没有科学依据。

即使 G0-A 通过，也只允许另行设计并冻结 G0-B；不自动授权完整模型训练。

## 2. 新颖性边界

本项目不再声称以下内容是首创：

- 条件多维噪声日程；MuLAN 已给出 context-adaptive multivariate schedule；
- 条件均值/协方差白化；CW-Diff 已给出相应框架；
- 非各向同性时间序列 forward transition 或频率异步 schedule；MA-TSD 等工作已覆盖邻近思想。

当前待验证的只是一条风电特异科学原则：

> NWP 所揭示的条件可预测性，能否为有限数据下的风电联合 diffusion 提供一种比自由学习 schedule 更有约束、更可解释的 corruption allocation 依据？

G0-A 只验证这条原则的第一个必要前提，不对最终论文新颖性作结论。

## 3. 数据边界

唯一允许的数据入口是：

```python
architecture_v1.data.build_architecture_v1_train_data(...)
```

冻结身份如下：

| 项目 | 冻结值 |
|---|---:|
| train calendar days | 267 |
| train date SHA256 | `f67eb5fd4382b2d69d2d22d23488ec618cd3b3b9547a9bbcc71bdd39587c8cf5` |
| train array SHA256 | `ec756d175edc4b265311e7a3b642a58d84aebd4e266392932ee0a6bb3dd56e3a` |
| train-only bundle SHA256 | `953ffccc9f0241c19757d0aeaf9de98cda2a268efef89b7cf977957ef5c0d7aa` |

运行器必须验证返回对象只有 `train` role，不构造 validation bank，不保留任何学习权重。任一身份漂移立即停止。

## 4. 连续目标与 atom mask

对 observed interior 功率使用与 D0-v 相同的 logit 表示：

\[
x_{dzh}=\operatorname{logit}\!\left(\operatorname{clip}(y_{dzh},10^{-4},1-10^{-4})\right).
\]

定义：

\[
M_{dzh}=I(\text{observed})I(\text{state=interior}).
\]

- 精确 0、精确 1 和缺失位置不进入条件均值损失；
- band projection 前，这些位置的 residual 必须严格为零；
- 0/1 值本身不能作为连续频谱的一部分；
- mask 的几何影响通过有效秩归一化和所有模型共享的 mask nuisance features 控制。

本次只读审计显示：observed cells 中约 91.8% 属于 continuous interior；最稀疏的 train day 仍有 123/240 个 active cells。该数字只用于确认协议可执行，不参与阈值选择。

## 5. 六个固定时空成分

不从 target 学 PCA，也不估计 240 个独立 mode。

### 5.1 空间投影

跨区共同成分：

\[
P_{common}=\frac{\mathbf 1\mathbf 1^\top}{10},
\]

区域局部成分：

\[
P_{local}=I-P_{common}.
\]

二者固定、正交，秩分别为 1 和 9。

### 5.2 时间投影

使用 24 小时正交 DCT-II：

| 频带 | DCT index | 解释 |
|---|---:|---|
| low | `[0,4)` | 日水平及约 16 小时以上的平滑结构 |
| mid | `[4,12)` | 约 4–12 小时结构 |
| high | `[12,24)` | 约 4 小时以下的快速变化 |

与两个空间投影组合为：

```text
low_common   low_local
mid_common   mid_local
high_common  high_local
```

### 5.3 mask-normalized residual energy

对条件均值 residual `r_d`，每个 group 的日级观测为：

\[
e_{db}=
\frac{\left\|P_{s,b}(M_d\odot r_d)P_{t,b}\right\|_F^2}
{\operatorname{tr}(P_bM_d)}.
\]

分母是该 day、该 band 在 active mask 下的有效秩。任何 group 的有效秩低于 1.5 都 fail closed。冻结前的只读 mask 审计中，最小值约为 2.04，因此规则在当前 train 数据上可执行。

## 6. 两层嵌套交叉拟合

267 个日期排序后分成 6 个连续 outer folds。每个 outer-training 集再按日期分成 4 个连续 inner folds。

### 6.1 第一层：条件均值

每个 zone-hour 单独拟合带截距 Ridge：

\[
\hat\mu_{zh}(c)=\beta_{0,zh}+c_{zh}^{\top}\beta_{zh}.
\]

输入是该 cell 的十个 NWP 特征。特征标准化只使用当前 fit fold。所有 240 个 cell 共享一个由 inner-fold masked logit MSE 选择的 Ridge alpha。

- outer-training 的 residual 必须来自 inner out-of-fold prediction；
- outer-test residual 来自在完整 outer-training 上拟合的模型；
- 任一 cell 的可用 fit observations 少于 40，立即停止。

因此不确定性模型不会把 in-sample mean overfit 当成低方差。

### 6.2 第二层：条件不确定性

对每个 band 分别拟合三个模型：

| 模型 | 输入 | 作用 |
|---|---|---|
| S0-static | 截距 | 全局静态方差参考 |
| M0-mask | 截距 + 7 个 mask nuisance features | 排除 atom/missing pattern 的解释 |
| N1-NWP | M0 输入 + 34 个冻结 NWP summaries | 测量 NWP 的增量价值 |

34 个 NWP summaries 包含：

- `U10/V10/U100/V100/WS10/WS100/WE10/WE100` 的日均值与 population std，共 16 个；
- `U100/V100/WS100` 在六个固定时空 group 上的 RMS，共 18 个。

采用 log-variance GLM，直接最小化 Gaussian variance quasi-score：

\[
\sum_d \nu_{db}\left[\eta_{db}+e_{db}\exp(-\eta_{db})\right]
+\lambda\|\beta\|_2^2,
\qquad \hat v_{db}=\exp(\eta_{db}),
\]

其中 `ν_db` 为有效秩，截距不受 Ridge penalty。M0 与 N1 分别通过 inner folds 选择 alpha。

## 7. Shuffle-NWP 反事实

Shuffle 不重新训练模型，也不破坏条件均值：

1. 保持 outer-test target、mask、日期和已经拟合的 N1 完全不变；
2. 只在 outer-test days 内 derange 不确定性模型所读取的 34 个 NWP summaries；
3. mask nuisance features 保持正确；
4. 使用 32 个预注册 PCG64 seeds，任何 permutation 不允许 fixed point。

这回答的是：

> 同一个不确定性模型只有在 NWP 与当天目标正确对应时才有效吗？

它不测试“把去噪器的全部天气条件打乱会不会变差”。

## 8. 正式分数

每个 day、每个 band 的 score 为：

\[
S_{db}=\log \hat v_{db}+\frac{e_{db}}{\hat v_{db}}.
\]

越低越好。主要 day-level estimand 是六个 band 等权平均后的：

\[
\Delta_d=S_{M0,d}-S_{N1,d}.
\]

正值表示 NWP 提高了条件不确定性预测。不同 schedule 的 raw training loss 不属于 G0-A，也不得作为未来跨路径证据。

同时报告：

- `S0 − N1` 静态安全对照；
- 六个 band 各自的 `M0 − N1`；
- 六个 outer folds 的方向一致性；
- 相对于逐日 saturated score 的 deviance explained；
- predicted-variance quintile、log-energy calibration slope 和 Spearman association。

统计不确定性使用 calendar-month cluster bootstrap，10,000 次、seed 28200。月份而非单个 cell 是重采样单位。

## 9. Go/No-Go

以下条件必须全部满足：

1. `M0 − N1` 的 95% month-cluster bootstrap 下界严格大于 0；
2. N1 至少解释 M0 与 saturated score 之间 5% 的可约 deviance；
3. 六个 outer folds 至少 5 个方向为正；
4. 六个 mode groups 至少 4 个方向为正；
5. 得到支持的 groups 同时覆盖 common 与 local，并覆盖至少两个 temporal bands；
6. N1 的 macro score 优于 S0；
7. 正确 NWP 的改善超过 32 个 Shuffle 改善的第 95 百分位；
8. 全程有限值、有效秩和数据 provenance 检查全部通过。

全部满足时：

```text
G0_A_NWP_MODE_UNCERTAINTY_GO
```

否则：

```text
G0_A_NWP_MODE_UNCERTAINTY_NO_GO
```

No-Go 后不为本假说追加更多 NWP 特征、频带切法或非线性模型。Go 也只允许设计 G0-B，不能宣称新 diffusion 有效。

## 10. 与未来 G0-B 的边界

G0-B 尚未冻结。若 G0-A 通过，G0-B 至少需要五条主路径与一个反事实：

```text
IID / CW / Fixed-band / MuLAN-lite / Predictability-aligned
                                      + Shuffled-NWP schedule
```

届时必须另行定义逐 diffusion time 的 aggregate nominal log-SNR 匹配、共同 estimator、共同小型 denoiser、重建指标和 Go/No-Go。G0-A 配置中不包含任何 schedule，因此当前结果不能被误写为 diffusion 模型收益。
