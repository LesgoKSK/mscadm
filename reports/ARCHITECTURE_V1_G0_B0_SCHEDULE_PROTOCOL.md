# Architecture-v1 G0-B0：信息保留式 SNR 分配可行性协议

冻结日期：2026-09-14

协议状态：在已披露的解析与手工 schedule 探索之后、可执行 G0-B0 runner 之前冻结。

机器配置：[`architecture_v1_g0_b0_schedule.json`](../repro_configs/architecture_v1_g0_b0_schedule.json)

配置 SHA256：`1e67af3389c55aa6e7127f4ea8f0a0dc6fc7b2fc4d1c38c3b066320f597a6ab8`

预执行说明：第一次 dry-run 后、正式 `--execute` 前，测试发现配置中的 noisy endpoint 十进制文本与已冻结的 float32 schedule 字节相差约 `8.89e-18`。本次仅把该显示值改为字节解码所得的精确 float32 数值；schedule 字节及其 SHA256、公式、阈值和主分析参数均未改变。

## 1. G0-B0 回答什么

G0-A 已经支持：

```text
NWP → 六类 mode-group conditional uncertainty
```

G0-B0 只把这项证据翻译成一个确定、受约束、数值可执行的 SNR allocation。它回答：

> 在不训练 denoiser、不读取任何原始 target 的前提下，能否让 NWP 条件方差改变六类 mode 的 corruption path，同时逐日逐步严格保持 IID Gaussian mutual-information budget，并避免裸 reverse water-filling 的硬 cutoff？

G0-B0 不回答生成质量、denoising utility、sample efficiency 或论文方法是否成立。

## 2. 探索披露

冻结前已经查看过：

- 267 天 G0-A out-of-fold `N1_variance`；
- G0-A 的逐日 `effective_rank`；
- 当前 D0-v cosine-250 schedule；
- 裸 reverse water-filling 的硬 cutoff；
- `eta=0.5` 的初步预算、单调性、正值和 shift 范围。

因此本协议属于 **工程可行性冻结**，不是独立统计预注册。后续 G0-B denoiser utility 必须另建协议，且不能用 G0-B0 的 Pass 代替模型证据。

## 3. 唯一允许的输入

执行路径只允许读取：

1. 本配置及 SHA256 sidecar；
2. 正式 `G0_A_RESULT.json` 及 sidecar；
3. 已登记的 `MaskedJointDDPM` cosine schedule；
4. 纯 NumPy G0-B0 allocation module。

正式 G0-A result SHA256 必须为：

```text
37a55921016bd701ffb2f5a66f56b6abbaa28d027cfdb9b46e0b3f01f8d9b7e7
```

runner 不得导入原始数据 loader，不得 materialize train、validation、calibration、selection、R-SEEN 或 final target array，也不得加载 G0-A 拟合权重。

runner 会解析完整的冻结结果文件，但从 G0-A 每日记录中仅消费：

- 日期与 outer-fold 标签，用于可追溯性；
- 六个 `N1_variance`；
- 六个 `effective_rank`。

## 4. 六个固定 mode groups

顺序保持为：

```text
low_common, low_local,
mid_common, mid_local,
high_common, high_local
```

名义秩为：

```text
4, 36, 8, 72, 12, 108
```

逐日 allocation 权重为：

\[
w_{d,k}=r_{d,k}^{\mathrm{eff}}\Big/\sum_jr_{d,j}^{\mathrm{eff}}.
\]

这对应原 240 维空间中的 dimension-weighted information 与 absolute posterior MSE。G0-A 用于报告的六组等权 macro score 不用于本次信息预算。

## 5. Gaussian reference 与公平预算

对某天、某 mode：

\[
z_k\mid c\sim\mathcal N(0,v_k),
\qquad
y_{t,k}=\alpha_{t,k}z_k+\sigma_{t,k}\epsilon,
\qquad
\rho_{t,k}=\alpha_{t,k}^2/\sigma_{t,k}^2.
\]

Gaussian mutual information 和 posterior variance 为：

\[
I_{t,k}=\frac12\log(1+v_k\rho_{t,k}),
\qquad
D_{t,k}=\frac{v_k}{1+v_k\rho_{t,k}}
=v_k\exp(-2I_{t,k}).
\]

对原 IID cosine schedule `rho_0(t)`，逐日逐步预算定义为：

\[
B_d(t)=\sum_kw_{d,k}\frac12\log(1+v_{d,k}\rho_0(t)).
\]

所有候选 allocation 必须保持同一个 `B_d(t)`。这不是平均 nominal log-SNR 合同；两者不得混写。

## 6. 裸 reverse water-filling 只作为 oracle

在约束 `sum w_k I_k=B` 下最小化 Gaussian posterior MSE 的解为：

\[
I_k^*=\frac12\left[\log\frac{v_k}{\theta}\right]_+.
\]

它会让 `v_k <= theta` 的 mode 获得零信息。真实 G0-A 方差上的先验探索已经确认，cosine schedule 的高噪声部分会大量关闭 mid/high mode。因此 `eta=0` 只保留为理论 sensitivity/oracle，不作为 G0-B0 主候选。

## 7. 主候选：reserve-then-water-fill

先保留 IID 信息：

\[
I_{k,\mathrm{res}}=\eta I_{k,\mathrm{IID}}.
\]

令保留信息之后的有效方差为：

\[
a_k=v_k\exp(-2I_{k,\mathrm{res}}).
\]

只对剩余预算求解：

\[
\min_{J_k\ge0}\sum_kw_ka_k\exp(-2J_k),
\qquad
\sum_kw_kJ_k=(1-\eta)B.
\]

解析 active-set 解为：

\[
J_k^*=\frac12\left[\log\frac{a_k}{\theta}\right]_+,
\qquad
I_k^*=I_{k,\mathrm{res}}+J_k^*.
\]

再转换回：

\[
\rho_k^*=\frac{\exp(2I_k^*)-1}{v_k},
\qquad
\bar\alpha_k^*=\frac{\rho_k^*}{1+\rho_k^*}.
\]

主值固定为：

```text
eta = 0.5
```

解释是：每个 mode 的一半 IID information 不可被拿走，只允许重分配另一半。该值不能根据未来 reconstruction 或 generation 结果修改。

边界关系必须成立：

- `eta=1`：精确 IID；
- `eta=0`：裸 reverse water-filling；
- 六个 `v_k` 相等：任意 `eta` 均精确 IID。

`eta=0/0.25/0.5/0.75/1` 全部报告，但仅用于解释强度和退化边界，不构成 outcome-based 选择集合。

## 8. baseline schedule 身份

必须复用当前 D0-v：

- cosine timesteps：250；
- offset：0.008；
- beta clip：`[1e-8, 0.999]`；
- buffer dtype：float32；
- `alpha_bar` 原始 little-endian float32 bytes SHA256：
  `ab4df3b8c173b936e8b3a6fb4ecddbcd96d5652fbd1a893c4ab8871b4d6a81dc`。

任何 schedule identity 漂移均为 fail-closed。

## 9. 全部 Go 门

主候选必须同时满足：

1. G0-A artifact、状态、配置和 train-only data identity 全部匹配；
2. G0-B0 未导入 loader、未 materialize raw target、未加载权重；
3. baseline `alpha_bar` SHA256 匹配；
4. 每个 `day × step` 的 information-budget 最大绝对误差不超过 `1e-12`；
5. 398,898 个逐日逐 mode 相邻步检查中，SNR 向 forward noisy endpoint 增大的违规数为 0，容差 `1e-10`；
6. 主候选所有 SNR 有限且严格为正；
7. 每个 mode 的 information-retention ratio 不小于 0.5；
8. 相对 IID 的 log-SNR shift 完全位于 `[-1.0, 2.5]`；
9. clean endpoint 最小 `alpha_bar >= 0.999`；
10. noisy endpoint 最大 `alpha_bar <= 1e-6`；
11. `eta=1` 的 allocated VP `alpha_bar` 与 IID identity 最大误差不超过 `1e-12`；
12. 合成 equal-variance 的 allocated VP `alpha_bar` identity 最大误差不超过 `1e-12`。

全部通过才记为：

```text
G0_B0_SCHEDULE_FEASIBILITY_GO
```

否则记为：

```text
G0_B0_SCHEDULE_FEASIBILITY_NO_GO
```

## 10. Pass 后仍不能说什么

G0-B0 Go 只能授权单独冻结 G0-B tiny-denoiser protocol。它不能支持：

- reserve-then-water-fill 优于 IID、CW、Fixed-band 或 MuLAN-lite；
- denoiser 更容易训练；
- ramp、joint dependence 或场景分数改善；
- Diffusion 已经成为最终模型；
- Gaussian water-filling 是新的数学结果。

G0-B 正式协议还必须解决 oracle-adjusted reconstruction、learning curve、六条 path、公平训练预算和 Shuffled-NWP 反事实，之后才允许任何 retained denoiser run。
