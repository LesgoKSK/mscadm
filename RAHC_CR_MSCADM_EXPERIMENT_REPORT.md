# RAHC-CR-MS-CADM 完整实验报告

## 1. 结论先行

本实验实现并评估了 **Regime-Adaptive Hierarchical Calibration（RAHC）**：在 CR-MS-CADM 的 100 成员原始场景上，用状态自适应、层级收缩的 Beta–Binomial 秩模型做后处理，并保持成员标签下的单调边际变换。

结论是：**研究假设只得到部分支持，当前 C6-RAHC 不应替换现有经验 PIT 校准。**

- 支持假设的部分：C6 将平均 90% 覆盖率从 **83.46% 提高到 88.33%**，family-equal 条件 ACE 从 **0.06910 降到 0.02921（下降 57.73%）**。按 50 个日历日期聚类的 10,000 次配对 bootstrap 中，条件 ACE 改善的支持率为 **100%**。
- 否定当前实现的部分：CRPS 从 **0.084655 恶化到 0.085068（恶化 0.49%）**；95% 日期簇 bootstrap 区间对应的 CRPS 改善为 **[-0.000979, 0.000164]**，严格改善支持率仅 **8.15%**。90% 区间平均加宽 **0.15727**，Winkler-90 反而恶化 **0.07698**，两者都说明可靠性提升主要来自过度扩宽，而不是更好的概率分布。
- 预注册的 11 项成功门槛仅通过 4 项：总体覆盖、条件 ACE 降幅、三种子均值下的最差 Zone×spread 覆盖、零严格秩反转。CRPS、MAE、VS、ramp-CRPS、三种子一致性、最差 Zone 和 exact stable-rank 均未通过。
- 线性尾部敏感性 C6-linear 的 CRPS 为 **0.084353**，较 legacy G0 好约 **0.36%**，但 95% 区间仍跨零，且覆盖率下降至 **82.75%**。这证明问题的核心不是“是否学到状态”，而是**如何把状态相关秩校准转化为既可靠又尖锐的有界尾部**。

因此，本轮最有价值的产出不是一个可直接采用的新模型，而是定位了下一步：**用 CRPS/MAE 非劣约束下的局部尾部质量模型，替代无约束的有界扩宽。**

## 2. 实验性质与有效性边界

这是一项完整的、可复现的**探索性 reused-test 扩展实验**，不是确认性实验。

RAHC 的想法受到此前同一 test 划分上 Zone 4/5、低 spread 状态欠覆盖诊断的启发，因此即使本轮所有超参数都只由 validation 选择，test 也已经间接参与过研究假设的形成。严格确认需要新 outer split，并在每个 split 上重新训练底层 CR-MS-CADM；不能只重新切分校准数据来冒充独立验证。

另外，三个模型 seed 共享同一组观测，它们是模型随机性的稳健性复现，不是三个独立数据集。统计推断始终以 50 个唯一日历日期为簇，没有把 500 个 zone-day 或三个 seed 当成独立样本。

## 3. 方法

### 3.1 潜在秩模型

对 case-hour 条件状态 (x_{ih})，定义潜在 PIT：

\[
P_{ih}\mid x_{ih}\sim\operatorname{Beta}(\alpha_{ih},\beta_{ih}),\qquad
R_{ih}\mid P_{ih}\sim\operatorname{Binomial}(M,P_{ih}).
\]

积分掉 (P_{ih}) 后，有限集合秩 (R_{ih}) 服从 Beta–Binomial。恒等校准对应 (alpha=eta=1)，此时 (R\in\{0,\ldots,M\}) 为离散均匀分布。

### 3.2 平局不是 mid-rank：使用区间删失似然

由于功率被裁剪到 ([0,1])，validation 中跨三种子有 **8.20%** 的 case-hour 出现场景成员与观测精确相等；test 中也有 **4.44%**。直接使用 `<`、`<=` 或确定性 mid-rank 会将大量边界平局系统性推向某一秩。

正式实现保存：

\[
R_L=\#\{\hat y_m<y\},\qquad
R_U=\#\{\hat y_m\le y\},
\]

并最大化区间事件的精确似然：

\[
\Pr(R_L\le R\le R_U\mid x)
=\sum_{r=R_L}^{R_U}\Pr_{\mathrm{BB}}(R=r\mid\alpha(x),\beta(x)).
\]

这样平局单元提供较弱的区间信息，而不会被强行指定到一个秩。

### 3.3 可识别的层级参数化

使用：

\[
m(x)=\sigma(\eta_{\text{bias}}(x)),\qquad
\kappa(x)=2\exp(\eta_{\text{disp}}(x)),
\]

\[
\alpha(x)=m(x)\kappa(x),\qquad
\beta(x)=(1-m(x))\kappa(x).
\]

所有参数为零时 (m=0.5,\kappa=2)，因此严格得到 (alpha=eta=1)。hour、zone 与 zone-specific spread slope 均在前向计算时中心化，避免与全局截距不可识别；hour 另加周期相邻差分惩罚。

只使用四个推断时可得、来自 **raw forecast** 的低维特征：

1. ensemble mean；
2. log ensemble standard deviation；
3. mean forecast 的绝对 ramp；
4. ensemble mean 到物理边界 0/1 的距离。

完整 C6 结构为：global + hour + zone + linear regime + 强收缩的 zone-specific log-spread slope。没有使用观测、误差、PIT 或校准后区间作为测试输入特征。

### 3.4 校准变换与尾部

拟合后定义条件 CDF：

\[
G(y\mid x)=H_x(F_{\text{raw}}(y\mid x)),
\]

其中 (H_x) 是 (operatorname{Beta}(\alpha(x),\beta(x))) 的 CDF。生成第 (q) 个校准成员时使用：

\[
F_{\text{raw}}^{-1}\!\left(H_x^{-1}(q)\right).
\]

主实验的 bounded tail 在经验分位点外增加物理伪节点 ((0,0)) 和 ((1,1))。另固定比较：

- `linear`：由两端次序统计量线性外推后裁剪到 ([0,1])；
- `clamp`：直接截到 raw min/max。

所有变换都按原成员的稳定秩放回标签，没有严格逆序；但边界裁剪和相同值映射会拆分或新建 ties，因此不能声称 exact empirical Copula 不变。

## 4. 数据、交叉验证与预注册分组

- 输入：`full_seed{0,1,2}_{validation,test}_raw.npz`，每个为 `[500,100,24]`。
- validation/test 各 50 个唯一日历日期 × 10 个 Zone；日期集合无交集。
- 三种子的 observation/zone/day 完全对齐。
- 5-fold 由真实 `day` 显式分组；每折 10 日、100 zone-day、2400 小时单元。同日全部 10 Zone 和三个 seed 始终同折。
- 每折在三个 seed 的训练日期 raw forecasts 上共同拟合一个校准器，三个 seed 的 held-out 结果分别计分；最终超参数是三种子 OOF 目标的共同选择。
- 条件分组阈值只由 validation raw forecast 冻结。mean 与 spread 都在每个 Zone 内独立取五分位，测试不得重算。
- 预注册 5 个 family、94 个组：10 Zone、24 hour、5 within-zone wind quintile、5 within-zone spread quintile、50 Zone×spread。
- 选择目标冻结为：

\[
\overline{\mathrm{CRPS}}
+0.10\,\overline{\mathrm{ACE}_{50,80,90}}
+0.05\,\overline{\mathrm{conditional\ ACE}_{90}}.
\]

这里后来暴露出一个设计缺陷：加权和允许 CRPS 变差来换覆盖率，而成功门槛又要求 CRPS 至少改善 2%。下一版应改成“CRPS 非劣约束 + 在可行域内最小化条件 ACE”，而非继续调权重。

## 5. 消融与 validation 选择

| 方法 | 结构 | 正则 | strength | OOF CRPS | OOF global ACE | OOF conditional ACE | 目标 |
|---|---|---:|---:|---:|---:|---:|---:|
| C3 | hour + zone | 0.03 | 1.0 | 0.075295 | 0.015407 | 0.021058 | **0.077888** |
| C2 | hour | 0.03 | 1.0 | 0.075080 | 0.018685 | 0.024526 | 0.078175 |
| C4 | hour + regime | 0.03 | 1.0 | 0.075962 | 0.013620 | 0.025663 | 0.078607 |
| C5 | hour + zone + regime | 0.03 | 1.0 | 0.076339 | 0.012194 | 0.023274 | 0.078722 |
| C6 | C5 + zone×spread slope | **0.10** | 1.0 | 0.075711 | 0.017454 | 0.030795 | 0.078997 |
| C1 | global BB | 0.03 | 1.0 | 0.074992 | 0.023398 | 0.034766 | 0.079070 |
| C0 | hour empirical PIT | 0 | 1.0 | **0.074063** | 0.031250 | 0.065795 | 0.080478 |
| C7 | C6 no shrink | 0 | 1.0 | 0.077536 | 0.018278 | 0.023470 | 0.080537 |

验证集已经显示：

- C0 的 CRPS 最好，但条件可靠性差；
- C3 的预注册综合目标最好；
- C6 选择候选中最强的正则 0.1，说明 Zone×spread 斜率在仅 50 个日期下必须强收缩；
- C7 去收缩明显恶化 CRPS，层级收缩是必要的；
- C6 的 OOF 90% coverage 为 88.16%，C3 为 91.29%，C5 为 90.40%。

## 6. test 总体结果（三种子均值）

| 方法 | CRPS ↓ | 90% coverage | width-90 ↓ | Winkler-90 ↓ | MAE ↓ | VS ↓ | ramp-CRPS ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw CR | 0.088170 | 65.59% | 0.288696 | 0.858670 | 0.117379 | 17.8066 | 0.052816 |
| legacy G0 | 0.084655 | 83.46% | 0.437231 | **0.669842** | 0.118580 | 16.8398 | 0.051760 |
| C0 bounded | **0.084613** | 83.47% | 0.437261 | 0.669804 | 0.118605 | 16.8341 | 0.051749 |
| C1 global | 0.084968 | 94.18% | 0.655946 | 0.707289 | 0.124480 | 17.1625 | 0.052388 |
| C2 hour | 0.085217 | 93.11% | 0.650802 | 0.722880 | 0.124960 | 17.1915 | 0.052559 |
| C3 hour+zone | 0.085492 | 92.28% | 0.659431 | 0.746096 | 0.125879 | 17.2393 | 0.052687 |
| C4 hour+regime | 0.085165 | 90.97% | 0.673485 | 0.780419 | 0.128681 | 17.5188 | 0.053122 |
| C5 additive | 0.085417 | 90.80% | 0.670372 | 0.778929 | 0.129399 | 17.5419 | 0.053180 |
| **C6 RAHC** | 0.085068 | **88.33%** | 0.594501 | 0.746822 | 0.127184 | 17.2565 | 0.052606 |
| C7 no shrink | 0.086254 | 91.63% | 0.704603 | 0.797003 | 0.131869 | 17.8921 | 0.053779 |

C6 在三个 seed 的 CRPS 均比 legacy G0 差，因此不存在“均值被单个 seed 拖累”的解释。C3 虽为 validation 目标最优，test 上却过覆盖到 92.28%，CRPS 也显著恶化；日期簇 bootstrap 的 CRPS 改善区间为 `[-0.001301,-0.000356]`。

## 7. 条件可靠性

三种子均值、阈值由 validation 冻结：

| family | legacy G0 mean ACE | C6 mean ACE | legacy 最低覆盖 | C6 最低覆盖 |
|---|---:|---:|---:|---:|
| Zone | 0.06567 | 0.02386 | Zone 4: 77.22% | Zone 3: 84.50% |
| Hour | 0.06544 | 0.02214 | Hour 03: 80.40% | Hour 09: 85.73% |
| spread quintile | 0.07068 | 0.02912 | Q1: 80.16% | Q5: 85.06% |
| Zone×spread | 0.07405 | 0.04057 | Z4×Q1: 69.97% | Z10×Q2: 80.28% |

这是 C6 最明确的正面结果：它确实修复了最初关注的条件欠覆盖，而且最差 Zone×spread 的三种子均值超过预注册的 80% 门槛。

但是 seed 分开看仍不够稳健：C6 的最差 Zone coverage 分别为 86.33%、82.17%、82.42%；最差 Zone×spread 分别为 83.12%、72.30%、80.35%。所以“三种子均值过线”不能替代逐 seed 稳健性。

## 8. 日期簇配对 bootstrap

主比较为 legacy G0 vs bounded-tail C6。每次重采样一个日历日期时同时带入该日全部 10 Zone，并对三个 seed 使用同一日期权重；先求 seed 内配对差，再对三 seed 取均值。

| 指标（legacy − C6，正为好） | 点估计 | 95% CI | 改善支持率 |
|---|---:|---:|---:|
| CRPS | -0.000413 | [-0.000979, 0.000164] | 8.15% |
| Winkler-90 | -0.076980 | [-0.105003, -0.048564] | 0% |
| 90% coverage 绝对误差改善 | 0.048694 | [0.036972, 0.057333] | 100% |
| conditional family-equal ACE 改善 | 0.039892 | [0.027126, 0.048599] | 100% |
| worst undercoverage 改善 | 0.089756 | [0.043104, 0.178819] | 99.99% |

区间宽度使用 `C6 − legacy`：点估计 **+0.157270**，95% CI `[0.132139,0.183228]`，支持“不增宽”的比例为 0%。

这些是 bootstrap sign-support fraction，不是后验概率，也不是 p 值。

## 9. 尾部敏感性与失败机制

| C6 尾部 | CRPS | 90% coverage | width-90 | Winkler-90 | temporal rank RMSE |
|---|---:|---:|---:|---:|---:|
| bounded（主） | 0.085068 | 88.33% | 0.594501 | 0.746822 | 0.101387 |
| linear | **0.084353** | 82.75% | 0.420989 | **0.668679** | 0.036680 |
| clamp | 0.084430 | 82.20% | 0.416248 | 0.672023 | 0.036095 |

对 legacy G0，C6-linear 的 CRPS 改善点估计为 0.000301，95% CI `[-0.000115,0.000734]`，改善支持率 92.17%，但覆盖误差显著恶化。主 bounded 规则反过来修复覆盖，却破坏 sharpness。

参数审计说明模型本身学到了预期方向：

- 全局 (eta_{disp}=-0.776)，对应小于 2 的 concentration，秩分布为 U 型，意味着原集合欠离散；
- log-spread 对 (eta_{disp}) 的系数为 **+0.406**，所以低 spread 状态会得到更小 concentration 和更强尾部扩宽；
- validation/test 的 (alpha) 中位数约为 0.501/0.516，(eta) 中位数约为 0.477/0.485；
- test 生成概率中 3.17% 低于 0.001，2.88% 高于 0.999，尾部变换频繁进入极端区。

因此失败机制是：**条件秩模型正确识别了“低 spread 需要更宽”，但 bounded pseudo-knot 把概率尾部直接拉向 0/1，造成大范围、非局部的物理区间扩宽。**

## 10. 排序与 Copula 审计

C6 三种子均为零严格秩反转，但这不等于 exact Copula 保持：

- 平均拆分 raw tied pairs：199,294；
- 平均将 raw strict pairs 折叠为 ties：453,169；
- 固定 stable tie-break 的 ordinal-rank 一致率：95.83%，未达到 100%；
- 边界成员比例由 raw 4.35% 增至 6.31%。

因此唯一可成立的表述是：变换没有引入严格逆序，成员标签下的弱时间秩模板大体保留；不能声称 exact temporal empirical Copula 不变，更不能声称空间 Copula，因为各 Zone 原本就不是联合采样。

总体 temporal rank RMSE 从 legacy 的 0.04258 恶化到 bounded C6 的 0.10139。该指标跨 case 混合所有成员轨迹，状态相关变换和新增 ties 都会改变它；这也是主 bounded 实现不宜采用的额外证据。

## 11. 边界事件

C6 的 (Y=0) Brier 从 0.044706 改善到 0.043385，(Y=1) Brier 从 0.002263 改善到 0.001361。说明状态模型对边界质量并非完全无效，但该收益不足以抵消整体 MAE、CRPS 与 multivariate dependence 的代价。

## 12. SUC 描述性敏感性

复用既有 7 个固定 zone-date case；它们只覆盖 5 个唯一日期，并把单区归一化曲线当作 1200 MW 风电代理，不能视为全系统或独立 7 日统计结果。

| 指标（7 case 均值） | legacy G0 | C6 RAHC | 相对变化 |
|---|---:|---:|---:|
| planned total cost | 335,922.50 | 337,933.58 | +0.60% |
| realized total cost | 346,743.21 | 343,396.82 | -0.97% |
| realized load shedding | 4.5539 | 0.2486 | -94.54% |
| realized wind curtailment | 40.5054 | 38.1390 | -5.84% |

所有 C6 规划求解为 optimal；legacy 的 index 420 规划达到 120 秒 time limit，因此运营均值比较还混入求解状态差异。该结果只能说“值得进一步扩大日期样本”，不能作为采用 C6 的依据。

## 13. 预注册门槛审计

| 门槛 | 结果 | 是否通过 |
|---|---:|:---:|
| CRPS 至少改善 2%，且 95% CI 排除 0 | -0.49%，CI 跨 0 | 否 |
| 90% coverage 在 [88%,92%] | 88.33% | 是 |
| conditional ACE 至少下降 30% | 57.73% | 是 |
| 最差 mean-seed Zone coverage ≥85% | 84.50% | 否 |
| 最差 mean-seed Zone×spread ≥80% | 80.28% | 是 |
| MAE 恶化 ≤1% | +7.26% | 否 |
| VS 恶化 ≤1% | +2.47% | 否 |
| ramp-CRPS 恶化 ≤1% | +1.63% | 否 |
| 三个 seed 的 CRPS 均改善 | 0/3 | 否 |
| strict reversal = 0 | 0 | 是 |
| stable ordinal ranks = 100% | 95.83% | 否 |

整体判定：**4/11，通过不足，C6-RAHC 主方案失败。**

## 14. 下一版最具体的改进方向

下一步不应继续给层级模型加更多特征，而应解决“可靠性—尖锐度”约束：

1. 将选择规则改成约束优化：先要求 OOF `CRPS <= C0 × 1.005`、`MAE/VS/ramp-CRPS` 恶化均不超过 1%，再在可行配置中最小化 conditional ACE；没有可行解就保持 C0。
2. 将 calibration 拆成内部形状与边界质量两部分：内部用 Beta concentration 校准，0/1 atom 用单独的 hurdle/Brier 模型；不要用一个 pseudo-knot 同时承担两件事。
3. 尾部只对预注册低-spread 且验证支持充分的状态开放，并对扩张距离加显式惩罚；其他状态保留 legacy linear empirical map。
4. 采用保持 strict order 的 nextafter/块级映射，或直接报告随机化 Copula；不再把“zero inversion”写成“exact Copula preserved”。
5. 在新 outer splits 上重训底模后再做确认；同时把通用校准器应用到强 DDPM，区分“后处理收益”与“CR 架构收益”。

一个可操作的暂定名称是 **CRPS-Constrained Atom-Aware RAHC（CAA-RAHC）**。本轮证据表明，它应优先解决尾部原子质量与非劣约束，而不是继续扩大层级网络。

## 15. 复现

```powershell
# 冻结 validation 协议与日期 folds
python -m repro_scripts.run_rahc_experiment prepare

# validation-only C0–C7 交叉拟合与共同超参数选择
python -m repro_scripts.run_rahc_experiment cv

# 读取已冻结选择，一次性应用 test
python -m repro_scripts.run_rahc_experiment finalize

# 总体、条件、边界与排序指标
python -m repro_scripts.run_rahc_experiment evaluate

# 50-date clustered paired bootstrap
python -m repro_scripts.rahc_bootstrap

# 描述性 SUC
python -m repro_scripts.rahc_suc --clusters 10 --time-limit 120

# 参数与成功门槛审计
python -m repro_scripts.rahc_diagnostics

# 图表（无 GUI 环境）
$env:MPLBACKEND='Agg'
python -m repro_scripts.rahc_figures

# 全仓库测试
python -m pytest -q
```

本次测试结果：**34 passed**。

## 16. 权威产物

- 冻结协议：`repro_configs/rahc.json`
- 实现：`rahc/v2_model.py`、`rahc/v2_calibration.py`、`rahc/v2_features.py`
- 分组指标：`rahc/group_metrics.py`
- 日期簇 bootstrap：`rahc/bootstrap.py`
- fold 与输入审计：`outputs/rahc_cr_mscadm/protocol/`
- OOF 与选择表：`outputs/rahc_cr_mscadm/cv/`、`outputs/rahc_cr_mscadm/scenarios/validation_oof/`
- test 场景：`outputs/rahc_cr_mscadm/scenarios/test/`
- 总体/条件/排序指标：`outputs/rahc_cr_mscadm/metrics/`
- bootstrap：`outputs/rahc_cr_mscadm/statistics/paired_calendar_day_bootstrap.json`
- 参数与门槛：`outputs/rahc_cr_mscadm/diagnostics/`
- SUC：`outputs/rahc_cr_mscadm/suc/`
- 图表：`outputs/rahc_cr_mscadm/figures/`

早期草案文件 `rahc/model.py`、`rahc/calibration.py`、`rahc/metrics.py` 不属于权威实验实现；本报告与产物只以 `v2_*`、`group_metrics.py` 和冻结配置为准。
