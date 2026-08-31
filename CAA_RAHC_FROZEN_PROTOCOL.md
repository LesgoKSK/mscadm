# CAA-RAHC 冻结实验协议

冻结日期：2026-07-18  
正式配置：`repro_configs/caa_rahc_frozen.json`  
底模 manifest：`outputs/caa_rahc/frozen_manifest.json`

## 1. 研究问题

前一轮 RAHC 的 bounded-tail C6 虽把 90% coverage 从 83.46% 提高到 88.33%，但 CRPS 恶化 0.49%，W90 平均增加 0.157，Winkler-90 也恶化。线性尾部 C6 的 CRPS 略好，却进一步欠覆盖。因此下一步不再扩大层级网络，而检验：

> 原子感知、状态门控且受 CRPS 非劣约束的局部尾部校准，能否降低条件 coverage error，同时不显著损害概率质量与 sharpness？

方法暂定名为 **CRPS-Constrained Atom-Aware RAHC（CAA-RAHC）**。

## 2. 结论的证据等级

本实验是“冻结方法后的内部 nested outer-split confirmation”，不是外部确认。原因是本地 GEFCom2014 数据只有 2012–2013 共 731 天，这些日期全部曾进入旧研究的 train、validation 或 test。新协议仍有两点实质加强：

1. 每个新 outer test 都从对应底模训练中完全剔除，并重新训练底模；
2. outer-test raw scenarios 在 `selection.lock.json` 生成前由程序硬阻断。

真正的外部确认仍需新的年份、另一地点或另一公开数据集。

## 3. 三个固定 outer splits

沿用旧 seed-0 日期划分的角色：

- 旧 validation 50 天：仅用于 CR-MS-CADM statistics head 选择；
- 旧 test 50 天：仅用于 CAA calibration、候选选择与门控拟合；
- 新 outer test：从旧 train 中预先固定选择 50 天；
- 当前 base train：旧 train 去掉当前 outer test，共 581 天。

每个 outer 因而包含 581/50/50/50 天，对应 5,810/500/500/500 个 zone-day。三个 test block 各 50 天且两两不重叠；当前 outer 的训练集可包含另外两个 outer 的 test 日期，但绝不包含自己的 test。

固定 test 日期 SHA-256：

- Outer 1：`adc254f8283b0b3f4699c51d75f59b55b4695e9e2b7465626c1edd5cd1729431`
- Outer 2：`0c5dbec382d12d931ddd577a216b9ade7bd1fe9c7371e8ec29ac236f21723225`
- Outer 3：`78c371c86c51337993f861faedd3f0fcc5494e196e93b3973460c8b915bd0923`

所有 NWP、flat-condition 和 target 标准化器只在当前 581 天训练集上拟合。旧权重、optimizer、标准化器和 raw scenarios 均禁止复用。

## 4. 底模

每个 outer 重训三个 CR-MS-CADM model seeds，共九个模型。网络、训练和采样参数保持上一轮完整复现：

- statistics head：4,000 steps；
- diffusion：18,000 steps；
- 100 ensemble members；
- 50-step DDIM，eta=1；
- `resume=false`。

Checkpoint 必须同时匹配配置、完整 split protocol、训练步数、文件 SHA 和 audit sidecar。

## 5. A0 基线

A0 是 hour-wise empirical PIT calibration，使用 finite-ensemble 线性尾部。Calibration 选择阶段使用五折日期 OOF：同一日十个 zone 始终属于同一折。每个 model seed 单独拟合 A0。

正式 test 变换时，A0 只在完整 50 天 calibration split 上拟合，再应用到 sealed outer test。

## 6. 原子 hurdle

边界分布写成非对称 hurdle：

\[
P(Y=0\mid x)=\pi_0(x),\qquad
P(Y=1\mid x)=(1-\pi_0(x))\rho_1,
\]

\[
P(0<Y<1\mid x)=(1-\pi_0(x))(1-\rho_1).
\]

`pi0` 由低容量 L2 logistic 在 base train 上拟合。输入仅含预测时可用的标准化 NWP、NWP 二次项、zone one-hot 与周期 hour。它不读取 calibration 或 test 的目标。

上边界事件在 151,440 个旧训练 cell 中仅 11 次，validation 甚至为零，因此不拟合自由的上原子网络。`rho1` 固定为 base-train-only Jeffreys estimate：

\[
\rho_1=\frac{n_1+0.5}{n_{Y>0}+1}.
\]

结构模型与 A0 场景中的零质量在 logit 空间按 `atom_strength` 混合。最终同时报告：

- 解析 `pi0/pi1` 的 Brier 与 log loss；
- 100 成员场景的实际 0/1 频率；
- 以 0.01 为概率分辨率的量化误差。

## 7. 状态门控的连续局部尾部

对 hurdle 的连续内部曲线记

\[
z_0(q)=\operatorname{logit} Q_0(q),\quad 0<q<1.
\]

以 `a=0.10` 为冻结锚点，左右门控为 `lambda_L(x), lambda_U(x)`：

\[
z^*(q)=z_0(q)-\lambda_L(x)[z_0(a)-z_0(q)]_+
+\lambda_U(x)[z_0(q)-z_0(1-a)]_+.
\]

因此连续条件概率 `q in [0.1,0.9]` 严格不变，只调整两端各 10%。Logit shift 再截断到 `[-dmax,dmax]`。门控使用 forecast-only 特征：raw mean、log spread、mean ramp、边界距离、A0 W90、平滑 raw `p0/p1` logit，加 centered hour、zone 和 zone×spread effects。

门控不学习“是否漏覆盖”标签，而直接最小化左右尾部的 proper pinball loss。每个 gate 外折又在其余日期内部重新做 A0 inner crossfit；因此 gate-held 日期既不进入 gate 参数，也不通过上游 A0 间接进入。

逐 cell width cap 只限制 tail 相对 atom-only 的额外扩宽：

\[
W90_{\text{final}}\le W90_{\text{atom-only}}+\delta.
\]

这是 calibration 阶段、outer test 尚未生成时冻结的修订。原因是 100 成员下 `pi0` 跨过 5% 分位会产生离散的 W90 跳变，若把 atom 也纳入逐 cell baseline cap，所有完整候选都会因少数 cell 不可行。Atom 对 A0 的总体 width 影响仍受后述点估计与 bootstrap 上界硬约束。

## 8. 候选与消融

- A0：C0 linear empirical baseline；
- A1：此前 full RAHC，改用 linear tail；
- A2：atom-only；
- A3：regularized gate-only；
- A4：atom + regularized local gate，主方法；
- A5：与 A4 相同候选，但选择时忽略非劣约束；
- A6：atom + no-shrink gate。

冻结网格：

- atom strength：0.5, 1.0；
- tail strength：0.25, 0.5, 0.75, 1.0；
- dmax：0.25, 0.5；
- per-cell tail W90 cap：0.02, 0.05；
- regularized gate penalty：0.03；no-shrink：0。

候选场景流式生成、立即转成日期充分统计量，不长期保存完整网格。

## 9. 约束选择

A0 永远是可行候选。对每个 calibration 日期使用同一 paired multinomial 权重，对三个 outers × 三个 model seeds 作为九个模型重复进行平均。A4 候选必须同时满足：

1. CRPS 相对恶化的 paired date-bootstrap U95 不超过 0.5%；
2. 九个模型重复中每一个 CRPS 点恶化不超过 0.5%；
3. MAE、VS、ramp-CRPS、Winkler-90 相对恶化 U95 均不超过 1%；
4. W90 相对 A0 的绝对增加，点估计和 U95 均不超过 0.05。

只在可行集中最小化 family-equal conditional ACE90；再依次以更低 Winkler、更小 gate、更小 cap、更强 shrinkage 和名称排序打破平局。若没有 A4 候选可行，精确回退 A0。A5 只在独立的 unconstrained 路径中选择，不能混入 A4。

## 10. Sealed test 与统计

一次性完成全部九套 calibration OOF、选择一个共享配置并写入 lock 后，才允许生成三个 outer 的 test raw scenarios。最终配置在 outer/seed 间完全相同；参数模型可按 outer 在其 50 天 calibration 上重新拟合。

最终报告：

- 每 outer、每 seed 与 pooled 150-date 指标；
- CRPS、MAE、VS、ramp-CRPS、coverage、W90、Winkler；
- zone/hour/wind/spread/zone×spread 条件指标；
- 日期簇 paired bootstrap；
- 解析 atom score 与 finite-M 量化；
- 严格秩反转、ties collapsed、central changed、数值失败和 width-cap 激活；
- SUC 描述性敏感性。

主要成功门槛为 coverage90 处于 [0.88,0.92]、conditional ACE90 至少下降 30%、CRPS 非劣 0.5%、次指标非劣 1%、W90 增加不超过 0.05、strict reversals=0，并检查 outer/seed 一致性。

## 11. 权威实现与原型

权威选择链为：

- `caa_rahc/nested_data.py`
- `caa_rahc/structural_atom.py`
- `caa_rahc/gate.py`
- `caa_rahc/hinge_tail.py`
- `caa_rahc/candidates_nested.py`
- `caa_rahc/selection.py`

早期 `caa_rahc/zero_atom.py`、`calibration.py`、`model.py`、`splits.py`、`outer_data.py` 和单层 `candidates.py` 仅为开发原型，正式 artifact metadata 不得引用它们为权威实现。
