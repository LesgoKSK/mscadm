# MM-JDWind：完整实现与实验结论

## 1. 研究问题

原始 MS-CADM/CR-MS-CADM 将归一化风功率视为连续变量，但 GEFCom2014 数据在
0（停机、切出、无风）处存在明显点质量。此前 CAA 实验中，结构零原子修正解释了
约 99.3% 的 CRPS 增益，因而提出：

> 将离散边界状态与连续内部幅值分别建模，是否能改善风电场景的边际校准、联合结构
> 和零事件持续性？

暂定方法名为 **MM-JDWind (Mixed-Measure Jump–Diffusion Wind Scenario
Generation)**。

## 2. 方法实现

每个样本是一个完整日的 10 区域 × 24 小时联合场，而不是相互独立的 zone-day。

模型包含三个可独立消融的部分：

1. **混合测度头**：分解为 `P(Y=0)`、`P(Y=1 | Y≠0)` 和内部连续分布的位置/尺度；
2. **离散跳跃生成器**：通过吸收 mask 的分类生成过程产生
   `{0, interior, 1}` 状态场；
3. **有界 rectified flow**：只为内部状态生成 logit 空间残差，再严格映射回
   `[0,1]`。

状态采样实现了四种候选：

- `none`：不使用离散跳跃；
- `independent`：逐位置独立跳跃；
- `correlated`：共享随机性的相关跳跃；
- `mass_preserving`：有限 ensemble 中保持解析原子质量，同时学习联合状态结构。

网络使用时间轴与区域轴的 axial attention。主要源码位于：

- `mm_jdwind/model.py`
- `mm_jdwind/sampling.py`
- `mm_jdwind/training_v2.py`
- `mm_jdwind/metrics.py`
- `mm_jdwind/confirmation_data.py`

模型参数量为 1,714,857，其中混合测度头 304,421、跳跃模块 491,139、连续流
919,297。重训的 DDPM comparator 为 389,121 参数。

## 3. 训练与选择协议

### 3.1 开发阶段

- 使用此前 CAA 的 3 个 nested outer splits；
- 每个 outer 训练 3 个 seeds，共 9 个模型；
- head、jump、flow 分阶段训练；
- 每 250 步验证，连续 8 次无改善则早停；
- loss、梯度或参数任一非有限即拒绝 checkpoint；
- 仅用 calibration 日期选择状态机制；
- 低成本选择采用 M=50、NFE=8。

开发集 9 次复制的均值：

| 状态机制 | CRPS | 90% coverage | Joint ES |
|---|---:|---:|---:|
| none | 0.082020 | 0.723991 | 1.727380 |
| independent | 0.081342 | 0.802722 | 1.720736 |
| correlated | 0.081074 | 0.756185 | 1.721444 |
| mass-preserving | **0.080381** | **0.811778** | **1.702919** |

因此在查看新 test 之前锁定 `flow_mass_preserving`。

### 3.2 proper-score 微调失败消融

预锁定 pilot 中，proper-score ensemble 微调产生 87 个非有限 flow 参数。该分支被
明确拒绝，没有用于候选锁定或确认实验。恢复 checkpoint 与失败记录保存在：

- `outputs/mm_jdwind_development/outer1/runs/seed0/proper_failure.json`

所有 v2 与确认模型均使用 validation-selected flow checkpoint。

### 3.3 新冻结确认阶段

- 在任何 MM-JDWind test 场景生成前冻结 3 × 50 个新 test 日期；
- 这些日期与此前 CAA 的 150 个 outer-test 日期完全不重叠；
- 每个 outer 为 581 train / 50 head-validation / 50 calibration / 50 test；
- MM-JDWind：3 outers × 3 seeds，M=100、jump NFE=12、flow NFE=16；
- DDPM：按原复现惯例每个 outer 使用 seed0，固定 26,000 步，
  250-step DDIM、M=100；
- bootstrap 单位为 calendar day，在每个 outer 内分层重采样，20,000 次。

该证据属于 **post-freeze internal confirmation**，不是外部验证，因为所有日期都来自
同一个 GEFCom2014 数据集。

## 4. 新冻结 test 结果

均值 ± 标准差；MM 两行的 n=9 outer-seed 复制，DDPM 的 n=3 outer 复制。

| 模型 | CRPS | MAE | Coverage 90 | Width 90 | Joint ES | Zero Brier |
|---|---:|---:|---:|---:|---:|---:|
| MM no-jump | 0.081897 ± 0.003186 | 0.116217 ± 0.004272 | 0.744241 ± 0.016456 | 0.373927 ± 0.016602 | 1.728414 ± 0.054230 | 0.046215 ± 0.007560 |
| MM-JDWind mass-preserving | **0.079431 ± 0.003382** | **0.111609 ± 0.004754** | **0.845259 ± 0.019721** | 0.386266 ± 0.017483 | **1.691450 ± 0.056216** | **0.046215 ± 0.007560** |
| DDPM seed0 | **0.079151 ± 0.003088** | 0.117547 ± 0.005459 | 0.842389 ± 0.018462 | 0.468872 ± 0.034473 | **1.688377 ± 0.044588** | 0.065053 ± 0.004879 |

### 4.1 锁定机制相对 no-jump

- CRPS：−0.002466，95% CI `[−0.003165, −0.001794]`；
- MAE：−0.004608，95% CI `[−0.005897, −0.003330]`；
- coverage：+0.101019，95% CI `[+0.086759, +0.116500]`；
- Joint ES：−0.036964，95% CI `[−0.048759, −0.025258]`；
- aggregate CRPS：−0.002985，95% CI `[−0.004160, −0.001845]`；
- adjacency VS：−0.001884，95% CI `[−0.002498, −0.001329]`；
- ramp CRPS：−0.000480，95% CI `[−0.000645, −0.000321]`；
- daily-zone any-zero Brier：−0.243457，
  95% CI `[−0.289015, −0.199576]`。

CRPS 在 9/9 outer-seed 复制中改善，并在 3/3 outer 上方向一致。

解析 `zero_Brier` 没有变化，因为 no-jump 与 mass-preserving 共享同一个解析混合测度
头；变化来自离散 ensemble 能否把解析原子质量实现为有限场景。这一点体现在
finite-zero rate、daily any-zero Brier 和 zero-run length，而不是共享的解析概率。

### 4.2 相对重训 DDPM

- CRPS：+0.000280，95% CI `[−0.000973, +0.001518]`，无显著差异；
- Joint ES：+0.003073，95% CI `[−0.022739, +0.028816]`，无显著差异；
- aggregate CRPS：−0.000077，95% CI `[−0.002136, +0.001985]`，无显著差异；
- MAE：−0.005937，95% CI `[−0.007859, −0.004067]`，MM-JDWind 更好；
- adjacency VS：−0.003198，95% CI `[−0.003966, −0.002452]`，
  MM-JDWind 更好；
- zero Brier：−0.018838，95% CI `[−0.026013, −0.012665]`，
  MM-JDWind 更好；
- width 90：−0.082606，95% CI `[−0.090294, −0.075242]`，
  MM-JDWind 更尖锐；
- ramp CRPS：+0.002527，95% CI `[+0.002125, +0.002951]`，
  DDPM 更好。

DDPM 是逐 zone-day 生成器，因此其跨区域 Joint ES/adjacency VS 只作描述性比较；
边际 CRPS、MAE、coverage、width、ramp 和原子指标是主要公平比较项。

## 5. 预声明成功门

### 相对 no-jump

- 通过：CRPS 相对改善 ≥1%；
- 通过：width 增加 ≤0.02；
- 通过：至少 2/3 outer 方向一致；
- 未通过：解析 zero Brier 改善 ≥5%（两者共享同一原子概率头）；
- 未通过：90% coverage 落在 `[0.88, 0.92]`。

### 相对 DDPM

- 通过：zero Brier 相对改善 ≥5%；
- 通过：width 不恶化；
- 通过：至少 2/3 outer 的 CRPS 方向更好；
- 未通过：平均 CRPS 相对改善 ≥1%；
- 未通过：90% coverage 落在 `[0.88, 0.92]`。

## 6. 可以写进论文的结论

支持的结论：

1. 风功率边界原子不是可忽略的连续噪声；显式混合测度能够稳定改善同一连续生成器；
2. 有限 ensemble 的质量守恒跳跃优于独立跳跃和简单相关跳跃；
3. 改进不仅发生在逐点 CRPS，也发生在联合、聚合与零事件持续性指标；
4. MM-JDWind 与强 DDPM 的 CRPS/Joint ES 统计上相当，同时在 MAE、锐度、
   邻接结构和零原子校准上更好。

不支持的结论：

1. 不能声称 MM-JDWind 全面超过 DDPM；
2. 不能声称已达到 90% 名义覆盖率；
3. 不能把失败的 proper-score 微调写成有效组件；
4. 不能把当前确认实验称为外部验证。

## 7. 后续最值得做的改进

当前最明确的下一步不是继续扩大模型，而是修复两个弱点：

1. 对内部连续流增加显式 ramp/temporal-increment 目标，缩小相对 DDPM 的
   ramp CRPS 差距；
2. 在不破坏原子质量的前提下，对 interior component 做 calibration-only
   scale correction，使 coverage 从约 84.5% 接近 90%。

如果这两点能在新的外部数据集或滚动年份 split 上确认，论文故事将从
“混合测度机制有效且有竞争力”升级为“边界原子与连续动态的统一生成模型”。

## 8. 复现实验入口

```powershell
python -m repro_scripts.run_mm_jdwind_v2 --config repro_configs/mm_jdwind_development_v2.json
python -m repro_scripts.mm_jdwind_candidates --config repro_configs/mm_jdwind_development_v2_selection.json
python -m repro_scripts.mm_jdwind_select_v2 --config repro_configs/mm_jdwind_development_v2_selection.json
python -m repro_scripts.run_mm_jdwind_confirmation --config repro_configs/mm_jdwind_confirmation_v1.json --phase all
python -m repro_scripts.run_ddpm_confirmation --config repro_configs/ddpm_confirmation_v1.json --phase train
python -m repro_scripts.generate_ddpm_confirmation_v2 --config repro_configs/ddpm_confirmation_v1.json --outer 1
python -m repro_scripts.evaluate_ddpm_confirmation --config repro_configs/ddpm_confirmation_v1.json
python -m repro_scripts.mm_jdwind_final_report_v3
pytest --basetemp tmp/pytest_mm_final -q
```

机器验证结果：`110 passed, 9 warnings`。

机器可读的完整统计位于：

- `outputs/mm_jdwind_confirmation_v1/final_analysis/mm_jdwind_final_analysis.json`
- `outputs/mm_jdwind_confirmation_v1/final_analysis/MM_JDWind_FINAL_REPORT.md`

