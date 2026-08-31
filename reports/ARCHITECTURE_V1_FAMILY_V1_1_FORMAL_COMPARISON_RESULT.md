# Architecture-v1 family-v1.1：F0 vs D0-v 正式 Validation 比较

日期：2026-08-26  
正式状态：**FAMILY_V1_1_UNRESOLVED_TRADEOFF_OR_NO_DOMINANCE**  
正式胜者：**无**

## 1. 评估是否公平

本次使用六个 best EMA checkpoint：family-v1 中复用的 F0 seeds 3/4/5，以及 family-v1.1 新训练的 D0-v seeds 3/4/5。

每个 checkpoint 都生成 3 个采样重复，每个重复包含 50 个 validation 日、100 个 members、10 个区域和 24 小时。双方都使用 31 次网络前向计算：F0 为 16-step Heun，D0-v 为 31-step deterministic DDIM。

共同随机契约终审通过：对于 sampling seeds 21000/21001/21002，六个模型的日期、truth、mask、atom probabilities 和分配后的 states 逐数组完全相同；initial noise 使用相同 CUDA generator、shape、dtype 和 `seed+2`；所有 18 个归档的 interior 解码边界计数均为 0。

六个 per-seed hard gate 和两个三种子稳定性门全部通过。selection 与 calibration 始终封存且未访问。

## 2. 主要结果

以下为先在同一天内平均三个训练种子和三个采样重复，再对 50 个 validation 日求均值的结果。CRPS、ES、variogram、Brier 和 width 越低越好；coverage90 应结合 0.90 名义覆盖率解释。

| 指标 | F0 | D0-v | 直观结果 |
|---|---:|---:|---|
| level CRPS | **0.079661** | 0.080484 | F0 低 0.000823，约 1.03% |
| ramp CRPS | **0.051627** | 0.051648 | 几乎相同，F0 低 0.000021 |
| normalized joint ES | **0.111059** | 0.112640 | F0 低约 1.40% |
| lagged increment variogram | 0.017720 | **0.017189** | D0-v 低约 3.00% |
| coverage90 | **0.851658** | 0.838193 | F0 高 1.35 个百分点，两者均低于 0.90 |
| width90 | 0.378115 | **0.368512** | D0-v 窄约 2.54%，但同时 coverage 更低 |
| atom-state Brier | 0.119737 | 0.119737 | 完全相同 |
| zero Brier | 0.059704 | 0.059704 | 完全相同 |
| one Brier | 0.000167 | 0.000167 | 完全相同 |

atom 指标完全相同是预期结果，因为两边使用同一个冻结 E/A，而不是 transport family 学出了相同 atom law。

## 3. 三种子稳定性

两个候选都通过全部预注册稳定性门。

| 候选 | level CRPS spread | ramp CRPS spread | joint ES relative spread | coverage spread | 结论 |
|---|---:|---:|---:|---:|---|
| F0 | 0.002361 | 0.000209 | 2.856% | 0.015807 | PASS |
| D0-v | 0.001847 | 0.000774 | 2.003% | 0.022501 | PASS |

D0-v 在 level CRPS 和 joint ES 的种子 spread 略小；F0 在 ramp CRPS 和 coverage spread 上更小。这里同样不存在单方向全面优势。

## 4. 为什么没有选出胜者

预注册判定不是“平均分略低就获胜”，而是要求候选：

1. 通过自身 hard gate 与三种子稳定性门；
2. 对另一候选在全部关键指标上非劣；
3. 至少一个 proper score 达到预注册的实质优效幅度并有 paired-day CI 支持；
4. 若两者等价，才允许用至少 20% 的采样加速作为 tie-break。

实际情况：

- F0 对 D0-v 在全部非劣门上通过，但没有 proper score 达到实质优效幅度：level CRPS 改善约 0.000823，小于 0.001；joint ES 改善约 1.40%，小于 2%。
- D0-v 的 lagged variogram 明确更低约 3.00%，但未达到 5% 实质优效幅度；同时它未通过 level CRPS 和 joint ES 的全指标非劣要求。
- 双向等价检验没有通过：level CRPS 的双侧界约 0.001616，超过 0.0015 margin；joint ES 的双侧界约 2.316%，超过 2% margin。
- F0 的采样时间约快 16.64%，paired CI 约为 15.47%–17.80%，但低于预注册的 20% tie-break 门。

因此正式结论只能是：**两者均有效且稳定，但呈现 marginal/joint calibration 与 lagged dependence/sharpness 之间的 trade-off，没有满足预注册规则的 family winner。**

## 5. 结果文件

- 正式比较：`outputs/architecture_v1_family_v1_1/formal_evaluation/FAMILY_COMPARISON.json`
- 18 个场景归档：`outputs/architecture_v1_family_v1_1/formal_evaluation/archives/`
- 每种子汇总：`outputs/architecture_v1_family_v1_1/formal_evaluation/per_seed/`
- 简表：`outputs/architecture_v1_family_v1_1/formal_evaluation/per_seed_summary.csv`
- calendar-day 推断数组：`outputs/architecture_v1_family_v1_1/formal_evaluation/family_seed_averaged_per_day.npz`
- 训练与评估冻结：`outputs/architecture_v1_family_v1_1/formal_training/training.freeze.json`

正式比较文件 SHA256：`4e33c2d3ad16daa84392cd48dcec420efeef0b20f59d4774cbd61d24f509f090`。
