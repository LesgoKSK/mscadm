# Architecture-v1 R0 stability-v2.3 归因 Pilot

完成日期：2026-08-19  
证据等级：validation-informed engineering attribution；不是独立确认。  
selection/calibration：保持 sealed，访问标志均为 false。

## 结论

R0 的三训练种子不稳定主要来自两个来源：

1. 各 seed 独立训练 E/A；
2. flow 训练使用不同的日期 shuffle 与 flow-time/source-noise 随机路径。

只固定共享 E/A 可让 level CRPS 和 joint ES 稳定性过门，但 ramp CRPS 仍以约 `9.3e-6` 的幅度略超门槛。进一步让三个 flow seed 共用训练随机路径后，三项稳定性门全部通过。

## 三个面板

| 面板 | 语义 | level spread | ramp spread | joint ES 相对 spread | 三项门 |
|---|---|---:|---:|---:|:---:|
| A | E/A、flow init、shuffle/path 均独立 | 0.003645 | 0.001464 | 3.737% | FAIL |
| B | 共享 E/A，flow shuffle/path 独立 | 0.001955 | 0.001209 | 2.254% | FAIL（仅 ramp） |
| C | 共享 E/A，flow init 独立，shuffle/path 共用 | 0.001331 | 0.000422 | 1.510% | PASS |

门槛保持为：level spread≤0.0025、ramp spread≤0.0012、joint ES 相对 spread≤3%。

## 归因量

以 spread 的减少量描述，不解释为因果估计：

| 指标 | A−B：E/A 贡献 | B−C：shuffle/path 贡献 | C：初始化/优化残差 |
|---|---:|---:|---:|
| level CRPS | 0.001690 | 0.000624 | 0.001331 |
| ramp CRPS | 0.000255 | 0.000787 | 0.000422 |
| joint ES 相对 spread | 1.483 pct-pt | 0.744 pct-pt | 1.510% |

E/A 对 level 和 joint 稳定性影响最大；shuffle/path 对 ramp 稳定性影响最大。

## 最小候选 D

归因结果不支持一次性加入 EMA、batch64、cosine schedule 等额外改动。最小 D 应当是：

- 固定并共享同一个 E/A checkpoint；
- 三个 flow seed 仅改变 flow 初始化；
- 三个 seed 共用完全相同的 epoch shuffle 和 flow time/source noise 随机银行；
- 其余模型、优化器、学习率、batch、早停、采样与质量阈值不变。

Pilot C 虽然通过，但 candidate D 在本 pilot 冻结时尚未预注册，因此不能追溯改称正式通过。下一步必须先冻结 D，再从头重跑三 flow seed。

## 完整性

- Pilot result SHA-256：`5b40fb6ff85539b959397677a60f0536781b7b8cdb761b917159e927a162cf0b`
- Pilot freeze SHA-256：`d13e330f271b44f45397c54d78f287bc8139391b0f2bec698bd0e371ee2384fa`
- 输出目录 38 个 payload 与 38 个 sidecar，逐项核验全部通过。

