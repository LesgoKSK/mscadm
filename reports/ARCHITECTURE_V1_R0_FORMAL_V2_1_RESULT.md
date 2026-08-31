# Architecture v1：R0 formal-v2.1 三随机种子结果与路线判定

生成日期：2026-08-15（Asia/Shanghai）

## 1. 一句话结论

R0 的三随机种子训练已经完整结束并冻结。共享 NWP 编码器/atom 模块的可学习性门 G0 全部通过；所有预注册的预测质量界限和三种子稳定性界限也全部通过。但正式总判定仍为 **`G1_NO_GO`**，因为三个种子都违反了同样两项预注册工程门：

1. flow 训练在 warm-up 后的梯度裁剪触发率为 100%，高于上限 25%；
2. 同一随机种子在不同 `member_chunk` 下没有达到逐比特完全一致。

因此，这不是“R0 的预测效果失败”，也不是“Rectified Flow 路线被否定”；它表示当前训练/数值协议尚未达到我们事先约定的可审计标准。按照预注册规则，本轮停止在 R0，不进入 T0/T1，也不打开 calibration 或 selection。

权威机器可读结果：`outputs/architecture_v1_formal_v2_1/R0_FORMAL_RESULT.json`。

## 2. 证据边界

formal-v2 使用严格隔离后的日期协议：

| 角色 | 天数 | 本轮是否读取目标值 |
|---|---:|---|
| train | 267 | 是 |
| validation | 50 | 是，仅用于 R0 训练早停和 G0/G1 |
| calibration | 50 | 否，保持 sealed |
| selection | 50 | 否，保持 sealed |
| R-SEEN | 314 | 否，永久隔离 |
| local final | 0 | 无；最终确认需外部数据 |

314 个 R-SEEN 日由先前诊断已经看过的 300 日和 smoke 工程测试触及的 14 日组成。正式训练进程使用 stage-scoped loader，只能物化 train 与 validation；结果文件同时记录：

- `selection_state = sealed`
- `selection_target_accessed = false`
- `calibration_state = sealed`
- `calibration_target_accessed = false`

这意味着后续仍保留一次公平比较候选架构的 selection 机会。

## 3. formal-v2 到 formal-v2.1 的工程修订

首次 formal-v2 尝试在 seed 0 的 E/A 阶段和 G0 完成后停止：新构造的 flow 模型仍在 CPU，而固定 validation bank 已移至 GPU。该次尝试发生在 zero-velocity baseline 之前，flow optimizer 更新次数为 0。

失败证据被原样保留在：

- `outputs/architecture_v1_formal_v2/FORMAL_REVISION_ABORTED.json`
- `repro_configs/architecture_v1_formal_v2_1_amendment.json`

v2.1 修订只允许把 flow 模型移动到已登记的 CUDA 设备，并补充修订身份绑定；数据、模型科学定义、训练预算、随机种子、采样计划和 Go/No-Go 门均未改变。v2.1 从 seed 0 的 E/A 阶段全部重新训练，没有复用失败尝试的权重。

## 4. 正式执行身份

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 2060 |
| 精度 | FP32，AMP=false，TF32=false |
| 训练 seeds | 0、1、2 |
| validation sampling seeds | 21000、21001、21002 |
| 每个场景集合成员数 | 100 |
| flow 积分 | Heun 16 steps，31 flow NFE/path |
| preflight batch | 16 日 |
| preflight 50 updates wall time | 6.14 s |
| preflight peak allocated VRAM | 192,818,688 bytes（约 184 MiB） |
| protocol SHA256 | `f48513f71597f6236aeac71b238338d620b44f19704e6e1c199b5cfca31d8049` |
| gate config SHA256 | `0c1cb8b87fd6d673dbb1a0fa352a3aa7de47c351cfc9d9ca9f436b220629fc4a` |
| retained code SHA256 | `25d68c6bcd2c6f515e75a343349f6c490ea22978e42fe16842ea67b64008c481` |

三 seed 的 atom 阶段均为 130 epochs / 2,210 updates；flow 阶段分别为 270 / 260 / 270 epochs，即 4,590 / 4,420 / 4,590 updates。最佳 flow validation epoch 分别为 149、139、149（从 0 开始计数）。

## 5. G0：共享 E/A 是否真的可学习

G0 的四个条件在三个 seed 上全部通过。上界 exact-one 样本仅 train 1 个、validation 2 个，不满足学习自由 upper-atom 分支的支持度要求，因此按预注册规则固定为 train-only Jeffreys prior `2.3452e-5`，并禁止声称模型学会了 upper atom。

| seed | atom NLL 相对常数先验改善 | interior location loss 相对零预测器改善 | encoder atom 梯度范数 | encoder location 梯度范数 | G0 |
|---:|---:|---:|---:|---:|---|
| 0 | 32.74% | 60.75% | 0.1916 | 0.4030 | 通过 |
| 1 | 30.82% | 60.38% | 0.1893 | 0.1406 | 通过 |
| 2 | 31.91% | 60.53% | 0.1622 | 0.2372 | 通过 |

这说明共享 encoder 并非只学到 atom 分类：train-only 定义的连续功率位置辅助任务也明确向 encoder 传递梯度，并在 validation 上取得稳定改善。

## 6. G1：预测质量与三种子稳定性

### 6.1 每个 seed 的 validation 指标

| 指标 | seed 0 | seed 1 | seed 2 | 三 seed 均值 | 预注册灾难性界限 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| level CRPS ↓ | 0.078482 | 0.080109 | 0.078451 | 0.079014 | ≤ 0.100 | 全部通过 |
| ramp CRPS ↓ | 0.051630 | 0.052075 | 0.051459 | 0.051721 | ≤ 0.065 | 全部通过 |
| normalized joint ES ↓ | 0.110037 | 0.111885 | 0.110513 | 0.110812 | ≤ 2.000 | 全部通过 |
| coverage 90% | 0.84835 | 0.85615 | 0.86915 | 0.85789 | 0.70–0.98 | 全部通过 |
| width 90% | 0.37760 | 0.39429 | 0.39139 | 0.38776 | 0.10–0.70 | 全部通过 |
| zero Brier ↓ | 0.059704 | 0.061653 | 0.059951 | 0.060436 | 用于 dispersion | 通过 |
| atom-state Brier ↓ | 0.119737 | 0.123635 | 0.120231 | 0.121201 | 用于 dispersion | 通过 |

flow 相对 zero-velocity fixed-bank loss 的比值分别为 0.2328、0.2394、0.2298，远低于上限 0.95，说明 flow 确实学到了非平凡速度场。

平均 90% coverage 为 0.8579，仍比名义覆盖率低约 4.2 个百分点。它通过的是用于拦截灾难性失败的宽松界限，并不表示模型已经校准良好；真正的 calibration 仍保持封存，不能在这里调参修正。

### 6.2 三种子离散性

| 指标 | max−min | 预注册上限 | 结果 |
|---|---:|---:|---|
| level CRPS | 0.001658 | 0.002500 | 通过 |
| ramp CRPS | 0.000616 | 0.001200 | 通过 |
| normalized joint ES 相对 spread | 1.67% | 3.00% | 通过 |
| coverage 90% | 0.02080 | 0.03500 | 通过 |
| zero Brier | 0.001949 | 0.006000 | 通过 |
| atom-state Brier | 0.003898 | 0.012000 | 通过 |

ramp CRPS 的三 seed 变异系数约为 0.50%，level CRPS 约为 0.98%。因此，旧 time-domain 基线只有一个训练 seed 的不确定性已经被显著缓解：在当前干净协议下，R0 的预测质量没有表现出明显 seed 偶然性。

这仍然只是 R0 validation 稳定性证据，不能被写成 R0 优于其他模型，更不能直接决定 flow、diffusion 或 SSM 家族。

## 7. 两个导致正式 No-Go 的工程问题

### 7.1 flow 梯度裁剪触发率为 100%

预注册上限是 warm-up 后不超过 25%，三个 seed 实际均为 100%。flow 的 gradient clip 设置为 1.0，而各 epoch 内原始梯度范数均值在 warm-up 后的范围如下：

| seed | epoch-mean raw grad norm 最小值 | 中位数 | 最大值 | clip fraction |
|---:|---:|---:|---:|---:|
| 0 | 2.550 | 3.711 | 5.084 | 1.000 |
| 1 | 2.496 | 3.688 | 5.412 | 1.000 |
| 2 | 2.887 | 3.857 | 5.273 | 1.000 |

训练没有出现 NaN、跳步或回滚，loss 和预测指标也很稳定；所以这不是梯度爆炸的直接证据。但它表示整个 flow 训练几乎处在“由 clip 决定更新尺度”的制度中，当前登记的 `clip=1.0` 与 `clip fraction≤0.25` 明显不相容。不能在看完结果后把门槛直接改成通过。

### 7.2 不同 member chunk 未达到逐比特一致

同一个模型、同一个 sampling seed、同一个 chunk 设置的重放是逐比特一致的；atom latent 在非 active 坐标也严格为零。但是把 `member_chunk` 从 10 改为 20 后，`values + states` 的联合逐比特比较为 false。

当前正式结果只保存了布尔判定，没有保存以下必要诊断：

- states 是否仍完全一致；
- values 的 max/mean absolute difference；
- 不同单元格数量；
- 在预先锁定的 atol/rtol 下是否数值等价；
- 指标差异是否为零或远低于报告精度。

批大小改变可能导致 GPU 矩阵乘法累加顺序不同，这是一个合理假设，但在量化之前不能把失败自动归类为“只有无害的浮点尾差”。

在保持 v2.1 正式结论冻结以后，又对三个冻结 checkpoint 做了一个 **CPU-only、validation-only 的后验工程敏感性审计**。它没有访问 calibration/selection，也不替代原 CUDA 门控；结果如下：

| seed | chunk 10/20 非逐位相同元素 | values 最大绝对差 | 99% 分位差 | proper-score 最大差 |
|---:|---:|---:|---:|---:|
| 0 | 4.90% | `1.79e-7` | `5.96e-8` | `<9e-10` |
| 1 | 4.37% | `2.98e-7` | `5.96e-8` | `<9e-10` |
| 2 | 4.74% | `2.38e-7` | `5.96e-8` | `<4e-10` |

CPU 审计中三个 seed 的 atom states 都逐位相同，coverage 与 atom 指标也完全相同。这强烈支持“连续值浮点尾差”的解释，但由于硬件与原 CUDA 门控不同，v2.1 仍保持 No-Go；它只用于在 v2.2 训练前冻结一个可检验的数值容差。

## 8. 为什么现在不能进入 T0

预注册的 G1 是交集门：每项都必须通过。预测质量好、三种子稳定，并不能覆盖数值协议失败。若现在继续 T0，会产生两个问题：

1. R0 与 T0 的训练更新尺度可能都被低阈值裁剪主导，机制比较难以解释；
2. chunk-dependent 输出会削弱 R0/T0 公共随机数配对的可复现性，尤其是未来因显存改变 chunk 时。

所以本轮的正确判定是：**R0 科学表现“可继续”，formal 工程验收“No-Go”，路线暂不升级。**

## 9. 下一阶段最小动作

在不读取 calibration/selection 的前提下，先做一个 engineering-only R0 修订周期：

1. **补齐 chunk 数值审计。** 在相同 checkpoint、allocation、initial noise 下，同时保存 chunk 10/20 的 state equality、max/mean absolute difference、非零差单元数、数值等价阈值和指标差。原 v2.1 判定保持 No-Go，不能追溯改写。
2. **做 train-only 梯度尺度审计。** 用丢弃权重的短 pilot 保存逐 update 的 pre-clip gradient norm 分布，检查分模块贡献，确定是 clip 阈值过低、loss 缩放问题，还是少数 batch 驱动。
3. **冻结 formal-v2.2 修订。** 若证据支持，应在训练前登记新的 clip/优化器设置和数值等价门；科学质量门、日期、seeds、M=100、Heun16 和 selection 规则保持不变。
4. **从 seed 0 全部重训 R0。** 不复用 v2.1 权重。只有 v2.2 的 G0、每 seed G1、三 seed dispersion 和新数值门全部通过，才实现 T0。
5. **继续封存 selection/calibration。** T0、T1-feature、T1-source、T1-shuffle 的代码、配置与选择分支仍需在首次打开 selection 前冻结。

如果 chunk 差异被证实只是小于预登记容差的浮点尾差，未来规则应改为“状态完全一致 + 数值容差 + 指标容差”，而不是要求跨不同 GEMM 批形状逐比特一致。这个修改只对未来 v2.2 生效。

## 10. 完整性核验

- 3/3 atom completion 存在；
- 3/3 flow completion 存在；
- 3/3 atom best checkpoints 存在；
- 3/3 flow best checkpoints 存在；
- training freeze 已写入且 `selection_authorized=false`；
- v2.1 输出目录中未发现 failure bundle；
- 正式输出下 45 个 payload 与 45 个 `.sha256` sidecar 一一对应并全部通过校验；
- training freeze SHA256：`bda9124e3232fd805874a32b4553b0d356f69be9c5adfdda8debae68a14a979b`；
- formal result SHA256：`e26e361f87da7d76fbd0fc514eef3235bed54d301c2b981e7873ea1e7b4e1b06`；
- protocol、model、training、formal-v2、formal-training、smoke-runner 六组自包含测试全部退出 0，共显示 28 项 PASS，另有 model 测试组以静默成功退出。

## 11. 关键文件

- 预注册说明：`reports/ARCHITECTURE_V1_FORMAL_V2_PREREGISTRATION.md`
- Go/No-Go 门：`repro_configs/architecture_v1_go_no_go_v1.json`
- formal-v2 配置：`repro_configs/architecture_v1_formal_v2.json`
- v2.1 工程修订：`repro_configs/architecture_v1_formal_v2_1_amendment.json`
- 正式结果：`outputs/architecture_v1_formal_v2_1/R0_FORMAL_RESULT.json`
- 训练冻结：`outputs/architecture_v1_formal_v2_1/r0_training.freeze.json`
- 初次失败证据：`outputs/architecture_v1_formal_v2/FORMAL_REVISION_ABORTED.json`
