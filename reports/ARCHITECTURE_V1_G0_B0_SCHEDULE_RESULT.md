# Architecture-v1 G0-B0：信息保留式 SNR 分配可行性结果

完成日期：2026-09-14

正式状态：`G0_B0_SCHEDULE_FEASIBILITY_GO`

证据范围：只使用冻结 G0-A 结果中的 267 天 out-of-fold 六组条件方差与有效秩，在已登记的 cosine-250 网格上进行解析 schedule 审计。本文不包含原始 target 读取、denoiser 训练、生成场景或 validation 模型比较。

冻结协议：[`ARCHITECTURE_V1_G0_B0_SCHEDULE_PROTOCOL.md`](ARCHITECTURE_V1_G0_B0_SCHEDULE_PROTOCOL.md)

## 1. 一句话结论

固定 `eta=0.5` 的 **reserve-then-water-fill** 可以在逐日、逐 diffusion step 精确匹配 IID Gaussian mutual-information budget，同时为每个 mode 至少保留一半 IID information，并保持完整 schedule 有限、严格为正和单调。

因此，G0-B0 的工程可行性门通过；下一步只获准另行冻结一个小型 denoiser utility Probe。当前结果不能说明这种 schedule 比 IID 更容易学习，也不能说明最终场景更好。

## 2. 数据、配置与实现身份

| 项目 | 正式值 |
|---|---:|
| G0-A result SHA256 | `37a55921016bd701ffb2f5a66f56b6abbaa28d027cfdb9b46e0b3f01f8d9b7e7` |
| train days | 267 |
| train date SHA256 | `f67eb5fd4382b2d69d2d22d23488ec618cd3b3b9547a9bbcc71bdd39587c8cf5` |
| G0-B0 config SHA256 | `1e67af3389c55aa6e7127f4ea8f0a0dc6fc7b2fc4d1c38c3b066320f597a6ab8` |
| result JSON SHA256 | `a352e46431b2594b123c2e2479b305c59e85a9c47f4614b69bd38552fe5cdf64` |
| allocation module SHA256 | `bf875b884f2b4bc68982fc4fda85c30f9589385a9edb9be005ff364abbc884ab` |
| runner SHA256 | `34ccde5dd0e072ca93024bd737cc2828dd5ab72e8e376787d1e607ccfc8c3963` |
| schedule module SHA256 | `11cca468b9867a7f0f46f0db85802b0a088a61ca84675e75a352d356154f6d0a` |

runner 解析完整的冻结 G0-A JSON，但每日记录只消费日期、outer-fold、`N1_variance` 和 `effective_rank`。它没有导入原始数据 loader，没有 materialize 任何 raw target，没有构造 denoiser 或 optimizer，也没有加载或保留学习权重。

第一次 dry-run 后、正式执行前，测试发现 frozen config 的 noisy endpoint 十进制显示值与 schedule float32 字节相差约 `8.89e-18`。只修正了该显示值及配置 sidecar；schedule 字节 SHA、公式、阈值和 `eta` 均未改变。该事件已在冻结协议中披露。

## 3. baseline schedule 身份

| 项目 | 结果 |
|---|---:|
| timesteps | 250 |
| `alpha_bar` float32 bytes SHA256 | `ab4df3b8c173b936e8b3a6fb4ecddbcd96d5652fbd1a893c4ab8871b4d6a81dc` |
| clean endpoint `alpha_bar` | 0.9998057485 |
| noisy endpoint `alpha_bar` | 3.8859283791e-8 |
| log-SNR 范围 | `[-17.0633, 8.5462]` |

六组方差全部来自 G0-A 的 out-of-fold N1，范围为 `0.1757–33.7741`；逐日有效秩范围为 `2.0449–108.0000`。allocation 权重由逐日有效秩归一化，row-sum 最大误差为 `2.22e-16`。

## 4. 主候选 `eta=0.5` 结果

| 冻结检查 | 正式结果 |
|---|---:|
| information-budget 最大绝对误差 | `1.776e-15` |
| 相邻 step 单调性比较数 | 398,898 |
| 单调性违规数 | **0** |
| 最大 forward SNR 增量 | `-1.941e-5` |
| SNR 全部有限 | 是 |
| SNR 全部严格为正 | 是 |
| 最小 information retention | **0.5000** |
| log-SNR shift 全范围 | `[-0.86225, 2.28783]` |
| clean endpoint `alpha_bar` 范围 | `[0.99980566, 0.99980587]` |
| noisy endpoint `alpha_bar` 范围 | `[1.943e-8, 2.384e-7]` |

这意味着 schedule 没有靠改变总 Gaussian information budget 获得优势：它只是在六个 mode 之间重新分配同一预算。`eta=0.5` 又保证任何 mode 都不会像裸 water-filling 那样被完全关闭。

分配方向与 G0-A 方差结构一致：`low_common` 获得最多额外信息，其 log-SNR shift 可达 `+2.288`；`high_common` 和 `high_local` 的最小 shift 约为 `-0.862`，但两者仍在每个位置保留至少 50% 的 IID information，SNR 正值比例均为 100%。

## 5. 退化边界与 sensitivity

| `eta` | 最小 retention | 全部 SNR > 0 | log-SNR shift 范围 | Gaussian proxy risk ratio 中位数 ↓ |
|---:|---:|---:|---:|---:|
| 0.00 | 0.00 | 否 | `[-9.946, 2.908]` | 0.9484 |
| 0.25 | 0.25 | 是 | `[-1.584, 2.634]` | 0.9518 |
| **0.50** | **0.50** | **是** | `[-0.862, 2.288]` | 0.9571 |
| 0.75 | 0.75 | 是 | `[-0.414, 1.756]` | 0.9684 |
| 1.00 | 1.00 | 是 | 约 0 | 1.0000 |

`eta=0` 再现裸 reverse water-filling 的硬 cutoff；`eta=1` 退化为 IID，其 allocated VP `alpha_bar` 最大误差为 `2.22e-16`。在合成 equal-variance 条件下，所有 sensitivity 值相对 IID 的 `alpha_bar` 最大误差为 `6.29e-15`。

Gaussian proxy risk 小于 1 是解析优化目标的直接结果，不是独立模型收益：water-filling 本来就是按该代理风险求解的。因此它只能用于检查实现和量化理论上限，不能充当 G0-B 的主要实验证据。

## 6. Go/No-Go 判定

| 冻结门 | 结果 |
|---|---:|
| G0-A 身份与访问边界 | Pass |
| baseline schedule 身份 | Pass |
| budget error ≤ `1e-12` | Pass |
| monotonicity violations = 0 | Pass |
| 所有主候选 SNR 有限且严格为正 | Pass |
| retention ≥ 0.5 | Pass |
| shift 位于 `[-1.0, 2.5]` | Pass |
| clean/noisy endpoints 在界内 | Pass |
| `eta=1` IID identity | Pass |
| equal-variance IID identity | Pass |

全部冻结门通过，故正式状态为：

```text
G0_B0_SCHEDULE_FEASIBILITY_GO
```

## 7. 可复现性

正式结果另行运行到独立的 `/tmp` 目录。移除 `started_utc` 和 `finished_utc` 后，两次完成运行的整个 scientific JSON payload 逐字段完全一致。

预执行还通过了 Python 编译、7 项 fail-closed 针对性测试以及不读取 G0-A evidence 的 dry-run。测试覆盖 schedule 字节身份、Gaussian information 正逆变换、active-set budget、`eta` 退化边界、equal-variance identity、单调性和非法输入拒绝。

## 8. 可以说什么，不能说什么

当前可以说：

> G0-A 的模式级条件方差可以被忠实翻译成一条 budget-matched、无硬 cutoff、数值稳定且可复现的 mode-wise diffusion schedule。

当前不能说：

- reserve-then-water-fill 优于 IID、CW、Fixed-band 或 MuLAN-lite；
- Gaussian proxy risk 的解析下降会转化为真实 denoising utility；
- ramp CRPS、joint ES、lagged variogram 或 atom 指标已经改善；
- Diffusion 已经被选为最终模型家族；
- reverse water-filling 或 spectral schedule 是本项目的数学首创。

下一步不是直接训练完整模型，而是先冻结一个单独的 G0-B tiny-denoiser utility protocol。它必须预先定义六条公平 path、oracle-adjusted reconstruction 指标、learning curve、Shuffled-NWP 反事实、训练预算与 Go/No-Go 门。
