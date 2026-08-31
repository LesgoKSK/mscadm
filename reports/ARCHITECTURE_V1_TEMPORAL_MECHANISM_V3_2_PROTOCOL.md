# Architecture-v1 时间机制 v3.2 冻结实验协议

冻结日期：2026-08-25  
配置：`repro_configs/architecture_v1_temporal_mechanisms_v3_2.json`  
配置 SHA-256：`86f261bd6dcf6bc2c64dd9126ba5733e5131e1d0a85f0bc7cc4a88e2e447db3c`  
证据属性：validation-informed architecture development，不是独立确认实验。  
角色状态：selection、calibration 均封存且未授权访问。

## 1. 实验问题

本实验只回答两个机制问题：

1. NWP 条件特征中的顺序记忆是否真正改善风电轨迹？
2. 方差保持 AR 随机源中的顺序相关是否真正改善风电轨迹？

“真正改善”不仅要求优于无新增记忆的 T0，还要求正常 1–24 小时顺序优于参数完全匹配、但小时访问顺序被打乱的负对照。

本实验不回答 Flow 是否优于 Diffusion。模型家族比较将在时间机制裁决后使用 matched joint DDPM 单独进行。

## 2. 五个候选

| ID | 条件特征 | 随机源 | 作用 |
|---|---|---|---|
| T0 | memoryless matched cell | IID N(0,1) | 无新增顺序记忆基线 |
| T1-feature | 1–24 小时递推 | IID N(0,1) | 检验条件特征记忆 |
| T1-source | memoryless | 方差保持 chronological AR | 检验随机源记忆 |
| T1-feature-shuffle | 固定乱序递推 | IID N(0,1) | feature 负对照 |
| T1-source-shuffle | memoryless | 固定乱序 AR | source 负对照 |

固定 Shuffle 顺序为：

`[17,2,12,0,15,18,1,11,16,20,13,7,5,4,9,23,22,10,3,8,6,21,19,14]`

所有候选共享相同 E/A、Flow velocity 网络、参数拓扑、同一种子初始张量、日期顺序、训练路径随机数、validation bank、场景成员数和积分器。

## 3. 种子与 12 个新训练

正式模型种子为 0、1、2。

v3.1 已完成且冻结的三个 seed0 checkpoint 按精确哈希复用：

- T0：`32a34b11...f1f4ab`
- T1-feature：`76658e2c...7c9cd0`
- T1-source：`6a649b0e...ea1c9`

它们被明确标记为“已经观察过的开发证据”，不伪装成新的独立验证。

新训练固定为以下 12 个，顺序不可变：

1. T1-feature-shuffle / seed0
2. T1-source-shuffle / seed0
3. T0 / seed1
4. T1-feature / seed1
5. T1-source / seed1
6. T1-feature-shuffle / seed1
7. T1-source-shuffle / seed1
8. T0 / seed2
9. T1-feature / seed2
10. T1-source / seed2
11. T1-feature-shuffle / seed2
12. T1-source-shuffle / seed2

## 4. 训练与采样预算

- AdamW，LR=1e-4，weight decay=0
- batch=16 个 calendar days
- gradient clip=5
- 最大 720 epoch
- 每 10 epoch 使用固定 K=4 validation bank
- 至少训练 120 epoch，早停 patience=12 次 validation
- FP32、AMP=false、确定性算法
- M=100 场景成员
- sampling seeds=[21000,21001,21002]
- 16-step Heun，即每条路径 31 次 Flow network forward
- 推断与 bootstrap 单位均为 calendar day，不能把成员或格点当独立样本

## 5. 单种子硬门

每个 candidate×seed 必须全部满足：

- 训练、采样、指标全部 finite；
- gradient clip fraction ≤0.50；
- coverage90 在 [0.75,0.98]；
- width90 在 [0.25,0.80]；
- level CRPS ≤0.13；
- ramp CRPS ≤0.07；
- normalized joint ES ≤0.18；
- inactive source latent 精确为 0；
- source mean 绝对值 ≤0.03；
- source variance 在 [0.92,1.08]；
- `abs(rho)` 不超过预注册上限 0.95。

另外报告 `abs(rho)>=0.94` 的比例，但不在运行后移动阈值。

## 6. 三种子与机制裁决门

三种子稳定性要求：

- level CRPS 最大种子差 ≤0.0025；
- ramp CRPS 最大种子差 ≤0.0012；
- normalized joint ES 相对种子差 ≤3%；
- coverage90 最大种子差 ≤0.04。

T1 相对 T0 必须满足：

- ramp CRPS 平均改善至少 0.001，且 paired calendar-day bootstrap CI 下界 >0；
- lagged increment variogram 相对改善至少 5%，且 CI 下界 >0；
- level CRPS、joint ES、coverage 和 width 通过预锁非劣门。

Chronological 相对 Shuffle 必须满足：

- ramp CRPS 改善至少 0.0005，且 CI 下界 >0；
- lagged increment variogram 改善的 CI 下界 >0。

不能以“不显著”解释为相等或优越。

## 7. 决策规则

- 两个 T1 均失败：保留 T0，进入 matched joint DDPM。
- 只有 feature 通过：保留 feature recurrence，拒绝 source recurrence 主张。
- 只有 source 通过：先审查训练成本与 rho 饱和，再决定是否保留。
- 两者都通过：默认优先更便宜的 feature；source 必须提供足以抵偿成本的依赖改善。

无论结果如何，v3.2 都不授权打开 selection/calibration。时间机制冻结后，下一阶段先实现 matched joint DDPM，再一次性冻结模型家族候选和最终选择规则。

## 8. 运行与恢复

启动命令：

```bash
python \
  repro_scripts/run_architecture_v1_temporal_mechanisms_v3_2.py \
  --execute-training
```

中断后仅允许以相同代码、配置、数据、环境和 identity hashes 从完整 epoch 恢复：

```bash
python \
  repro_scripts/run_architecture_v1_temporal_mechanisms_v3_2.py \
  --execute-training --resume
```

无 `--resume` 遇到非空输出目录将拒绝运行。全部 15 个 candidate×seed 完成评估后才写 `training.freeze.json`。

## 9. 当前状态

v3.2 已于 2026-08-26 完成并冻结，正式状态为 `V3_2_T1_NO_GO`。12 个新训练和 15 个 candidate×seed 评估全部完成，无 failure bundle；selection/calibration 始终封存。详细解释见：

`reports/ARCHITECTURE_V1_TEMPORAL_MECHANISM_V3_2_RESULT.md`

运行产物位于：

`outputs/architecture_v1_temporal_mechanisms_v3_2/`
