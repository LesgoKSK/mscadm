# Architecture-v1 family-v1.1：D0 v-prediction 修订与训练结果

日期：2026-08-26  
当前结论：**family-v1 已冻结为 D0 sampler contract No-Go；family-v1.1 的 P0 与三种子 D0-v 正式训练均通过。尚未完成共同 validation scenarios，因此不能宣布 Flow 或 DDPM 更好。**

## 1. 为什么关闭 family-v1

family-v1 的 D0 使用 epsilon-prediction。第一次正式 validation 采样即违反输出语义：在 1,094,220 个 active interior 坐标中，469,518 个解码为精确 0，507,402 个解码为精确 1，边界占比 89.280035%，50/50 个 validation 日都受影响。latent 虽然有限，但范围达到约 [-11196.73, 12674.73]。

原因是 epsilon-prediction 的 DDIM 反演包含

`x0 = (xt - sqrt(1-alpha_bar) * epsilon) / sqrt(alpha_bar)`。

余弦日程的噪声端点 `alpha_bar` 很小，网络误差会被除法大幅放大。这不是指标稍差，而是采样器输出违反“interior 必须严格位于 (0,1)”的契约，所以 family-v1 不能继续比较模型优劣。

冻结证据：`outputs/architecture_v1_family_v1/FAMILY_V1_NO_GO.freeze.json`，SHA256 `5297d3c817281d7563128a7c62cd8d16757fac76e6424c48278a77f691055aa5`。

## 2. family-v1.1 的唯一算法改动

D0-v 改为 v-prediction：

`v = sqrt(alpha_bar) * epsilon - sqrt(1-alpha_bar) * x0`

采样时使用不含小数除法的正交逆变换：

`x0 = sqrt(alpha_bar) * xt - sqrt(1-alpha_bar) * v`

`epsilon = sqrt(1-alpha_bar) * xt + sqrt(alpha_bar) * v`

本轮明确**没有**同时加入 train-only latent normalization、SNR weighting 或 predicted-x0 clipping。raw-logit latent、250 步 cosine schedule、31-NFE deterministic DDIM、网络拓扑、优化器、EMA、数据、E/A 与 atom allocation 全部保持不变。这样可以把结果归因到参数化本身。

冻结协议：`repro_configs/architecture_v1_family_v1_1.json`，SHA256 `a3850ac18082b6e5e430e3a63d0371622338e59496dca41be6e51515a83328d4`。

## 3. P0 运行级检查

F0 reference 与 D0-v 各执行 50 次 train-only 更新；所有临时权重随后删除。两者均通过：

- 初始 tensor hash 相同；
- 每次更新只调用一次共同网络；
- loss、梯度、optimizer、EMA 与 sampler 全部有限；
- checkpoint resume 的 online、EMA、optimizer、RNG 与重放更新逐位一致；
- 固定 train-proxy 随机库精确重放；
- 31-NFE 采样与 member chunk 检查通过；
- inactive latent 恒为 0，zero/one atom 精确重建；
- active interior 解码边界计数为 0；
- 峰值显存约 0.205 GiB，低于 4.5 GiB 门；
- validation、calibration、selection、R-SEEN 与 final targets 均未访问。

P0 结果：`outputs/architecture_v1_family_v1_1/P0_preflight/P0_RESULT.json`，状态 `P0_GO`，SHA256 `364c9fe703b3eb036c7cea5e6fce7a4d71b36b73e0df0530871f5140f37577ef`。

## 4. 三种子 retained 训练

F0 不重训，继续复用 family-v1 中已通过采样与单种子指标门的三个 best EMA checkpoint，并逐个绑定 SHA256。D0-v 对 seeds 3/4/5 重新训练。

| seed | 完成 epoch | optimizer updates | best epoch | best fixed-bank EMA v-loss | 训练门 | best checkpoint SHA256 |
|---:|---:|---:|---:|---:|:---:|---|
| 3 | 370 | 6290 | 250 | 0.98449863 | PASS | `9be812f35562bf01ced66220f28b3e97fe79088940868a3d09950fa4e0ba3c6e` |
| 4 | 370 | 6290 | 250 | 0.97862255 | PASS | `b65ad95f0ce5bcb24963b8a2b321a9cc8453af1849c58eb5624cbf0cf7f26685` |
| 5 | 370 | 6290 | 250 | 0.96425606 | PASS | `ec471bdb0bce4b2e77152db5ed5401fedb244666c866f9a40ee9978d61848420` |

三个 run 都在 epoch 250 达到预注册意义上的 best，随后 validation loss 平台/回升，并在 stale validation 达到 12/12 时于 epoch 370 自动早停。三个 run 的 frozen E/A state hash 均保持为 `270d3ec01ed70954b33c4a44e4ff249121f64cd5081743ceea3b32c681c69ef6`。

正式训练总结果：`outputs/architecture_v1_family_v1_1/formal_training/TRAINING_RESULT.json`，状态 `THREE_OF_THREE_D0_V_TRAINING_COMPLETE`，SHA256 `b26d0cadde76e15e397ff73e3bad1a544f874114253c282485e1ed9b90005f8b`。

## 5. 当前能说和不能说的结论

现在能说：v-prediction 已通过运行级 sampler contract，并能稳定完成三种子训练；旧 epsilon-prediction D0 已封存，不再参与有效比较。

现在不能说：D0-v 的 CRPS、ramp CRPS、joint ES、lagged variogram、coverage、width 或 atom 指标优于 F0；训练 v-loss 也不能替代这些正式场景指标。

后续共同 validation scenarios 已完成。正式结果为 `FAMILY_V1_1_UNRESOLVED_TRADEOFF_OR_NO_DOMINANCE`，family winner 仍为空。详见 `reports/ARCHITECTURE_V1_FAMILY_V1_1_FORMAL_COMPARISON_RESULT.md`。
