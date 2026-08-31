# STGF-Flow 研究结论报告

## 1. 研究问题与方法

本项目检验的核心假设是：将 10 个 Zone 的风电联合场景映射到“训练集相关图的图傅里叶域 × 24 小时 DCT 频域”，再用联合 Rectified Flow 建模残差，是否能比时域联合 flow 更准确地恢复空间—时间依赖。

完整模型暂名 **STGF-Flow（Spatio-Temporal Graph-Frequency Flow）**，包含：

1. 仅用训练集目标相关性构建正权重、对称 kNN 图；
2. 归一化图 Laplacian 特征向量作为 GFT 基；
3. 正交 DCT-II 作为时间频率基；
4. 物理空间条件中心网络；
5. 图模态 × 时间频率网格上的联合 Rectified Flow；
6. Heun ODE 采样、物理均值锚定、残差温度校准；
7. 严格 `[0,1]` 物理边界；
8. CRPS、覆盖率、Energy Score、邻接 Variogram、频谱能量、跨 Zone 相关等联合评价。

四个同容量消融为：

- time-domain：空间和时间均不变换；
- graph-only：仅 GFT；
- time-frequency：仅 DCT；
- STGF：GFT + DCT。

## 2. 防泄漏实验协议

开发阶段只使用旧划分的 calibration 数据选择表示、均值锚定系数和残差温度。最初发现候选的频谱误差使用了各自坐标基，跨表示不可比；在访问确认 test 前，已改为每个开发划分统一使用仅由训练集拟合的 GFT×DCT 评估基重算全部 72 个候选。

修正后的锁定结果仍为：

- 表示：STGF；
- physical mean anchor：0.50；
- residual temperature：0.90；
- 选择锁 SHA256：`f52d2b5f446e9e93624a09d209e1c5d1d14c4cfff073f60228c6acae7e2b2261`。

确认实验另外冻结了 150 个新日期，和此前 CAA/MM 的 300 个 test 日期零重叠。三个互斥 test blocks 各 50 日。排除全部 150 个确认日期后，三个 block 共用：

- train：481 日；
- validation：50 日；
- calibration：50 日。

因此正确设计是训练 **6 个唯一 flow 模型**，分别在三个互斥 test blocks 上评估，而不是把完全相同的训练过程包装成 18 次独立重训。重复训练审计也证实，相同 seed/mode 在 outer1/outer2 得到的权重哈希逐字节一致。

确认阶段共生成：

- 18 个 flow 评估单元：6 模型 × 3 test blocks；
- 每单元 100 条 `10 Zone × 24 h` 联合轨迹；
- 1 个 26,000-step DDPM seed0 对照 × 3 test blocks；
- STGF 采样 16 NFE，DDPM 采样 250 steps。

## 3. 确认实验结果

表中均值和标准差来自三个 test blocks；“STGF all seeds”包含 3 seeds × 3 blocks 共 9 个评估单元。所有频谱指标均在统一训练集 GFT×DCT 基下计算。

| 模型 | CRPS ↓ | MAE ↓ | Coverage90 | Width90 ↓ | Joint ES ↓ | Adj. VS ↓ | Spectral MAE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time-domain seed0 | 0.079947 ± 0.002419 | 0.108948 ± 0.003224 | 0.786861 ± 0.007676 | 0.334026 ± 0.014319 | 1.707561 ± 0.033115 | 0.043812 ± 0.002159 | 0.070765 ± 0.001644 |
| Graph-only seed0 | 0.080638 ± 0.002170 | 0.110210 ± 0.003177 | 0.814250 ± 0.011480 | 0.364902 ± 0.014089 | 1.706967 ± 0.031201 | 0.044109 ± 0.001857 | 0.072896 ± 0.002706 |
| Time-frequency seed0 | 0.081640 ± 0.001918 | 0.111690 ± 0.003636 | 0.866278 ± 0.007325 | 0.438484 ± 0.014315 | 1.708508 ± 0.030013 | 0.046481 ± 0.001375 | 0.075584 ± 0.004201 |
| STGF seed0 | 0.082572 ± 0.001834 | 0.111806 ± 0.003393 | 0.876611 ± 0.006352 | 0.461083 ± 0.007560 | 1.716764 ± 0.027330 | 0.046881 ± 0.001474 | 0.076519 ± 0.003122 |
| STGF all seeds | 0.083135 ± 0.001453 | 0.113274 ± 0.002616 | 0.872185 ± 0.008407 | 0.467544 ± 0.010865 | 1.731220 ± 0.023567 | 0.047690 ± 0.001372 | 0.078209 ± 0.002616 |
| DDPM seed0 | 0.083188 ± 0.003405 | 0.122494 ± 0.006612 | 0.859583 ± 0.004311 | 0.462063 ± 0.032055 | 1.790156 ± 0.078219 | 0.047527 ± 0.001642 | 0.095210 ± 0.013122 |

### 3.1 完整图频域假设没有得到确认

STGF 三 seed 平均相对 time-domain：

- CRPS 恶化 3.99%；
- Joint ES 恶化 1.39%；
- 邻接 Variogram 恶化 8.85%；
- MAE 恶化 3.97%；
- 90% 覆盖率从 0.787 提高到 0.872，但仍未进入预注册的 `[0.88,0.92]`；
- 区间宽度从 0.334 增至 0.468。

逐日、按 test block 分层的 20,000 次 paired bootstrap：

- STGF 三 seed 平均 − time-domain 的 CRPS 差值为 `+0.003187`；
- 95% CI 为 `[+0.002281,+0.004099]`；
- 三个 test blocks 中 STGF 的 CRPS 均未优于 time-domain。

所以这不是“没有显著涨点”，而是有统计证据表明当前固定图频域方案在 CRPS 上更差。

### 3.2 固定相关图没有提供净增益

STGF seed0 相对 time-frequency seed0：

- CRPS 差值 `+0.000933`；
- 95% CI `[+0.000428,+0.001440]`；
- 三个 test blocks 均更差。

时间 DCT 主要负责扩大不确定性并修复覆盖率；它本身也没有改善 CRPS。继续加入训练相关图 GFT 后，覆盖率仅小幅增加，但 CRPS 确认性恶化。因此不能把“固定相关图有效”写成论文正贡献。

### 3.3 联合 Rectified Flow 本身仍有正结果

STGF all seeds 与 250-step 独立 DDPM 相比：

- CRPS：0.083135 vs 0.083188，基本持平；逐日差值 95% CI `[-0.001976,+0.001814]`；
- MAE：降低 0.009220，95% CI `[-0.012627,-0.005960]`；
- Joint ES：降低 0.058936，95% CI `[-0.100574,-0.018546]`；
- Spectral MAE：0.078209 vs 0.095210，约低 17.9%；
- 仅使用 16 NFE，而 DDPM 使用 250 个采样步；
- 原生输出 10 Zone 联合轨迹，而 DDPM 对照逐 Zone-day 独立生成。

但 STGF 的 ramp CRPS 和 zero Brier 明显差于 DDPM，因此只能表述为“高效且有竞争力”，不能宣称全面优于 DDPM。

## 4. 预注册门槛判定

六项门槛全部未通过：

- CRPS 至少改善 1%：未通过；
- Joint ES 至少改善 1%：未通过；
- 邻接 Variogram 至少改善 1%：未通过；
- Coverage90 位于 `[0.88,0.92]`：未通过；
- 至少 2/3 test blocks 的 CRPS 更好：未通过；
- 次要指标无超过 1% 的退化：未通过。

因此，**当前 STGF 固定图频域版本不应作为“已涨点”的论文主方法投稿。**

## 5. 从发论文角度应如何转向

最有证据的主线不是“固定相关图有效”，而是：

> 低 NFE 的跨 Zone 联合 Rectified Flow，可以在显著少于 DDPM 的采样步数下，取得相当的边际 CRPS、更低的 MAE和更好的联合 Energy/Spectral 结构。

下一版建议把固定 GFT 降为被严格否证的消融，并研究真正可能涨点的图机制：

1. 条件动态图：邻接随天气条件、季节和预测时刻变化；
2. 可学习谱基或低秩空间混合器：不强迫相关性图等于生成依赖图；
3. 图机制只作用于协方差/残差依赖，避免同时放大所有模态方差；
4. 直接以 calibration-only proper score 训练可学习图，但仍需重新冻结新 test 或 nested cross-fitting；
5. 保留 16-NFE 联合 flow 作为已经得到支持的效率主干。

## 6. 可复核产物

- 开发选择锁：`outputs/stgf_development_v2/selection.common_basis.lock.json`
- 确认配置：`repro_configs/stgf_confirmation_v2.json`
- 确认清单：`outputs/stgf_confirmation_v2/confirmation_manifest.json`
- 数值分析：`outputs/stgf_confirmation_v2/final_analysis/stgf_final_analysis.json`
- 完整性审计：`outputs/stgf_confirmation_v2/final_analysis/completion_audit.json`
- 全量测试：114 passed

证据范围仍应明确写为：**post-lock internal confirmation on new frozen splits, not external validation**。
