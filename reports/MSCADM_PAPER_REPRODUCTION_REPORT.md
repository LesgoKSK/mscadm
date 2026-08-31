# MS-CADM：论文方法与复现专题报告

网络架构、扩散原理、代码映射与实验审计

复现对象：Zhang et al., Electric Power Systems Research 255 (2026) 112753

数据：GEFCom2014 Wind · 10 zones · 731 days · 24 hours

本地档案：multi_scaleCADM / outputs/full_reproduction

报告日期：2026-08-09

[[COVER_END]]

## 阅读指南

本报告只讨论原始 MS-CADM 论文及其 baseline 复现，不把 CR-MS-CADM、RAHC、CAA、MM-JDWind、STGF-Flow 或 PS-DFSC 等后续改进结果混入原论文结论。阅读时需要区分四类证据：**论文披露**指 PDF 中明确写出的公式、图表或配置；**本地补全**指论文缺失细节后采用的可执行工程假设；**正式复现产物**指 `outputs/full_reproduction` 下带配置、场景、表格或哈希的结果；**诊断分支**用于解释失败原因，不等同于论文 headline。

> 核心结论：MS-CADM 的方法思想和主要实验链已经被完整地工程重构，但论文没有给出足以唯一确定实现的网络尺寸、扩散日程、方差损失权重、掩码粒度和缩步采样规则。当前代码应称为“基于论文原理与标准 Improved-DDPM/DiT 解释构造的忠实实现”，不能称为作者私有实现的逐行复制。该实现未复现论文的核心数值：headline CRPS 为 0.107400，而论文为 0.0873；名义 90% 区间 coverage 仅 0.3595，表现出严重欠离散。

## 专题报告结构

1. 论文任务、贡献与原始主张
2. 数据、条件输入与实验样本口径
3. 条件扩散的数学原理
4. MS-CADM 网络架构与张量流
5. 论文模块到代码实现的逐项映射
6. 训练、学习方差、RCM 与采样协议
7. 评价指标和 baseline 重构
8. 主表、区间、单区、步数、消融与 SUC 复现结果
9. 未复现原因和原文可复现性漏洞
10. 工程审计、资产索引与 baseline 使用建议

# 1. 论文任务、贡献与原始主张

## 1.1 文献元信息

复现对象为 Jiawei Zhang、Shuhao Liu、Zeyi Shi 和 Yuancheng Li 的论文《Wind power scenario generation via multi-scale condition adaptive diffusion model》，发表于 Electric Power Systems Research 255 (2026) 112753，DOI 为 [10.1016/j.epsr.2026.112753](https://doi.org/10.1016/j.epsr.2026.112753)。论文第 9 页只写明数据可按请求提供，没有公开代码、训练权重或完整配置。

论文研究的是日前条件概率场景生成：给定未来 24 小时 NWP 条件 `c`，学习风功率轨迹 `x₀` 的条件分布并生成 `M=100` 条场景。形式上可写为：

> Xₛ:ₛ₊ₛ ~ q(pₛ:ₛ₊ₛ | cₛ:ₛ₊ₛ)。输出不是单一预测曲线，而是能够表达时间相关性和不确定性的 24 小时轨迹集合。

## 1.2 论文针对的三个问题

- 单层条件嵌入难以同时抽取不同时间尺度的 NWP 模式。
- 将扩散时间步和外部气象条件直接混合，可能增加网络区分不同条件语义的负担。
- 只学习反向过程均值、固定方差，可能产生过平滑场景并削弱峰值表达。

论文相应提出四项机制：多尺度条件嵌入 CE、时间步驱动的 AdaLN Transformer、learned variance，以及随机/随机性条件遮蔽 RCM/SCM。论文正文有时称“三个主要组件”，但表 4 实际消融了四项，这也是原文表述不一致之一。

## 1.3 原始结果主张

论文摘要和第 6 页声称，相对 VAE，MS-CADM 的 MAE 降低约 4.26%，RMSE 降低约 1.91%；表 1 中 MS-CADM 在 MAE、RMSE、CRPS、QS 和 ES 上最优，但 VS=18.14 并非最优，VAE 的 VS=17.87 更低。论文还通过单区域实验、去噪步数、组件消融、预测区间和 RTS-24 随机机组组合说明方法价值。

论文的核心可验证主张可拆为三层：

| 层级 | 原始主张 | 本地复现状态 |
|---|---|---|
| 架构 | 多尺度条件编码与 AdaLN 能更好利用 NWP | 已实现；CE/AdaLN 的方向作用部分得到支持 |
| 概率质量 | 六项指标整体优于基线，区间紧且可靠 | 未复现；CRPS/ES/VS 明显偏高，coverage 严重不足 |
| 决策价值 | 在 RTS-24 SUC 中成本和失负荷最低 | 无法数值复现；论文协议缺失，本地公平重构中 MS-CADM 排名靠后 |

# 2. 数据、条件输入与实验样本口径

## 2.1 GEFCom2014 数据

本地数据目录包含 10 个区域的训练 NWP/风功率文件、10 个区域的测试 NWP 文件以及测试目标文件。合并后每个区域有 731 个完整 calendar days、每天 24 个小时，总计 17544 个小时/区。

预处理流程为：

1. 合并官方 train 与 test 文件；
2. 按区域和时间排序，对目标缺失执行前向填充；
3. 由 `U10、V10、U100、V100` 派生风速、风能和风向；
4. 将小时数据整理为 `[zone-day, 24, feature]`；
5. 只用训练集拟合条件和目标标准化器；
6. 生成场景后逆标准化，并裁剪到 `[0,1]`。

本地派生公式为：`WS=√(U²+V²)`，`WE=0.5·WS³`，`WD=atan2(U,V)·180/π`。风向中的 `atan2(U,V)` 顺序继承自 Dumas 的可执行参考代码，属于需要显式记录的非通常写法。

## 2.2 条件张量

论文 Eq. (13)列出 10 个 NWP 分量和 10 维区域 one-hot。论文列出的前四项顺序为 `u10,u100,v10,v100`，本地遵循 Dumas 代码的 `U10,V10,U100,V100`。本地每个小时 token 包含：

- 10 个标准化 NWP 特征：`U10,V10,U100,V100,WS10,WS100,WE10,WE100,WD10,WD100`；
- 10 个不标准化、逐小时重复的区域 one-hot。

因此单样本条件为 `[24,20]`，训练批次为 `[B,24,20]`；目标为逐小时标准化后的 `[B,24,1]`。用于 VAE、WGAN 和 NF 的平坦条件则按 feature-major 方式形成 `10×24+10=250` 维向量。

## 2.3 日期划分与池化含义

论文只说沿用 Dumas et al. 的数据划分，没有给出数量、日期或随机种子。本地把该引用解释为 `random_state=0` 的两阶段随机日期划分：

| Split | Calendar days | Pooled zone-days | 条件形状 | 目标形状 |
|---|---:|---:|---|---|
| Train | 631 | 6310 | `[6310,24,20]` | `[6310,24]` |
| Validation | 50 | 500 | `[500,24,20]` | `[500,24]` |
| Test | 50 | 500 | `[500,24,20]` | `[500,24]` |

十个区域共用同一组 calendar-day 划分。需要特别强调：全区域 MS-CADM 是以 zone one-hot 区分区域的 **pooled zone-day 模型**。正式测试档案形状 `[500,100,24]` 表示 `50 天×10 区×100 场景×24 小时`，不是 `[50,100,10,24]` 的十区域联合场景，因此不能用它证明跨区域空间相关性建模能力。

# 3. 条件扩散的数学原理

## 3.1 前向扩散

扩散模型先定义一个固定的马尔可夫加噪过程。令 `βₜ∈(0,1)` 为第 `t` 步噪声强度，`αₜ=1−βₜ`，`ᾱₜ=∏ₛ₌₁ᵗαₛ`，则：

> q(xₜ | xₜ₋₁) = N(√αₜ xₜ₋₁, βₜI)

利用高斯闭式性质，可不经过前面所有步骤，直接从真实轨迹 `x₀` 构造任意噪声状态：

> xₜ = √ᾱₜ x₀ + √(1−ᾱₜ) ε，ε ~ N(0,I)

训练时对每个样本均匀采样一个 `t`，再采样同形状高斯噪声 `ε`。这样每次优化只需一次网络前向，而不必真的执行 250 次加噪。

论文 Eq. (3)后把 `αₜ` 直接定义成累计乘积，Eq. (8)又把 `αₜ` 定义为 `1−βₜ` 并另设 `ᾱₜ` 为累计量，符号自相矛盾。本地采用上述 DDPM 标准记号，以保证所有后验系数一致。

## 3.2 反向后验与噪声预测

给定 `x₀` 和 `xₜ`，前向过程的真实后验仍为高斯：

> q(xₜ₋₁ | xₜ,x₀) = N(μ̃ₜ(xₜ,x₀), β̃ₜI)

生成时真实 `x₀` 不可见，因此网络预测噪声 `εθ(xₜ,c,t)`，先反推出 `x̂₀`，再计算后验均值。反向模型写为：

> pθ(xₜ₋₁ | xₜ,c) = N(μθ(xₜ,c,t), Σθ(xₜ,c,t))

每一步从当前 `xₜ` 递推到 `xₜ₋₁`，最终得到条件场景 `x₀`。外部条件 `c` 决定当前日和区域对应的分布，扩散时间步 `t` 决定当前噪声尺度。

## 3.3 训练损失

论文给出噪声 MSE `L_simple` 和后验 KL/VLB，但没有给出二者的最终权重。复现采用 Improved DDPM 风格目标：

> L_total = mean[(εθ−ε)²] + 10⁻³·L_VLB

`t=0` 时 VLB 使用高斯负对数似然，其余时间步使用真实后验与模型后验之间的 KL，并除以 `ln 2` 以 bits 计。计算 VLB 时对均值/噪声路径 stop-gradient，使 VLB 主要训练方差参数；共享 trunk 仍会收到来自方差头的梯度。

## 3.4 为什么必须纠正论文 Algorithm 2

论文第 5 页 Algorithm 2 按字面在每一个反向步重新采样独立 `xₜ~N(0,I)`，这会覆盖上一步得到的状态，使反向马尔可夫链断裂；伪代码还省略了 `Σθ` 和随机项，与论文强调 learned variance 的 Eq. (9)冲突。本地实现选择一次采样 `x_T`，之后严格执行 `xₜ→xₜ₋₁` 递推，并在 ancestral 路径中使用学习方差。

[[CUSTOM_FIGURE:diffusion_workflow|图 1  本地训练与生成流程。该流程保留论文的条件噪声预测与学习方差思想，同时纠正 Algorithm 2 的反向链断裂问题。]]

# 4. MS-CADM 网络架构与张量流

## 4.1 总体结构

论文 Fig. 3 可概括为“多尺度条件编码器 + 条件 Diffusion Transformer”。NWP 条件经过 CE 得到与时间 token 对齐的 latent；noisy wind 经过 Patchify；两者逐元素相加后进入 `N` 个 AdaLN block；扩散时间步单独编码，只生成每个 block 的缩放、平移和残差门控参数；末端输出噪声与方差。

论文只画出了概念结构，没有给出 `N`、hidden dimension、attention heads、FFN expansion、patch size、位置编码和初始化。本地将这些缺失项固定为一套可审计配置。

[[CUSTOM_FIGURE:architecture|图 2  本地 MS-CADM 的实际张量流和复现锁定值。图中数值来自 paper.json 与可执行模型；底部同时标明论文披露与本地补全的边界。]]

## 4.2 多尺度条件嵌入 CE

本地 `MultiScaleEmbedding` 的精确流程如下：

| 阶段 | 运算 | 输入形状 | 输出形状 |
|---|---|---|---|
| 输入投影 | Conv1D, kernel=1, 20→128 | `[B,20,24]` | `[B,128,24]` |
| 通道压缩 | Conv1D, kernel=1, 128→64 | `[B,128,24]` | `[B,64,24]` |
| 分支 1 | Conv3→SiLU→Conv3 | `[B,64,24]` | `[B,64,24]` |
| 分支 2 | Conv5→SiLU→Conv5 | `[B,64,24]` | `[B,64,24]` |
| 分支 3 | Conv7→SiLU→Conv7 | `[B,64,24]` | `[B,64,24]` |
| 融合 | concat→Conv1D 192→64→SiLU | `[B,192,24]` | `[B,64,24]` |
| 恢复与残差 | Conv1D 64→128 + initial | `[B,64,24]` | `[B,128,24]` |
| 归一化 | transpose + LayerNorm | `[B,128,24]` | `[B,24,128]` |

两层同核卷积的有效时间感受野分别约为 5、9、13 小时。论文文字称“层级、由粗到细”和“element-wise integration”，本地实际使用三个**平行**分支和 `concat + learned 1×1 merge`；残差连接与 LayerNorm 也是论文未披露的补全。因此结构思想对齐，但算子并非由原文唯一决定。

## 4.3 Noisy target token 与位置编码

论文 Fig. 3 画出 Patchify，但没有给出 patch size。本地将每个小时视为一个 token：`Linear(1,128)` 把 `[B,24,1]` 映射为 `[B,24,128]`，相当于 patch size=1。模型再加入一个可学习绝对位置参数 `[1,24,128]`。条件 latent、noisy target token 和位置嵌入三者直接相加。

这种解释保留了 24 小时分辨率和 self-attention 的全日依赖，但不能声称与作者未公开的 Patchify 完全相同。

## 4.4 时间步嵌入

整数时间步 `t∈{0,…,249}` 先进入 128 维正余弦嵌入，再经过 `Linear 128→512、SiLU、Linear 512→128`。得到的 `[B,128]` 时间向量不会直接拼到 NWP 条件，而是分别进入每个 AdaLN block 的调制 MLP。

这种设计把条件内容和噪声阶段分工开：NWP/zone 提供“这一天可能是什么轨迹”，时间步提供“当前去噪到了什么强度”。这正是论文所谓 decouple timestep condition from external condition 的核心。

## 4.5 AdaLN Transformer block

本地使用 4 个相同 block。每个 block 的输入和输出均为 `[B,24,128]`，没有 causal mask，也没有 cross-attention；self-attention 直接在 24 个小时 token 上建模全天依赖。每层包括：

1. 无 affine 的 LayerNorm；
2. 时间向量生成第一组 `shift₁、scale₁、gate₁`；
3. `4-head` self-attention，每头维度 32；
4. `gate₁` 控制 attention 残差；
5. 第二个无 affine LayerNorm；
6. 时间向量生成 `shift₂、scale₂、gate₂`；
7. FFN `128→512→128`，激活为 GELU，dropout=0；
8. `gate₂` 控制 FFN 残差。

可写为：

> h ← h + g₁·Attn((1+s₁)·LN(h)+b₁)

> h ← h + g₂·FFN((1+s₂)·LN(h)+b₂)

每个 block 的调制末层全零初始化，使初始 `shift/scale/gate=0`、block 近似 identity；最终双输出头也全零初始化。这是 DiT/AdaLN-Zero 风格的合理稳定化补全，论文并未披露。

## 4.6 噪声与方差双头

4 个 block 后经过 LayerNorm 和 `Linear(128,2)`，输出 `[B,24,2]`，切分为：

- `epsilon:[B,24,1]`：预测加到真实轨迹上的噪声；
- `raw_variance:[B,24,1]`：逐小时对角方差的插值参数。

方差采用 learned-range 参数化：

> v=(tanh(raw_variance)+1)/2

> log σ²θ = v·log βₜ + (1−v)·log β̃ₜ

这样预测方差被限制在真实后验方差和前向 `βₜ` 所定义的合理区间内，避免直接预测负方差。论文只说“学习方差”，没有给出此参数化。

## 4.7 参数量

完整模型有 **1,478,018** 个可训练参数，检查点 state tensor 计数与即时构造模型完全一致。

| 模块 | 参数量 | 占比 |
|---|---:|---:|
| Multi-scale condition embedding | 155,136 | 10.50% |
| Noisy-target input projection | 256 | 0.02% |
| Learned position | 3,072 | 0.21% |
| Timestep embedding MLP | 131,712 | 8.91% |
| 4 × AdaLN Transformer blocks | 1,187,328 | 80.33% |
| Final LayerNorm | 256 | 0.02% |
| Dual output head | 258 | 0.02% |
| Total | 1,478,018 | 100.00% |

每个 AdaLN block 有 296,832 个参数，其中 attention 66,048、FFN 131,712、时间调制 MLP 99,072。可见主干参数主要用于时间调制的全日 self-attention，而不是多尺度条件卷积。

# 5. 论文模块到代码实现的逐项映射

## 5.1 运行时真实模块

仓库同时存在 `repro/diffusion.py` 与 `repro/diffusion/`、`repro/sampling.py` 与 `repro/sampling/`。Python 实际导入包目录：`repro/diffusion/__init__.py` 再导出 `repro/diffusion_core.py`，采样入口为 `repro/sampling/__init__.py`。同名影子文件内容接近，但审计和论文引用应以运行时包路径为准。

## 5.2 Paper-to-code mapping

| 论文组件 | 本地实现 | 对齐内容 | 主要补全/偏离 |
|---|---|---|---|
| Eq.13 NWP+Zone | `repro/data.py` | 10 NWP + 10 one-hot | 特征顺序、标准化和 `[24,20]` 组织由本地锁定 |
| Multi-scale CE | `repro/models/mscadm.py` | 3/5/7 卷积、上下投影 | 平行双卷积、concat 融合、残差、LN 均未披露 |
| Patchify | `MSCADM.input_embedding` | noisy wind 转 token | 本地 patch size=1、Linear(1→128) |
| Position | `MSCADM.position` | 24 小时顺序 | learned absolute position 为本地补全 |
| Timestep embed | `SinusoidalEmbedding` + MLP | 时间步单独编码 | 128/512 维和频率规则未披露 |
| AdaLN attention | `AdaLNBlock` | scale/shift/gate、attention+FFN | depth4、heads4、FF×4、zero-init 未披露 |
| ε/Σ 双头 | `final_norm` + `output` | 噪声和方差两部分 | learned-range 公式及逐小时对角形式未披露 |
| Forward/loss | `repro/diffusion_core.py` | 高斯扩散、MSE+VLB | cosine schedule、T=250、λ=10⁻³、stop-grad 为补全 |
| RCM/SCM | `repro/training.py` | 训练期丢弃条件 | headline 为整样本遮蔽，不是论文文字暗示的逐元素矩阵 |
| Scenario generation | `repro/sampling/__init__.py` | 从高斯噪声生成 100 场景 | 本地纠正断链伪代码并明确 ancestral/DDIM |

## 5.3 关键张量流

| 节点 | 张量形状 | 含义 |
|---|---|---|
| `condition` | `[B,24,20]` | 标准化 NWP + repeated zone one-hot |
| `clean/noisy/noise` | `[B,24,1]` | 标准化风功率及扩散噪声 |
| `condition_latent` | `[B,24,128]` | 多尺度条件表示 |
| `target_token` | `[B,24,128]` | 每小时 noisy wind token |
| `time_embedding` | `[B,128]` | 扩散阶段表示 |
| `hidden` | `[B,24,128]` | 4 层 attention 的主状态 |
| `epsilon/variance` | `[B,24,1]` | 噪声与逐小时方差参数 |
| 一个 generation batch | `[8,100,24]` | 8 个 zone-day、每个 100 场景 |
| 完整测试档案 | `[500,100,24]` | 50 日×10 区的 pooled 场景 |

# 6. 训练、学习方差、RCM 与采样协议

## 6.1 主训练配置

论文明确披露 PyTorch 2.1.1、NVIDIA TITAN、Adam、固定学习率 `1e-4`、26,000 步、batch 256、每个测试样本 100 场景。其余参数由本地配置锁定：

| 类别 | 本地正式值 | 论文是否披露 |
|---|---|---|
| Model dim / bottleneck | 128 / 64 | 否 |
| Depth / attention heads | 4 / 4 | 否 |
| FF expansion / dropout | 4 / 0 | 否 |
| Diffusion T / schedule | 250 / cosine, offset 0.008 | 否；只可从步数表推知最大250 |
| Learned variance | true | 只披露概念 |
| VLB weight | 10⁻³ | 否 |
| Optimizer | Adam, default betas | Adam 是；betas 否 |
| Learning rate | 10⁻⁴ fixed | 是 |
| Weight decay | 0 | 否 |
| Updates / batch size | 26,000 / 256 | 是 |
| Gradient clip | 1.0 | 否 |
| Condition mask probability | 0.1 | 否 |
| Checkpoint/log interval | 1000 / 100 | 否 |
| Seed | 0 | 否 |

训练使用单 GPU float32，没有 AMP、梯度累积、多 GPU或学习率调度。正式环境清单记录 Python 3.11.15、PyTorch 2.6.0+cu124、CUDA 12.4 和 RTX 2060。训练 history 含 260 条记录，最终 step=26000，累计时间 2293.46 秒，约 38 分 13 秒；训练固定取最终步，没有 validation early stopping。

[[CUSTOM_FIGURE:training_curve|图 3  正式 MS-CADM 训练损失轨迹。优化损失稳定下降，但训练损失收敛本身不能证明生成分布的 coverage 或 proper score 已校准。]]

## 6.2 RCM/SCM 的真实行为

论文第 4 页写二值 mask matrix `m`，并说对应 condition dimension 被置零，语义更接近逐元素掩码；但未披露 `p`、掩码粒度和 `p` 是保留率还是遮蔽率。

headline trainer 生成 `[B,1,1]` mask，概率 0.1。一旦某个样本命中，整条 `[24,20]` 条件全部清零；因此它实际是 sample-wise whole-condition dropout，类似 classifier-free condition dropout，而不是 element-wise mask。推理阶段没有 cond/uncond guidance，遮蔽只承担训练正则作用。

另一个 `FeatureMaskTrainer` 才使用 `[B,24,20]` 的逐元素随机 mask，更贴近论文文字。但该诊断分支同时把 cosine schedule 改为 linear，并使用 DDIM50、eta=0，因此其改善不能单独归因于 mask 粒度。

## 6.3 learned variance 在两种采样器中的作用

250-step ancestral sampler 每一步使用网络预测的 `log σ²θ` 加噪，因此直接使用 learned variance。相反，DDIM sampler 调用模型后只保留 `epsilon`，完全丢弃方差头；其随机性由 `eta` 和 alpha 日程决定。

这带来一个重要审计结论：本地 Table 4 的 Zone 1 消融全部使用 50-step DDIM、eta=1，`w/o LV` 并没有直接比较 learned variance 的采样作用，只比较 VLB/共享表示的间接训练影响。因此本地 learned-variance 消融与论文“方差改善后验表达”的机制并不完全同构。

## 6.4 必须区分的正式场景档案

| 档案 | 采样器 | 步数 | eta | 是否使用 variance head | 用途 |
|---|---|---:|---:|---|---|
| `mscadm_test_250steps.npz` | ancestral | 250 | — | 是 | Table 1 headline |
| `mscadm_test_50steps.npz` | DDIM | 50 | 0 | 否 | 实用诊断 |
| `mscadm_test_50steps_eta1.npz` | DDIM | 50 | 1 | 否 | 随机 DDIM 诊断 |
| `sampling_steps/mscadm_*steps.npz` | DDIM | 10–250 | 1 | 否 | Table 3 step sweep |
| `mscadm_featuremask_linear_test.npz` | DDIM | 50 | 0 | 否 | mask+schedule 诊断 |

Table 1 headline 的 250-step ancestral CRPS=0.107400；step-sweep 的 250-step DDIM、eta=1 CRPS=0.107981。二者是不同的正式档案，不能混用。

# 7. 评价指标和 baseline 重构

## 7.1 六个主指标

所有主指标均越低越好，场景和真值已回到 `[0,1]` 标幺空间。

- MAE、RMSE：以 100 条场景的均值作为点预测，在全部 zone-hour 上求误差。
- CRPS：经验分布的 observation distance 减去一半的场景间 pairwise distance，逐小时平均。
- QS：对 1%–99% 共 99 个分位数的 pinball loss 取平均。
- ES：把一天看作 24 维向量，在欧氏距离上计算 multivariate energy score。
- VS：order=0.5，对全部小时对求和，衡量预测与真实时间差分结构。

区间审计使用中央预测区间。ACE 实现为 `coverage−nominal`，不是绝对值；负数表示欠覆盖。PIAW 使用正确的 `upper−lower`，纠正论文 Eq. (22)写成 `lower−upper` 的符号错误。

## 7.2 baseline 的工程重构

论文没有公开 baseline 的结构、训练预算和调参过程，因此以下模型是为了形成公平可执行对照而做的显式重构，并非作者原代码：

| Baseline | 本地实现 | 关键边界 |
|---|---|---|
| RAND | 从评价集观测重采样 | 能对齐论文，但存在 test-observation leakage；另报告 train-only RAND |
| QRGBM | 99 quantiles、300 trees + residual-rank ECC | 24 个小时独立 XGBoost checkpoint |
| WGAN-reference | Dumas 风格 WGAN-GP，约3000 generator updates | 正式 WGAN 只归属于 reference 分支 |
| VAE-reference | Dumas 非标准重参数行为，约2000 updates | 用于复现公开参考行为 |
| NF | conditional RealNVP，8 个 flow steps | 论文未公开 NF 结构 |
| DDPM | 公共能源 WaveNet 式 denoiser，250 steps | 作为更强扩散基线 |

额外的标准 26k-update WGAN control 只有 1000-step `latest.pt`，没有 `final.pt`、场景或正式评分，不能列为完成的正式 baseline。

## 7.3 单种子与统计边界

baseline 的训练和采样均为 seed 0 单次运行，没有多随机种子、置信区间或显著性检验。测试 500 个 zone-day 共享 50 个 calendar days，不能把 500 条轨迹当作完全独立日期。主表可用于工程对表和模型诊断，但不能由小数点差异推导稳定优越性。

# 8. 复现结果

## 8.1 Table 1：全区域 headline

论文报告值如下：

| Model | MAE | RMSE | CRPS | QS | ES | VS |
|---|---:|---:|---:|---:|---:|---:|
| RAND | 0.2583 | 0.3012 | 0.1692 | 0.0855 | 0.9615 | 23.21 |
| QRGBM | 0.1314 | 0.1759 | 0.1036 | 0.0524 | 0.6255 | 20.09 |
| WGAN | 0.1331 | 0.1789 | 0.0979 | 0.0495 | 0.6052 | 19.87 |
| VAE | 0.1244 | 0.1677 | 0.0880 | 0.0445 | 0.5482 | 17.87 |
| NF | 0.1267 | 0.1753 | 0.0907 | 0.0458 | 0.5671 | 18.54 |
| DDPM | 0.1281 | 0.1815 | 0.0981 | 0.0486 | 0.5985 | 19.61 |
| MS-CADM | 0.1191 | 0.1645 | 0.0873 | 0.0441 | 0.5380 | 18.14 |

本地正式结果如下：

| Model | MAE | RMSE | CRPS | QS | ES | VS |
|---|---:|---:|---:|---:|---:|---:|
| RAND evaluation-split | 0.258707 | 0.300767 | 0.168710 | 0.085216 | 0.959265 | 23.1721 |
| QRGBM + ECC | 0.122651 | 0.165515 | 0.085746 | 0.043311 | 0.537024 | 17.1205 |
| WGAN-reference | 0.129290 | 0.177066 | 0.096574 | 0.048859 | 0.592980 | 18.5964 |
| VAE-reference | 0.123630 | 0.167013 | 0.087727 | 0.044337 | 0.545616 | 17.7175 |
| Conditional RealNVP | 0.117057 | 0.164107 | 0.085050 | 0.043012 | 0.532192 | 16.9482 |
| WaveNet DDPM | 0.120106 | 0.160717 | 0.080505 | 0.040624 | 0.506541 | 15.8709 |
| MS-CADM ancestral-250 | 0.124248 | 0.177632 | 0.107400 | 0.054139 | 0.681950 | 22.0200 |

MS-CADM headline 的直接差异为：

| 指标 | 论文 | 本地 | 绝对差 | 相对变化 |
|---|---:|---:|---:|---:|
| MAE | 0.1191 | 0.124248 | +0.005148 | +4.32% |
| RMSE | 0.1645 | 0.177632 | +0.013132 | +7.98% |
| CRPS | 0.0873 | 0.107400 | +0.020100 | +23.02% |
| QS | 0.0441 | 0.054139 | +0.010039 | +22.76% |
| ES | 0.5380 | 0.681950 | +0.143950 | +26.76% |
| VS | 18.14 | 22.0200 | +3.8800 | +21.39% |

[[CUSTOM_FIGURE:reproduction_scorecard|图 4  MS-CADM headline 本地/论文比值。点预测误差略有退化，概率分数的退化更集中，说明问题主要不是平均轨迹，而是生成分布。]]

RAND 几乎精确对齐论文，VAE-reference 也非常接近，说明日期划分、逆标准化和评分公式整体具有较高可信度；但 RAND 从评价观测中重采样，只能用于表格对齐，不能当无泄漏科学基线。更强的 QRGBM、RealNVP 和 DDPM 重构改变了论文的相对排名，也说明原文 baseline 细节缺失会直接影响“最优方法”结论。

## 8.2 区间覆盖与欠离散

headline MS-CADM 的名义 90% 中央区间只覆盖 35.95% 的 12,000 个 test zone-hour cells，PIAW 仅 0.120447，ACE=-0.5405。区间很窄但大量漏掉真实值，这是典型的系统性欠离散，而不是少数极端日造成的偶发误差。

| 档案 | Coverage90 | ACE90 | PIAW90 |
|---|---:|---:|---:|
| DDPM ancestral-250 | 0.866667 | −0.033333 | 0.487636 |
| MS feature-wise + linear | 0.495833 | −0.404167 | 0.219794 |
| MS headline ancestral-250 | 0.359500 | −0.540500 | 0.120447 |
| MS DDIM-50 eta=0 | 0.371167 | −0.528833 | 0.135577 |
| MS DDIM-50 eta=1 | 0.307167 | −0.592833 | 0.104102 |
| NF | 0.779083 | −0.120917 | 0.352126 |
| QRGBM | 0.759000 | −0.141000 | 0.414204 |
| VAE-reference | 0.819750 | −0.080250 | 0.409094 |

feature-wise mask + linear schedule 分支把 coverage 提高 13.63 个百分点，但同时改变了 mask、beta schedule 和 eta，不能把全部改善归因于单一模块；而且 49.58% 仍远低于名义 90%。

[[FIGURE:outputs/full_reproduction/figures/figure4_mscadm_test_250steps.png|图 5  headline MS-CADM 的一个测试 zone-day 示例。红色观测多次落在第10/90分位带之外；图中两条分位线构成中央80%区间，不应误称为90%区间。]]

[[FIGURE:outputs/full_reproduction/figures/figure5_intervals.png|图 6  不同方法在名义 coverage 0.1–0.9 下的 PIAW 和带符号 ACE。MS-CADM 曲线长期显著低于零，显示全覆盖水平的系统性欠覆盖。]]

## 8.3 Table 2：独立单区域复现

论文未披露所选区域。根据 RAND 点指标的接近程度，本地把 Zone 1 作为后验推断并独立训练，但 RAND 的 ES/VS 并未同时对齐，因此不能认定论文一定使用 Zone 1。

| Model | MAE | RMSE | CRPS | QS | ES | VS |
|---|---:|---:|---:|---:|---:|---:|
| Zone1 RAND | 0.257533 | 0.299320 | 0.165584 | 0.083633 | 0.936493 | 20.4195 |
| Zone1 QRGBM | 0.126816 | 0.161964 | 0.090395 | 0.045698 | 0.540241 | 16.0524 |
| Zone1 WGAN-reference | 0.149919 | 0.204118 | 0.130360 | 0.065689 | 0.788364 | 22.6724 |
| Zone1 VAE-reference | 0.132324 | 0.177216 | 0.104507 | 0.052842 | 0.650294 | 20.4224 |
| Zone1 NF | 0.134915 | 0.188432 | 0.108969 | 0.055049 | 0.674355 | 19.8047 |
| Zone1 DDPM | 0.131425 | 0.176913 | 0.094129 | 0.047645 | 0.583121 | 16.2641 |
| Zone1 MS-CADM | 0.118892 | 0.167177 | 0.114263 | 0.057272 | 0.726970 | 21.7558 |

论文 MS-CADM 单区值为 `0.1053/0.1448/0.0785/0.0397/0.4703/15.96`。本地相对论文在 MAE/RMSE/CRPS/QS/ES/VS 上分别恶化约 12.91%/15.45%/45.56%/44.26%/54.58%/36.32%。本地 MS-CADM 的 MAE 最优，但 CRPS 排第 5、VS 排第 6，不能概括为单区总体最优。

## 8.4 Table 3：去噪步数

五档实验都使用同一个 26k-update 全区域检查点、DDIM、eta=1、seed 0，生成完整 `[500,100,24]` 测试档案。

| Steps | Seconds | Relative to 50 | MAE | RMSE | CRPS | ES | VS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 15.019 | 0.211× | 0.124455 | 0.177774 | 0.111599 | 0.705703 | 22.7253 |
| 20 | 28.392 | 0.400× | 0.124007 | 0.177204 | 0.110453 | 0.700112 | 22.5752 |
| 50 | 71.037 | 1.000× | 0.124168 | 0.177555 | 0.109229 | 0.693248 | 22.4018 |
| 100 | 142.111 | 2.000× | 0.124161 | 0.177580 | 0.108513 | 0.688910 | 22.2403 |
| 250 | 351.383 | 4.947× | 0.124343 | 0.177865 | 0.107981 | 0.685780 | 22.1568 |

从 50 增到 250 步，墙钟变为 4.947 倍，而 CRPS、ES、VS 只改善约 1.14%、1.08%、1.09%。这在方向上复现了论文“50 步以后边际收益很小”的判断；但绝对分数没有复现。计时是 RTX 2060 的单次完整档案 wall-clock，没有重复和误差条，不能跨硬件比较。

## 8.5 Table 4：组件消融

本地消融均为 Zone 1 独立训练、26k updates、seed 0、50-step DDIM、eta=1。

| Variant | MAE | RMSE | CRPS | QS | ES | VS | CRPS vs full |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full | 0.118892 | 0.167177 | 0.114263 | 0.057272 | 0.726970 | 21.7558 | — |
| w/o CE | 0.124045 | 0.169852 | 0.119012 | 0.059657 | 0.733028 | 21.3654 | +4.16% |
| w/o AdaLN | 0.123661 | 0.174296 | 0.119210 | 0.059744 | 0.754289 | 21.8743 | +4.33% |
| w/o LV | 0.118792 | 0.167099 | 0.114156 | 0.057219 | 0.726341 | 21.7355 | −0.09% |
| w/o RCM | 0.123796 | 0.172589 | 0.119372 | 0.059826 | 0.740274 | 21.8043 | +4.47% |

可支持的结论是：CE、AdaLN 和 RCM 对多数指标有约 4% 的方向性贡献，但 CE 的 VS 反而略差；`w/o LV` 六项都极小幅优于 full，没有复现论文所声称的 learned variance 明显收益。由于 DDIM 忽略方差头，这个 LV 消融本身也不能直接检验方差采样贡献。单种子无显著性检验，不能把约 4% 排序表述成稳定因果效应。

## 8.6 边界分布

GEFCom2014 在 0 和 1 附近有明显边界质量。直方图显示不同模型对边界和内部连续分布的处理差异；MS-CADM 的总体直方图可能接近观测，但这不等价于条件分布和逐日 coverage 正确，因为全样本边际分布可掩盖条件欠离散。

[[FIGURE:outputs/full_reproduction/figures/figure7_distribution.png|图 7  测试集观测与生成值的总体边际直方图。该图只能描述全局边际形状，不能替代逐日、逐区条件校准或联合依赖检验。]]

## 8.7 Table 5：RTS-24 SUC 重构

论文披露 IEEE RTS-24、12 台机组、3375 MW 火电、6 个风场共 1200 MW、K-Means 压缩、失负荷/弃风罚金 1000/80 USD/MWh，但没有给出 `k`、区域到风场映射、机组/网络数据版本、备用规则、求解器、gap、时限和失败处理。正文先说随机 7 天，下一段又称表 5 是 100 天平均，因此论文 Table 5 无法唯一复现。

本地统一使用 7 个 pooled zone-date cases、K=10、1200 MW 风电、相同真值和模型协议，得到：

| Model | Total cost | Penalty cost | Load shedding MWh | Curtailment MWh |
|---|---:|---:|---:|---:|
| VAE-reference | 340,133.86 | 3,991.69 | 0.0000 | 49.8961 |
| DDPM | 353,839.16 | 9,588.99 | 2.0907 | 93.7287 |
| QRGBM | 356,558.87 | 20,326.41 | 15.3590 | 62.0923 |
| NF | 358,583.83 | 21,913.09 | 16.7785 | 64.1818 |
| MS-CADM | 395,386.19 | 60,626.54 | 54.1654 | 80.7637 |
| WGAN-reference | 404,085.87 | 68,459.07 | 66.5294 | 24.1212 |

本地每个 case 取一条单区归一化轨迹乘 1200 MW，再平均分到六个风场，不是十区域联合映射；7 个 zone-date cases 只有 5 个唯一日期。规划为 binary MILP reconstruction，默认 1% gap、120 秒，但 CSV 没有保存 status、dual bound、实际 gap 或 runtime。因此这只能叫公平的描述性重构，不能叫带完整最优性证书的 exact SUC，也不能与论文 `10⁷ USD` 数值直接对表。MS-CADM 本地排第 5，与论文最优结论不一致。

# 9. 未复现原因和原文可复现性漏洞

## 9.1 为什么可以排除“只是指标代码全面错误”

RAND 六项几乎精确匹配论文，VAE-reference 也非常接近；测试场景均经过同一逆标准化与评分函数。这说明数据日期划分、目标尺度和主评分公式整体可信。偏差集中在 MS-CADM、单区模型和部分未披露 baseline，最可能来自模型/协议不可识别，而不是一个统一的指标缩放错误。

## 9.2 阻断逐位复现的缺失项

- Transformer 深度、宽度、attention heads、FFN expansion、patch size、位置编码未披露。
- 多尺度分支通道、每支层数、激活、padding、融合方式、残差和 normalization 未披露。
- 扩散训练步数、beta schedule、缩步采样时间索引和 DDPM/DDIM/eta 未披露。
- learned variance 的参数化、VLB 权重、t=0 项和 stop-gradient 未披露。
- RCM/SCM 命名、概率、遮蔽/保留语义和掩码粒度未披露。
- 数据标准化、缺失值、checkpoint 选择、随机种子和重复次数未披露。
- VAE、GAN/WGAN、NF、QRGBM、DDPM 的架构与调参预算未披露。

## 9.3 公式和算法冲突

1. Eq. (3)与 Eq. (8)对 `αₜ` 的定义冲突。
2. Algorithm 1 虽说计算 Eq. (11)+Eq. (12)，参数更新却只写 `∇L_simple`，按字面方差头没有 KL 梯度。
3. Algorithm 2 每个反向步重新采样 `xₜ`，无法形成递推链。
4. Algorithm 2 的均值公式与 Eq. (8)不同，并完全省略 `Σθ` 和随机项。
5. 论文推荐 50 步，却没有定义如何从训练链抽取 50 个时间索引。

## 9.4 指标与图表问题

- PIAW Eq. (22)把上下界顺序写反，会得到负宽度。
- ACE Eq. (20)的求和指标和分母符号不一致，且省略置信水平参数。
- Fig. 5 的负 ACE 若按 `coverage−nominal` 解释，并不支持正文“95%以下 coverage 稳定在90%–100%”的强表述。
- Table 1 的 MS-CADM 六项与 Table 3 的 250 步列完全相同，但正文最终推荐 50 步。
- Table 1 写 GAN，正文与其他表写 WGAN。
- Table 4 声称“三个核心组件”却列出四项。
- Table 5 标题误抄成组件消融标题。
- Fig. 4/6 图例和图注的颜色、single day/three days 描述不一致。

## 9.5 实验设计漏洞

- RAND 按论文文字从 train、validation 和 test 一起抽样，会使用测试观测，属于评价泄漏。
- 单区域编号未披露。
- 十区域采用 pooled one-hot，不是明确的十区域联合分布。
- 所有结果均无多种子、置信区间或显著性检验。
- SUC 的 7 天/100 天矛盾、`k` 和映射缺失，使决策结果不可唯一重建。

## 9.6 对 headline 失败的审慎诊断

现有证据能够确认的是“本地分布严重欠离散”，不能唯一确认其单一原因。以下机制与结果相关，但彼此存在耦合：sample-wise whole-condition mask 与论文 element-wise 语义可能不同；cosine/linear schedule 未披露；learned variance 与 DDIM 路径脱节；Patchify、CE 融合和维度均由本地补全；单种子可能带来波动；作者 baseline 和 checkpoint 选择未知。feature-wise+linear 分支显著改善 proper score 和 coverage，说明这些工程选择确实重要，但由于同时改变多个因素，不能作单因果归因。

# 10. 工程审计、资产索引与 baseline 使用建议

## 10.1 完成审计

`outputs/full_reproduction/completion_audit.json` 当前 `passed=true`，包含 11 项产物完整性检查：Table 1/2 场景形状、五档采样步数、五个独立消融档案、42 行公平 SUC、QRGBM 小时检查点、表格存在性、PNG 可解码、训练 checkpoint、归档测试日志和 manifest 存在性。

归档 baseline 测试日志为 `22 passed, 2 warnings in 37.35s`；另行复核模型与采样的 4 项关键测试通过。这里的 22 项是 baseline 当时的测试集合，不代表后来增加的整个仓库测试总数。

完成审计的边界也必须说明：checkpoint 检查主要验证数量和非空，图像只验证可解码，SUC fairness 只验证共享 case 索引，旧测试只搜索归档日志字符串；`passed=true` 证明规定产物齐全和基本可读，不证明论文数值正确、统计显著或作者实现被精确还原。

## 10.2 环境与哈希

| Artifact | SHA256 |
|---|---|
| 原始论文 PDF | `7AE050DA384469485422D10F810D959103F1CD2C9E7597BD387C8AB099C729C4` |
| `repro_configs/paper.json` | `00922F2942BCA7F36C5403A251E7A73CF648D9702138E383DD15FFB64808F87D` |
| `mscadm/final.pt` | `7A667FBF576C3BAE07C24D2C968224E5B59FD911229738052E089507EA146027` |
| `mscadm_test_250steps.npz` | `4786409787B9A6E7296093AECF4BE22484DAC7770BE4DA82D64F7BAA2136B466` |

环境 manifest 记录 Python 3.11.15、PyTorch 2.6.0+cu124、CUDA 12.4、RTX 2060；但 `git_commit` 和 `git_status` 为 null，manifest 也没有覆盖原始数据、代码、CSV、图片和测试日志的全部哈希，故尚不是完整的可追溯软件发布包。

## 10.3 主要入口

- 训练 MS-CADM：`python -m repro_scripts.train --config repro_configs/paper.json --model mscadm --device cuda`
- 生成 headline：`python -m repro_scripts.generate --config repro_configs/paper.json --model mscadm --split test --scenarios 100 --steps 250 --sampler ancestral --device cuda`
- 采样步数：`python -m repro_scripts.sampling_steps --config repro_configs/paper.json --device cuda`
- 生成表格：`python -m repro_scripts.tables_cli --root outputs/full_reproduction`
- 完成审计：`python -m repro_scripts.completion_audit --root outputs/full_reproduction`

模型训练和场景生成使用 GPU；指标与统计为 CPU；早期 SUC 使用 SciPy/HiGHS 的 CPU MILP。RTX 2060 的 6GB 显存适合当前 1.48M 参数模型，但生成时仍需通过 `day_batch=8` 将 `8×100` 条轨迹分批处理。

## 10.4 可复用资产

| 类型 | 权威路径 | 用途 |
|---|---|---|
| 论文配置 | `repro_configs/paper.json` | 所有本地补全值的统一定义 |
| 论文报告值 | `repro_configs/paper_reported_results.json` | Table 1–5 对表基准 |
| 数据管线 | `repro/data.py` | GEFCom2014 合并、特征、split、standardizer |
| 网络 | `repro/models/mscadm.py` | CE、AdaLN、双头 |
| 扩散运行时 | `repro/diffusion_core.py` | forward、VLB、ancestral、DDIM |
| 训练 | `repro/training.py` | seed、mask、optimizer、checkpoint |
| 采样运行时 | `repro/sampling/__init__.py` | GPU 场景生成与 metadata |
| 主检查点 | `outputs/full_reproduction/mscadm/final.pt` | 26k-step model |
| 主场景 | `outputs/full_reproduction/scenarios/mscadm_test_250steps.npz` | Table 1 headline |
| 正式表 | `outputs/full_reproduction/tables/` | Table 1–4 与区间 JSON |
| SUC | `outputs/full_reproduction/suc/` | Table 5 本地重构 |
| 完成审计 | `outputs/full_reproduction/completion_audit.json` | 产物齐全性 |
| 环境清单 | `outputs/full_reproduction/experiment_manifest.json` | 环境与部分哈希 |

## 10.5 如何把它作为后续 baseline

这套实现适合用作研究 baseline，因为数据、网络、训练、采样、评分、消融和场景档案都可执行且有明确身份；但论文写作中应使用以下措辞：

> 我们根据论文公开的多尺度条件编码、AdaLN 时间步调制、噪声/方差双头和条件遮蔽机制，并结合标准 Improved-DDPM/DiT 实践，构建了可执行 MS-CADM 重实现。数据与评分口径通过 RAND 和参考行为 VAE 得到交叉验证；然而，重实现未达到原文报告值，并表现出显著欠覆盖。因此后续改进统一相对该受控实现及更强 DDPM 对照评价，而不把它称为经确认的原论文 SOTA。

不应使用的表述包括：“完整逐行复刻作者模型”“数值复现论文最优结果”“十区域联合场景已被 baseline 建模”“SUC 证明 MS-CADM 显著节省成本”。

# 11. 最终评价

## 11.1 复现成功的部分

- 论文的核心方法链可以被清楚实现：多尺度 NWP 编码、条件加法、时间步 AdaLN、noise/variance 双头、条件扩散训练和轨迹生成。
- 数据与评分口径有较强交叉证据，RAND 和 VAE-reference 与论文接近。
- 50 步后的时间—性能边际收益递减得到定性复现。
- CE、AdaLN 和 RCM 在单次 Zone 1 消融中得到多数指标的方向支持。
- 正式产物包含配置、检查点、30 个场景档案、主要表图、旧测试日志和哈希清单。

## 11.2 没有复现的部分

- MS-CADM 全区域和单区的绝对数值、领先幅度及模型排名没有复现。
- 名义 90% 区间可靠性没有复现，headline coverage 仅 0.3595。
- learned variance 的显著正贡献没有复现，本地 DDIM 消融甚至略偏向 no-LV。
- 论文 SUC 的数值和最优排名无法复现；本地协议只能给出描述性重构。
- 十区域联合空间分布不在本 baseline 的输出定义内。

## 11.3 一句话结论

MS-CADM baseline 已经达到“可运行、可审计、可作为后续研究起点”的完整度，但没有达到“论文数值被精确复现”的标准；最重要的复现发现不是原模型再次成为最优，而是它在明确工程解释下呈现严重欠离散，并且其结果对论文未披露的 mask、schedule、sampling 和 baseline 细节高度敏感。

# 附录 A  论文原始 Table 2–5

## A.1 Table 2：论文单区域结果

| Model | MAE | RMSE | CRPS | QS | ES | VS |
|---|---:|---:|---:|---:|---:|---:|
| RAND | 0.2564 | 0.3012 | 0.1677 | 0.0841 | 0.8380 | 19.85 |
| QRGBM | 0.1244 | 0.1575 | 0.1092 | 0.0550 | 0.8236 | 17.85 |
| WGAN | 0.1383 | 0.1707 | 0.1162 | 0.0583 | 0.5708 | 19.53 |
| VAE | 0.1232 | 0.1660 | 0.0948 | 0.0481 | 0.5108 | 17.27 |
| NF | 0.1202 | 0.1688 | 0.0938 | 0.0472 | 0.5176 | 17.53 |
| DDPM | 0.1226 | 0.1740 | 0.1028 | 0.0519 | 0.6259 | 20.74 |
| MS-CADM | 0.1053 | 0.1448 | 0.0785 | 0.0397 | 0.4703 | 15.96 |

## A.2 Table 3：论文去噪步数

| Metric | 10 | 20 | 50 | 100 | 250 |
|---|---:|---:|---:|---:|---:|
| MAE | 0.1198 | 0.1192 | 0.1192 | 0.1190 | 0.1191 |
| RMSE | 0.1649 | 0.1646 | 0.1646 | 0.1645 | 0.1645 |
| CRPS | 0.0884 | 0.0877 | 0.0875 | 0.0874 | 0.0873 |
| QS | 0.0447 | 0.0443 | 0.0442 | 0.0442 | 0.0441 |
| ES | 0.5412 | 0.5388 | 0.5384 | 0.5379 | 0.5380 |
| VS | 18.70 | 18.58 | 18.39 | 18.25 | 18.14 |
| Relative time | 0.20× | 0.52× | 1.00× | 2.03× | 5.09× |

## A.3 Table 4：论文组件消融

| Variant | MAE | RMSE | CRPS | QS | ES | VS |
|---|---:|---:|---:|---:|---:|---:|
| w/o CE | 0.1092 | 0.1532 | 0.0834 | 0.0421 | 0.4984 | 16.69 |
| w/o AdaLN | 0.1352 | 0.1862 | 0.1007 | 0.0509 | 0.6069 | 19.70 |
| w/o LV | 0.1227 | 0.1682 | 0.0901 | 0.0456 | 0.5549 | 18.96 |
| w/o RCM | 0.1075 | 0.1474 | 0.0807 | 0.0408 | 0.4835 | 16.39 |
| MS-CADM | 0.1053 | 0.1448 | 0.0785 | 0.0397 | 0.4703 | 15.96 |

## A.4 Table 5：论文 SUC

| Model | Total cost (10⁷ USD) | Penalty cost (10⁷ USD) | Load shedding MWh | Curtailment MWh |
|---|---:|---:|---:|---:|
| QRGBM | 0.286 | 0.253 | 2319.18 | 2649.10 |
| WGAN | 0.283 | 0.252 | 2379.36 | 1829.52 |
| VAE | 0.291 | 0.260 | 2445.12 | 1990.08 |
| NF | 0.259 | 0.228 | 2151.84 | 1594.80 |
| DDPM | 0.220 | 0.186 | 1664.88 | 2545.20 |
| MS-CADM | 0.185 | 0.153 | 1355.04 | 2158.08 |

# 附录 B  复现假设台账

| 不可识别项 | 本地选择 | 影响 |
|---|---|---|
| Transformer N/width/heads | 4 / 128 / 4 | 决定容量、训练稳定和全日依赖 |
| Patchify | 每小时 patch=1 | 决定 token 数与局部表示 |
| Position encoding | learned absolute `[1,24,128]` | 表示小时顺序 |
| CE channels/merge | 20→128→64；3/5/7 双卷积；concat merge | 决定感受野与条件容量 |
| Diffusion T/schedule | 250 cosine | 决定噪声分配与反向难度 |
| Variance parameterization | learned range | 决定 ancestral 随机性 |
| VLB weight | 10⁻³ | 决定方差训练强度 |
| RCM | p=0.1 whole-condition | 与论文 element-wise 文字存在偏离 |
| Reduced-step sampling | linspace DDIM, eta=1 | 决定 Table 3 速度和分布 |
| Split | Dumas random 631/50/50, seed0 | 决定训练与测试样本 |
| Single zone | Zone 1 后验推断 | 论文未披露区域，不能唯一对表 |
| Baseline architectures | 显式工程重构 | 会改变相对排名 |
| SUC mapping | 单轨迹×1200 MW后均分六风场 | 只是公平代理，不是物理十区映射 |

# 附录 C  术语表

| 术语 | 含义 |
|---|---|
| CE | Conditional Embedding，多尺度条件嵌入 |
| AdaLN | Adaptive Layer Normalization，由时间步生成 scale/shift/gate |
| RCM/SCM | Random/Stochastic Conditional Masking，论文命名不一致 |
| LV | Learned Variance，学习反向高斯方差 |
| DDPM | Denoising Diffusion Probabilistic Model |
| DDIM | 可缩步的隐式扩散采样方法；本地 eta 控制随机性 |
| CRPS | 一维概率分布 proper score |
| ES | 24 维轨迹 energy score |
| VS | 时间对差分结构 variogram score |
| ACE | 本报告为 coverage−nominal 的带符号误差 |
| PIAW | 预测区间平均宽度 |
| Pooled zone-day | 将每个区域每天当独立样本，并用 one-hot 共享模型 |
| Joint scenario | 同一场景编号同时包含多个区域的联合轨迹；本 baseline 不具备 |

# 参考文献

1. Zhang, J., Liu, S., Shi, Z., & Li, Y. Wind power scenario generation via multi-scale condition adaptive diffusion model. Electric Power Systems Research 255 (2026) 112753.
2. Ho, J., Jain, A., & Abbeel, P. Denoising Diffusion Probabilistic Models. NeurIPS 2020.
3. Nichol, A. Q., & Dhariwal, P. Improved Denoising Diffusion Probabilistic Models. ICML 2021.
4. Peebles, W., & Xie, S. Scalable Diffusion Models with Transformers. ICCV 2023.
5. Dumas, J., Wehenkel, A., Lanaspeze, D., Cornélusse, B., & Sutera, A. A deep generative model for probabilistic energy forecasting in power systems: normalizing flows. Applied Energy 305 (2022) 117871.

[[END_REPORT]]
