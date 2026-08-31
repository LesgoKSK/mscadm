# MS-CADM 复现与改进实验综合研究报告

从文献复现、概率校准与联合生成，到安全决策导向场景校准

项目：GEFCom2014 风电概率场景生成与随机机组组合

报告日期：2026-08-09

文档性质：可审计研究总结 / 论文选题决策依据

[[COVER_END]]

## 阅读说明

本报告系统整理了项目从原始文献复现到后续六条改进路线的全部主要工作、结果、失败模式和证据边界。报告中的“确认”均指在预先冻结的、同一 GEFCom2014 数据域内的新划分上进行的内部确认，不等同于跨数据集、跨地区或跨年份的外部验证。

报告坚持四个原则：第一，论文报告值与本地重现值分开；第二，开发性、探索性和确认性证据分开；第三，概率指标与调度指标分开；第四，成功结果、负结果和未完成事项分开。任何没有生成正式产物的消融，不被写成已经完成的成果。

## 报告结构

1. 执行摘要与研究结论
2. 原始文献、任务定义与证据协议
3. MS-CADM baseline 的完整复现
4. CR-MS-CADM：条件残差与经验校准
5. RAHC：秩自适应层级校准
6. CAA-RAHC：原子感知的安全校准
7. MM-JDWind：混合测度联合风电生成
8. STGF-Flow：时空图频域流模型
9. PS-DFSC：Proper-score 约束的决策导向校准
10. 跨实验综合、论文路线与下一步
11. 工程资产、审计状态与复现入口
12. 附录：完整关键表格、开放问题与术语

# 1. 执行摘要

## 1.1 一句话结论

本项目已经完成从“复现一篇扩散场景生成论文”到“建立可审计的联合概率生成、校准与 exact stochastic UC 研究平台”的转变。原论文的相对趋势可以部分复现，但其核心数值和优越性不能被严格重现；后续最稳健的正结果来自 MM-JDWind 的混合测度与质量守恒设计，而 RAHC、STGF-Flow 和 PS-DFSC 的负结果揭示了校准宽度、图频归纳偏置以及决策收益与概率安全之间的真实冲突。

## 1.2 主要成果总览

| 阶段 | 核心问题 | 最重要结果 | 证据等级 | 当前判断 |
|---|---|---|---|---|
| MS-CADM baseline | 原文能否完整重现 | 趋势部分复现，但 CRPS 比论文高 23.02%，90% coverage 仅 0.3595 | 正式复现审计 | 原论文 headline 不可严格重现 |
| CR-MS-CADM | 能否修复严重欠离散 | CRPS 降 22.50%，coverage 升至 0.8346 | 同一旧 test 上探索 | 强动机，不是确认性结论 |
| RAHC | 能否进一步修复条件覆盖 | conditional ACE 降 57.73%，但 CRPS、宽度和 Winkler 恶化；仅 4/11 门槛通过 | 重用 test 的完整探索 | 过度扩张尾部，路线未通过 |
| CAA-RAHC | 原子质量与安全门控能否稳健涨点 | CRPS 改善约 1.11%，9/9 outer-seed 同向；coverage 0.8500 仍未达门槛 | 3×50 日内部确认 | 有效，但贡献主要来自 atom-only |
| MM-JDWind | 能否原生生成十区域联合混合分布 | 相对 no-jump，CRPS 改善 3.01%，coverage 0.7442→0.8453；与 DDPM CRPS 持平 | 新 150 日内部确认 | 当前最强概率建模成果 |
| STGF-Flow | 图频域是否优于时域联合流 | 相对 time-domain，CRPS 恶化 3.99%，6/6 门槛失败；但 16 NFE 可竞争 250-step DDPM | 新 150 日内部确认 | 核心假设被否定，效率结果有价值 |
| PS-DFSC | 能否在 proper-score 硬约束下改善 exact SUC | 12 个候选均未通过安全门槛，三个 outer 全部锁定 identity；确认成本差为 0 | 完整开发与确认协议 | fail-closed 机制有效，尚无决策收益 |

[[CUSTOM_FIGURE:timeline|图 1  研究路线、证据等级与最终状态。绿色代表通过主要研究问题，琥珀色代表有条件正结果，红色代表预注册假设未通过，蓝灰色代表完成复现但不能支持原论文 headline。]]

## 1.3 最值得写进论文的发现

第一，风电边界原子不是边缘细节。CAA-RAHC 中 atom-only 的 A2 几乎解释了 A0→A4 的全部 CRPS 增益；MM-JDWind 中显式混合测度与 absorbing jump 又在新的 150 日上稳定改善 CRPS、coverage、联合 ES 和 zero-atom Brier。两条独立实验链共同说明：把 0/1 边界质量当作连续噪声处理，会系统性损害风电场景质量。

第二，复杂结构不天然带来收益。STGF-Flow 的固定图频表示显著输给更简单的 time-domain 联合流；RAHC 的层级 rank 校准在改善 conditional ACE 的同时扩宽区间并损害 proper score。模型复杂度只有在对应结构假设被数据支持时才有价值。

第三，决策导向训练必须设置可执行的安全边界。PS-DFSC 的代理目标确实给出过 0.51%–2.00% 的预测成本下降和 3.44%–6.44% 的预测 CVaR 下降，但所有对应候选均不安全。严格回退使最终确认结果不退化，但也使决策改善为零。这一结果直接证明“代理成本下降”不能替代 exact MILP 与 proper-score 联合验收。

## 1.4 当前最合理的论文主线

若以一篇高质量方法论文为目标，建议把 MM-JDWind 作为主模型，把 CAA-RAHC 的 atom-only 结果作为结构动机，把 PS-DFSC 作为下一阶段决策层扩展，而不是把所有模型串成一个过度庞大的故事。更稳妥的叙事是：

- 问题：风电联合分布含有边界原子、连续内部密度和跨区域时空依赖；普通连续 diffusion/flow 难以同时处理。
- 方法：混合测度分解 + 质量守恒跳跃机制 + 有界联合连续流。
- 核心证据：相同连续生成器下，mass-preserving 相对 no-jump 在 9/9 outer-seed 上改善；与强 DDPM 的 CRPS/ES 持平，同时 MAE、区间宽度和 zero-event Brier 更好。
- 诚实边界：coverage 仍只有约 0.845；ramp-CRPS 劣于 DDPM；确认仍是同一数据域的内部确认。

# 2. 原始文献、任务定义与证据协议

## 2.1 原始文献

本项目复现对象为 Jiawei Zhang、Shuhao Liu、Zeyi Shi 和 Yuancheng Li 的论文《Wind power scenario generation via multi-scale condition adaptive diffusion model》，发表于 Electric Power Systems Research 255 (2026) 112753，DOI 为 10.1016/j.epsr.2026.112753。

论文提出 Multi-Scale Condition Adaptive Diffusion Model（MS-CADM），主要包含多尺度条件嵌入、AdaLN/时间步调制、学习方差和随机条件遮蔽。论文摘要声称相对已有方法，MAE 降低 4.26%，RMSE 降低约 1.91%；实验采用 GEFCom2014 风电数据，并用随机机组组合展示决策价值。

## 2.2 任务形式化

给定日前的 10 个 NWP 衍生特征及 10 维区域 one-hot 标识 c，目标是学习未来 24 小时风电归一化功率轨迹 y 的条件分布 p(y|c)，并生成 M=100 条场景。正式输入没有显式日历/时钟特征；扩散时间步 t 单独进入 AdaLN，24 小时位置由模型位置嵌入表达。早期 baseline 将 10 个区域池化为 zone-day 样本，因此输出本质上是 [500,100,24] 的单区域条件场景集合；后续 MM-JDWind、STGF-Flow 与 PS-DFSC 才转为 [day,scenario,zone,hour] 的十区域联合建模。

评价覆盖三类目标：

- 点预测质量：场景均值的 MAE、RMSE。
- 概率质量：CRPS、quantile score、energy score、variogram score、coverage、区间宽度、Winkler、conditional ACE、边界原子 Brier、ramp-CRPS。
- 决策质量：早期 baseline/CR/RAHC/CAA 使用 binary MILP reconstruction 做描述性比较；PS-DFSC 才使用保存 incumbent、dual bound、实际 gap 与终止原因的 exact binary two-stage SUC，评价 realized total cost、regret、CVaR、负荷损失、弃风、备用短缺和求解状态。

## 2.3 数据与基础划分

GEFCom2014 风电数据包含 10 个区域，每个区域 731 个完整日，每日 24 小时。预处理合并官方 train/test 文件，对目标缺失执行前向填充，构造 10 个 NWP 衍生特征；标准化统计量只由训练集估计，生成结果逆变换后裁剪到 [0,1]。正式 baseline 按 Dumas 风格协议，以 random_state=0 得到 631/50/50 个 train/validation/test calendar days；池化后分别为 6310/500/500 个 zone-days。

后续实验根据研究问题采用不同的隔离强度。早期 CR 和 RAHC 仍使用同一旧 test，因此只能提供探索性证据。CAA-RAHC 引入 3 个互斥 outer test，每个 50 日。MM-JDWind 与 STGF-Flow 又使用与 CAA 不重叠的新 150 日确认集。PS-DFSC 将旧 150 日降级为开发数据，并从剩余 181 日中冻结 3×50 日 outer test，另有 31 日不用于确认评价。

| 实验 | 训练/开发/测试结构 | 是否读取旧 test | 证据标签 |
|---|---|---|---|
| baseline | 631/50/50 日 | 原始协议 | 正式复现 |
| CR-MS-CADM | 训练与验证后在原 50 test 日评价 | 是，且假设受其诊断启发 | 完整探索 |
| RAHC | 沿用 CR 的原 50 test 日 | 是 | 完整探索 |
| CAA-RAHC | 3 组 sealed outer，每组 50 日 | 旧 test 仅作校准 | 内部确认 |
| MM-JDWind | 新的 3×50 日确认集 | 与 CAA test 不重叠 | 内部确认 |
| STGF-Flow | 另一套锁定流程，3×50 日 | 与 CAA/MM 开发隔离 | 内部确认 |
| PS-DFSC | dev 100/50；outer 3×50；unused 31 | 旧 150 日仅作开发 | 内部确认与安全回退 |

## 2.4 证据等级的解释

“由于某方法假设受此前同一 test 划分诊断启发，因此结果属于探索性实验”具体表示：研究者先看到了 test 上的欠离散或覆盖不足，再据此设计方法，最后仍在同一 test 上报告改善。即使训练参数没有直接拟合 test 标签，方法选择本身已经使用了 test 信息，因而结果会受到研究者自由度和多次尝试的选择偏差影响。它可以证明机制值得继续研究，却不能估计方法对真正未见数据的泛化效果。

确认性结论要求重新冻结 outer splits，在看不到 test 真值和指标的情况下完成底模训练、候选选择、超参数选择和模型锁定，然后只读一次 test。CAA、MM、STGF 和 PS-DFSC 比早期实验更接近这一标准，但仍属于单一数据集、同一气候域内的内部确认。

## 2.5 计算平台与“为什么不是全部用 GPU”

神经网络训练和场景生成适合 GPU：卷积、attention、diffusion 与 flow 的核心是大规模张量运算，GPU 通常显著更快。SUC 的 exact mixed-integer linear programming 则不同：SciPy/HiGHS 主要执行分支定界、割平面和稀疏线性规划，这类不规则搜索当前主要依赖 CPU，不能因为机器有 GPU 就自动加速。

PS-DFSC 最耗时的阶段恰好是 150 日、多场景、两套方法和 oracle 的 exact MILP。单次上限 600 秒，最终平均 planning time 约 276 秒，所以终端长时间没有新输出并不等同于程序卡死。后续运行应增加逐日 heartbeat、已完成/总数、incumbent、gap 和预计剩余时间；神经训练用 GPU，exact HiGHS 用 CPU 多核或任务级并行，两者分工才合理。

# 3. MS-CADM baseline：从论文到可运行复现

## 3.1 复现范围

复现工作覆盖数据管线、MS-CADM 主模型、VAE、WGAN、QRGBM、RealNVP、DDPM 与随机基线，重建了主要指标表、采样步数实验、模型消融、区间图、分布图和 RTS-24 SUC 评价。MS-CADM 主运行配置使用 model dimension 128、bottleneck 64、4 层、4 attention heads、FF expansion 4、250 个 cosine diffusion steps、learned variance、VLB 权重 10⁻³、26,000 updates、batch size 256、learning rate 10⁻⁴、condition mask 0.1。正文 headline 的复现 MS-CADM 来自 250-step ancestral sampling；另有默认实用采样 50-step DDIM、η=1.0。VAE-reference 使用 200 epochs、约 2,000 updates，WGAN-reference 使用 300 epochs、约 3,000 generator updates，不能把 26,000 updates 误解为所有 baseline 的共同日程。

## 3.2 主表复现结果

| 方法 | MAE | RMSE | CRPS | QS | ES | VS |
|---|---:|---:|---:|---:|---:|---:|
| 论文 MS-CADM | 0.1191 | 0.1645 | 0.0873 | 0.0441 | 0.5380 | 18.1400 |
| 复现 MS-CADM（250-step ancestral） | 0.124248 | 0.177632 | 0.107400 | 0.054139 | 0.681950 | 22.0200 |
| 相对偏差 | +4.32% | +7.98% | +23.02% | +22.76% | +26.76% | +21.39% |
| 复现 VAE-reference | 0.123630 | 0.167013 | 0.087727 | 0.044337 | 0.545616 | 17.7175 |
| 复现 WGAN-reference | 0.129290 | 0.177066 | 0.096574 | 0.048859 | 0.592980 | 18.5964 |
| QRGBM+ECC 工程重构 | 0.122651 | 0.165515 | 0.085746 | 0.043311 | 0.537024 | 17.1205 |
| 条件 RealNVP 工程重构 | 0.117057 | 0.164107 | 0.085050 | 0.043012 | 0.532192 | 16.9482 |
| 公共能源 WaveNet DDPM 重构 | 0.120106 | 0.160717 | 0.080505 | 0.040624 | 0.506541 | 15.8709 |
| 论文 RAND（评价集采样） | 0.258300 | 0.301200 | 0.169200 | 0.085500 | 0.961500 | 23.2100 |
| 复现 RAND（评价集采样） | 0.258707 | 0.300767 | 0.168710 | 0.085216 | 0.959265 | 23.1721 |
| RAND-train 科学控制 | 0.257022 | 0.301367 | 0.170702 | 0.086209 | 0.971573 | 23.5098 |

随机基线几乎精确匹配论文，说明数据缩放和指标实现并非全面错误；VAE-reference 也与论文接近。RAND 的对表版本从评价观测中抽样，存在泄漏，因此只用于验证论文表格；RAND-train 才是科学控制。QRGBM 的 99 分位、300 棵树和 ECC、条件 RealNVP 以及公共能源 WaveNet DDPM 都是因论文细节不足而做的工程重构，不等同于作者原代码。偏差主要集中在 MS-CADM 及其声称的领先幅度。复现中 DDPM、RealNVP、QRGBM 和 VAE-reference 均在多个 proper score 上优于 MS-CADM，因而不能支持“MS-CADM 是当前最优方法”的原始结论。

## 3.3 严重欠离散

复现 MS-CADM 的 headline 90% 区间 coverage 只有 0.3595。切换为 feature-wise mask 与 linear schedule 后可升至 0.495833，但仍远低于名义 0.90。低 coverage 与较窄区间共同说明：模型不是偶发漏掉几个极端日，而是整体分布过度集中。

[[FIGURE:outputs/full_reproduction/figures/figure5_intervals.png|图 2  baseline 各方法预测区间宽度与覆盖率。MS-CADM 的主要问题是系统性欠离散，而不只是点预测偏差。]]

[[FIGURE:outputs/full_reproduction/figures/figure7_distribution.png|图 3  观测与生成分布比较。分布形状和边界质量差异为后续 residual calibration 与 mixed-measure 建模提供了直接动机。]]

论文的单区域 Table 2 没有披露 zone ID。本项目依据 RAND 指标最接近原则推断为 Zone 1，但 RAND 的 ES 与 VS 又不能同时与任何单一区域完全对齐，因此该身份不是确定事实。按 Zone 1 重构的核心结果如下：

| 方法 | MAE | RMSE | CRPS | QS | ES | VS |
|---|---:|---:|---:|---:|---:|---:|
| 论文 MS-CADM（未披露区域） | 0.1053 | 0.1448 | 0.0785 | 0.0397 | 0.4703 | 15.9600 |
| Zone 1 MS-CADM | 0.118892 | 0.167177 | 0.114263 | 0.057272 | 0.726970 | 21.7558 |
| Zone 1 DDPM | 0.131425 | 0.176913 | 0.094129 | 0.047645 | 0.583121 | 16.2641 |
| Zone 1 QRGBM+ECC | 0.126816 | 0.161964 | 0.090395 | 0.045698 | 0.540240 | 16.0524 |
| Zone 1 RealNVP | 0.134915 | 0.188432 | 0.108969 | 0.055049 | 0.674355 | 19.8047 |
| Zone 1 VAE-reference | 0.132324 | 0.177216 | 0.104507 | 0.052842 | 0.650294 | 20.4224 |
| Zone 1 WGAN-reference | 0.149919 | 0.204118 | 0.130360 | 0.065689 | 0.788365 | 22.6724 |

## 3.4 采样步数与消融

采样步数从 10 增至 250 时，CRPS 从 0.11160 缓慢改善到 0.10798，但运行时间从 15.02 秒增至 351.38 秒。50 步时 CRPS 为 0.10923、耗时 71.04 秒，基本复现了论文关于“50 步后边际收益迅速下降”的定性判断。该 step sweep 是独立的 DDIM、η=1 采样实验，所以表中 250-step CRPS 0.10798 与 headline 250-step ancestral 的 0.107400 不应混为同一档案。

| 采样步数 | 时间/秒 | CRPS |
|---:|---:|---:|
| 10 | 15.02 | 0.11160 |
| 20 | 28.39 | 0.11045 |
| 50 | 71.04 | 0.10923 |
| 100 | 142.11 | 0.10851 |
| 250 | 351.38 | 0.10798 |

Zone 1 消融中，full、no-CE、no-AdaLN、no-LV、no-RCM 的 CRPS 分别为 0.114263、0.119012、0.119210、0.114156、0.119372。条件嵌入、AdaLN 与随机条件遮蔽的方向性作用得到支持；learned variance 几乎没有可辨识收益，甚至 no-LV 略优。论文所称的所有模块贡献并未被同等强度地复现。

## 3.5 SUC 复现

在统一的 7 个 zone-date case、K-Means k=10 和 1200 MW 风电规模下，复现的平均运行结果如下。共同索引为 [37,20,134,253,153,315,420]；每个 case 实际取一个 pooled 单区域轨迹并整体放大到 1200 MW，再平均分配到 RTS-24 的六个固定风电母线，不是十区域到六风场的物理映射。负荷损失/弃风罚金为 1000/80 USD per MWh；规划与真实回放默认 1% MIP gap、120 秒时限。早期 CSV 没有保存 success、dual bound 或实际 gap，因此这里应称 binary MILP reconstruction，而不是带完整最优性证书的 exact 结果。

| 方法 | 总成本（约 USD/day） | 罚成本（约 USD/day） | 负荷损失（MWh/day） | 弃风（MWh/day） |
|---|---:|---:|---:|---:|
| VAE | 340,133.86 | 3,991.69 | 0.0000 | 49.896 |
| DDPM | 353,839.16 | 9,588.99 | 2.0907 | 93.729 |
| QRGBM | 356,558.87 | 20,326.41 | 15.3590 | 62.092 |
| RealNVP | 358,583.83 | 21,913.09 | 16.7790 | 64.182 |
| MS-CADM | 395,386.19 | 60,626.54 | 54.1650 | 80.764 |
| WGAN | 404,085.87 | 68,459.07 | 66.5290 | 24.121 |

概率分数更好的方法并不必然在这一小规模 SUC 上按同样顺序排名；MS-CADM 的严重欠离散与较高负荷损失在方向上相容，但并未被识别为因果关系。模型顺序还受随机 case、单区代理、K-Means 压缩、1% gap 和时限影响。只有 7 个 case，不能据此给出显著性结论。

## 3.6 原文与复现流程中的关键漏洞

- 论文未公开代码与权重，多项架构尺寸、训练/采样细节和 baseline 超参数缺失。
- Algorithm 2 若按字面执行，会在每个反向步骤重新采样 x_t，且未写出 learned variance 的随机项；这与标准反向扩散不一致。
- 正文讨论 KL/VLB，但核心训练式只展示 MSE，权重和实际组合方式不清楚。
- α 的累计与逐步记号存在歧义， reduced-step sampling 的时间步选择规则也未给出。
- RCM 文字更像 element/feature-wise mask；本地 headline 明确使用形状 [batch,1,1] 的 sample-wise mask，alternate branch 才使用 feature-wise mask，两种选择显著影响 coverage。
- RAND 从评价观测中采样，存在信息泄漏；本项目保留 RAND 仅用于对表，另设 rand_train 作为科学控制。
- 论文把所有区域池化，不能证明生成了十区域联合分布。
- 缺少随机种子、多次训练、置信区间和显著性检验。
- PIAW 公式上下界符号写反；Table 1 的 MS-CADM 数值对应 250-step，而正文又推荐 50-step。
- Eq. (13) 的原始气象分量顺序与公共 Dumas 代码不一致；本地选择可执行参考顺序，并保留不常见的 atan2(U,V) 与 degree 风向定义。
- 单区域 Table 2 没有报告 zone ID，且 RAND 的 ES/VS 不能同时与任一区域完全对齐，区域身份无法唯一恢复。
- Table 4 正文称“三个组件”，实际列出 CE、AdaLN、LV、RCM 四项消融。
- 正文写“随机选 7 日”，随后又说 Table 5 为 100 日平均；SUC 的场景压缩、区域到母线映射和备用参数均未完整披露。

## 3.7 baseline 阶段结论

我们完成的是“工程和实验层面的完整重建”，不是“论文数值完全复制”。最可靠的复现结论是：采样步数的时间—性能趋势和若干模块的方向作用可以重现；论文 headline 的绝对指标、相对领先幅度以及概率校准质量不能重现。欠离散成为后续所有改进路线的共同起点。

# 4. CR-MS-CADM：条件残差与经验校准

## 4.1 研究动机与方法

CR-MS-CADM 针对 baseline 的系统性欠离散，采用四个组合改动：条件 location/scale 头把日前条件均值与异方差显式分离；diffusion 只学习标准化 residual；随机条件遮蔽改为 feature-wise；最后使用 validation-only 的单调 empirical PIT calibration 修正边际分位数。

这里的关键思想不是简单“把区间乘宽”，而是把可预测的条件尺度从随机 residual 中分离，再用独立 validation 数据学习单调校准映射。该单调映射在每个条件场景集合内未产生严格秩反转，但可能改变 ties；它不保证跨日全局秩或 exact empirical copula 完全不变。

## 4.2 三种子结果

| 指标 | 受控 baseline | CR-MS-CADM 均值 | 相对变化 |
|---|---:|---:|---:|
| MAE | 0.124168 | 0.118580 | −4.50% |
| CRPS | 0.109229 | 0.084655 | −22.50% |
| ES | 0.693248 | 0.527975 | −23.84% |
| VS | 22.4018 | 16.8398 | −24.83% |
| 90% coverage | 0.3070 | 0.83456 | +0.52756 |

六个预注册方向门槛全部通过，说明“条件残差 + feature-wise mask + 单调校准”确实能系统性修复 baseline 的概率分布。然而，强 DDPM 的 CRPS 0.080505、VS 15.8709、coverage 0.8667 仍优于 CR-MS-CADM，因此该路线主要是修复原模型，而不是建立新的 SOTA。

[[FIGURE:outputs/cr_mscadm/figures/figure1_reliability.png|图 4  CR-MS-CADM 的可靠性曲线。校准后覆盖明显靠近对角线，但 90% 区间仍未达到 0.90。]]

## 4.3 消融与真正的涨点来源

fixed-scale residual、heteroscedastic but no CRPS auxiliary、full raw seed0 的 CRPS 分别为 0.088691、0.087050 和 0.088369。这个结果不能证明 CRPS auxiliary 是必要组件；相反，no-aux 在该次比较中更好。

更清晰的归因来自 mask 实验：sample-wise mask 的校准后 CRPS 为 0.091114、coverage 为 0.67808；feature-wise mask 的 CRPS 为 0.084585、coverage 为 0.84342。由此可见，headline 改善在很大程度上依赖 feature-wise RCM，而不应全部归因于 residual diffusion 或复杂损失。

[[FIGURE:outputs/cr_mscadm/figures/figure2_ablation.png|图 5  CR-MS-CADM 消融。不同模块对 proper score 和 coverage 的影响并不一致。]]

## 4.4 描述性 SUC

在 7 个 zone-date case 上，MS-CADM raw 的平均成本 443,792.82，CR 校准后为 346,743.21，下降 21.87%；负荷损失从 101.58 降到 4.55，罚成本从 109,000.88 降到 7,794.36。方向与“修复欠离散可减少供电风险”一致。

但这 7 个 case 只有 5 个唯一日期，且是单区域代理放大至 1200 MW，不是十区域 joint generator，也没有置信区间。因此这组结果只能作为机制展示，不能作为论文的确认性成本收益。

[[FIGURE:outputs/cr_mscadm/figures/figure8_suc.png|图 6  CR-MS-CADM 描述性 SUC 结果。成本下降主要由负荷损失罚金显著减少驱动。]]

## 4.5 证据边界

CR-MS-CADM 的设计受同一 test 上的欠离散诊断启发，三种子也共享相同 50 个 test calendar days；早期 bootstrap 还把 500 个 zone-days 当作独立单位，而不是以 50 个 calendar days 重采样，可能低估相关性。因此本实验是完整、可复核的探索性证据，不是确认性证据。它最重要的作用是识别 feature-wise mask、条件尺度和校准的有效方向，为后续冻结 outer splits 提供假设。

# 5. RAHC：秩自适应层级校准

## 5.1 研究问题

CR-MS-CADM 的总体 coverage 已大幅提高，但不同风电水平、小时、区域和 spread regime 仍存在条件失配。RAHC 使用 conditional Beta-Binomial rank model，对 ties 采用 interval censoring，并通过 hour/zone/regime/spread 的层级结构估计条件 rank 映射；审计分组包含 zone、hour、wind quintile、spread quintile 与 zone×spread，不包含季节效应。候选同时比较 bounded tail、linear tail 和 clamp tail。

## 5.2 主要结果

从 legacy G0 到 bounded C6：

| 指标 | G0 | C6-bounded | 变化 |
|---|---:|---:|---:|
| CRPS | 0.084655 | 0.085068 | 恶化 0.49% |
| Coverage | 0.83456 | 0.88325 | +0.04869 |
| 区间宽度 | 0.43723 | 0.59450 | +0.15727 |
| Winkler | 0.66984 | 0.74682 | 恶化 |
| MAE | 0.11858 | 0.12718 | 恶化 7.26% |
| VS | 16.8398 | 17.2565 | 恶化 |
| Conditional ACE | 0.06910 | 0.02921 | 改善 57.73% |

配对 bootstrap 中，CRPS“改善量”为 −0.000413，95% CI 为 [−0.000979, 0.000164]，即不能排除无差异或恶化；宽度增加的 CI 为 [0.13214, 0.18323]；conditional ACE 改善的 CI 为 [0.02713, 0.04860]。11 个门槛只通过 4 个。

[[FIGURE:outputs/rahc_cr_mscadm/figures/test_crps_coverage_tradeoff.png|图 7  RAHC 的 CRPS—coverage 权衡。bounded tail 把 coverage 推向目标区间，但代价是明显的宽度扩张与 proper-score 退化。]]

## 5.3 失败原因分析

RAHC 确实找到了条件覆盖失配，但 bounded tail 的修正强度过大，把“局部欠覆盖”处理成“全局尾部扩张”。coverage 接近 0.90 不代表校准更好：当区间靠无差别变宽来覆盖更多观测时，Winkler 和 CRPS 会惩罚这种做法。

C6-linear 的 CRPS 为 0.084353，点估计略优于 G0，但 coverage 降到 0.8275，且置信区间跨过零。这说明 bounded 与 linear tail 分别偏向 coverage 和 sharpness，当前模型没有找到两者的 Pareto 改善。

[[FIGURE:outputs/rahc_cr_mscadm/figures/conditional_zone_spread_heatmaps.png|图 8  zone×spread 条件失配热图。误差具有结构性，但简单层级平滑和统一尾部规则不足以同时修复所有组。]]

## 5.4 排序结构与 SUC

校准后的 stable ordinal rank 约为 0.9583，说明大部分场景顺序被保留；但“没有排序反转”不等同于保留精确 copula，因为边际变换、ties 和有限场景权重都会改变联合经验分布。

描述性 SUC 中，RAHC 相对 CR 的平均成本从 346,743.21 降到 343,396.82，下降 0.97%；负荷损失从 4.5539 降到 0.2486，但 planning cost 增加 0.60%。该实验只有 7 个 zone-date cases、5 个唯一日期，仍是单 Zone 放大到 1200 MW 的系统代理；它重用旧 test，且存在一个 legacy time-limit case，不能抵消 proper-score 门槛失败。

## 5.5 RAHC 的研究价值

RAHC 不是成功的最终方法，但它把“coverage 变高”与“概率质量变好”明确区分开来。它直接推动了下一条路线的两个设计原则：显式建模边界原子；用 hard non-inferiority gate 与 exact fallback 阻止过度校准被发布。

# 6. CAA-RAHC：原子感知的安全校准

## 6.1 方法与确认协议

CAA-RAHC 在 3×50 日 sealed outer tests 上进行内部确认。每个 outer 使用 3 个训练 seed，形成 9 个 outer-seed 结果。方法包含：0/1 atom hurdle、局部 tail gate、proper-score 非劣约束和无可行候选时精确回退 A0。锁定候选为 A4_atom1_tail1_d0p25_w0p02。

该设计针对 RAHC 的两个失败：连续 rank 模型难以表达边界原子；没有发布门槛时，coverage 改善可能掩盖 CRPS 与 sharpness 恶化。

## 6.2 A0 到 A4 的确认结果

| 指标 | A0 | A4 | 变化 |
|---|---:|---:|---:|
| CRPS | 0.0823213 | 0.0814042 | 改善约 1.11% |
| MAE | 0.114580 | 0.113728 | 改善 |
| VS | 16.8004 | 16.5848 | 改善 |
| Coverage | 0.822287 | 0.849972 | +0.027685 |
| 区间宽度 | 0.407945 | 0.409933 | +0.001987 |
| Winkler | 0.677668 | 0.668876 | 改善 |
| Conditional ACE | 0.077370 | 0.051045 | 改善 34.0% |

CRPS 改善量为 0.000917，paired bootstrap 95% CI 为 [0.000619, 0.001242]；conditional ACE 改善量 0.026325，CI 为 [0.021208, 0.031408]；区间宽度只增加 0.001987，CI 为 [0.001086, 0.002869]。9/9 outer-seed 的 CRPS 都改善，3/3 outer 方向一致，说明结果不是由单一 split 或 seed 驱动。

[[FIGURE:outputs/caa_rahc/figures/a4_paired_bootstrap.png|图 9  CAA-RAHC A4 相对 A0 的配对 bootstrap。CRPS 与 conditional ACE 改善稳定，区间宽度代价较小。]]

[[FIGURE:outputs/caa_rahc/figures/outer_seed_consistency.png|图 10  三个 outer、三个 seed 的一致性。A4 在 9/9 outer-seed 上改善 CRPS。]]

## 6.3 最关键的消融发现

A2 atom-only 的 CRPS 为 0.081410，A4 为 0.081404。A2 解释了约 99.37% 的 A0→A4 CRPS 增益。换言之，复杂 tail gate 对 headline CRPS 的边际贡献极小；真正稳健、可解释的创新是对边界原子质量的显式修复。

A4 与 A5 最终选中了同一候选，因此现有结果也没有证明 hard constraint 改变了候选选择。A6 可达到更低的 CRPS 0.081330 和更高 coverage 0.860889，但区间宽度增至 0.423325，反映更激进的 sharpness 交换。

[[FIGURE:outputs/caa_rahc/figures/ablation_tradeoff.png|图 11  CAA-RAHC 消融。atom-only 已获得几乎全部 CRPS 收益，复杂门控的增益有限。]]

## 6.4 未通过的门槛

A4 通过 10 项预注册成功门槛中的 9 项，唯一未通过的是绝对 coverage：0.849972 不在 [0.88,0.92]。最差 pooled zone×spread 组 coverage 仅 0.753788。稳定 rank 约 0.9474，也不能被解释为“精确 copula 保持”。

这决定了论文表述必须是“在几乎不增宽区间的情况下，同时改善 CRPS 与 conditional calibration”，而不是“实现了名义 90% 覆盖”。

## 6.5 描述性 SUC 与审计状态

9 个唯一日期上，A0 成本 386,965.76，A4 为 368,002.50，下降 4.90%；A2 为 369,157.46。负荷损失从 23.8763 降至 4.0008。该实验全部使用 Zone 5，拼接三个 seed 得到 300 场景后压缩为 K=10，再放大为单区域系统代理。由于无置信区间、MIP gap 1%、时限 120 秒，而且协议明确标注 success_gate_role=none，这组结果不属于确认性决策证据。

CAA 的当前权威完成审计是 outputs/caa_rahc/current_schema_completion_audit.json，13/13 检查通过。旧文件 outputs/caa_rahc/completion_audit.json 使用早期 schema，显示 incomplete，属于 stale 产物，不应在论文或后续自动化中引用。

# 7. MM-JDWind：混合测度联合风电生成

## 7.1 为什么需要新生成器

早期 MS-CADM、CR、RAHC 和 CAA 仍以单区域 zone-day 为基本样本，不能直接提供十区域联合场景。与此同时，风电归一化功率在 0 和 1 处存在离散概率质量，内部 (0,1) 又是连续分布。用单一 Gaussian diffusion 或连续 flow 统一建模，会把原子质量模糊成靠近边界的连续样本。

MM-JDWind 将条件分布拆分为 0 原子、1 原子和内部连续部分，使用 mixed-measure head 预测类别质量；用 absorbing jump state field 建模原子状态；对内部连续部分在 logit 空间使用 bounded rectified flow；通过 axial time/zone attention 学习十区域 24 小时依赖；最终在 M=100 的 0.01 有限分辨率与整数化误差范围内，使场景原子频数近似守恒于预测质量，同时保留联合状态结构。实际数据中上边界事件非常稀少，确认测试的 one observed rate 约 0.000139，因此主要实证主张应围绕 zero atom，对 one atom 保守表述。

## 7.2 开发与确认协议

开发阶段为 3 outer×3 seeds，正式候选包括 no-jump、independent、correlated 与 mass-preserving。另尝试过 proper-score fine-tune，但 pre-lock pilot 出现非有限参数而被拒绝；带 proper 前缀的档案仅用于 Monte Carlo 重复审计，不属于候选身份。开发完成后锁定 mass-preserving。确认使用与 CAA 不重叠的新 150 日，3 个 outer、每个 3 seeds，M=100；jump 为 12 NFE、flow 为 16 NFE；强 DDPM 对照使用每个 outer 的 seed0、250 steps。统计采用 20,000 次 calendar-day paired bootstrap。

## 7.3 相对 no-jump 的主要结果

| 指标 | No-jump | Mass-preserving | 结论 |
|---|---:|---:|---|
| CRPS | 0.081897 | 0.079431 | 改善 0.002466；CI [0.001794,0.003165] |
| MAE | 0.116217 | 0.111609 | 改善 |
| Coverage | 0.74424 | 0.84526 | 大幅改善 |
| Joint ES | 1.72841 | 1.69145 | 改善 |
| CRPS outer-seed 一致性 | — | 9/9 改善 | 3/3 outer 同向 |

adjacency VS、aggregate CRPS、ramp 指标和 daily-any-zero Brier 也整体改善。这是在相同连续生成器基础上只改变混合测度/跳跃机制得到的受控证据，因而是当前项目中归因最清楚的结构创新。

## 7.4 与强 DDPM 的比较

| 指标 | MM-JDWind | DDPM | 判断 |
|---|---:|---:|---|
| CRPS | 0.079431 | 0.079151 | CI 跨 0，统计持平 |
| Joint ES | 1.69145 | 1.68838 | 无显著差异 |
| MAE 差 | — | — | MM 优 0.00594 |
| 区间宽度 | 0.38627 | 0.46887 | MM 更尖锐 |
| Zero Brier | 0.04622 | 0.06505 | MM 更好 |
| Ramp-CRPS | 0.05117 | 0.04865 | DDPM 更好 |
| Coverage | 0.84526 | 未作为优势 | MM 仍低于目标 |

因此最准确的结论是：MM-JDWind 在 CRPS 和 joint ES 上与 250-step DDPM 竞争，同时以更少 NFE 获得更好的 MAE、区间宽度和 zero-event probability；但 ramp dynamics 和 absolute coverage 仍需改进。更少 NFE 不自动等于同比例 wall-clock 加速，不同网络单次调用成本不同，仍需同硬件计时基准。

DDPM 对照是逐 zone 独立训练/采样后拼接的基线，不是原生联合模型。因此逐 zone CRPS、MAE、coverage、width、ramp 和 atom 指标是主要公平比较；Joint ES、adjacency VS 等跨区域联合指标只能作描述性强基线比较。MM 为 3 outer×3 seeds，而 DDPM 每个 outer 只有 seed0，当前结论不能覆盖 DDPM 的模型 seed 不确定性。

按预声明成功门统计，mass 相对 no-jump 通过 CRPS≥1%、width 和 outer direction，未通过 analytic zero-Brier≥5% 与 coverage [0.88,0.92]；mass 相对 DDPM 通过 zero-Brier、width 和 outer direction，未通过 CRPS 平均改善≥1% 与 coverage。两组均为 3/5，而不是所有目标全部完成。

## 7.5 原子指标的解释限制

No-jump 和 mass-preserving 共享 analytic atom head，因此“解析的 zero Brier”在两者间理论上不能因为采样机制而改变。注册时若把该项作为 jump 的必然改善门槛，结构上并不合理。更有意义的是 finite M=100 realization 的原子频数、daily-any-zero event Brier 和最终混合分布 proper score。

proper-score fine-tune 曾产生 87 个非有限 flow 参数，已按协议拒绝，没有进入确认。这不是被隐藏的失败，而是数值稳定性审计的一部分。

## 7.6 MM-JDWind 的论文价值

该路线同时解决了三项 baseline 局限：从 pooled zone-day 升级为 native ten-zone joint generation；从纯连续分布升级为 mixed measure；从 250-step diffusion 降到约 12+16 NFE 的组合生成。NFE 下降是计算结构优势，但在补充同硬件 wall-clock benchmark 前不宣称等比例加速。与 CAA 的独立 atom-only 证据相互印证，使“边界质量守恒”成为项目最有潜力的论文贡献。

# 8. STGF-Flow：时空图频域流模型

## 8.1 假设与架构

STGF-Flow 假设十区域之间存在可由固定图基表示的空间相关，24 小时轨迹也可在频域稀疏表达。模型比较 time-domain、graph-only、time-frequency 与 full STGF（GFT+DCT），统一采用 Heun 16 NFE。确认前发现不同候选使用的 basis 不完全一致，因此先实施 common-basis correction，再冻结锁文件；最终锁定哈希为 f52d…b2261。正式确认共使用 6 个唯一 flow 模型并在 3 个互斥 test blocks 上评价，形成 18 个 evaluation cells；这不是 18 次独立重训，也不是 18 个独立测试集。所有模型使用排除完整 150 test 日的 universal 481-day train。

## 8.2 确认结果

| 方法 | CRPS | MAE | Coverage | Width | Joint ES |
|---|---:|---:|---:|---:|---:|
| Time-domain | 0.079947 | 0.108948 | 0.78686 | 0.33403 | 1.70756 |
| STGF all seeds | 0.083135 | 0.113274 | 0.87219 | 0.46754 | 1.73122 |
| DDPM | 0.083188 | 0.122494 | 0.85958 | 0.46206 | 1.79016 |

相对 time-domain，STGF 的 CRPS 恶化 3.99%，Joint ES 恶化 1.39%，adjacency VS 恶化 8.85%；CRPS 差为 +0.003187，95% CI [0.002281,0.004099]，三个 blocks 均变差。更严格的 matched-seed 比较中，STGF seed0 − time-domain seed0 的 CRPS 差为 +0.002625，CI [0.001826,0.003412]，说明失败不是由 seed1/2 拖累。STGF seed0 相对 time-frequency 的 CRPS 也恶化 +0.000933，CI [0.000428,0.001440]。6/6 预注册门槛全部失败。

## 8.3 为什么固定图频结构失败

实验直接证明的是：固定图频表示在当前协议下的分数更差，graph-only/STGF 相对相应消融没有净增益。一个与结果一致但尚未被因果识别的解释是：固定相关图把训练期平均相关当成稳定拓扑，而区域相关会随天气系统、风向、季节和误差状态变化；GFT 可能产生错误平滑，DCT 对 24 小时非平稳 ramp 也未必稀疏。

Time-domain 模型虽结构简单，却能直接在原始坐标学习条件依赖，避免了错误 basis 的信息瓶颈。该结果说明下一步如果继续图模型，应学习动态图或条件图，而不是增加固定图频模块的深度。

## 8.4 正向结果

核心假设失败不等于模型毫无价值。STGF 相对 250-step DDPM 的 CRPS 基本持平，MAE 更低 0.00922，Joint ES 更低 0.058936，且后者 CI 全负；spectral MAE 为 0.07821，对比 DDPM 的 0.09521。它只需 16 NFE，并且原生生成十区域联合场景；在没有同硬件计时前，只主张函数评估次数更少，不主张同比例加速。

但 STGF 的 ramp 与 zero Brier 劣于 DDPM。DDPM 仍是逐 zone 独立基线，因此 Joint ES、adjacency 等联合指标属于结构不对称的描述性比较；检验图频归纳偏置的主要受控证据是 STGF 相对同为原生联合 flow 的 time-domain 对照。论文若使用这组结果，必须写成“低 NFE 的竞争性联合 flow”而非“图频结构显著优于时域模型”。

# 9. PS-DFSC：Proper-score 约束的决策导向场景校准

## 9.1 研究定位

PS-DFSC 不重训场景生成器，而是冻结 MM-JDWind 的 M=100 个十区域联合场景，用小型重加权器和有界单调 transport 改善 stochastic UC 决策。它与简单加入“调度成本损失”的区别是：CRPS、ES、VS、ramp-CRPS、zero-atom Brier、coverage、conditional ACE、ESS、熵和 transport budget 都作为发布约束；无安全候选时整个 outer 精确回退原分布。

已有工作已经研究 decision-focused scenario generation/selection、直接把 stochastic optimization 目标放入 diffusion，以及 value-oriented UC forecast combination。因此本路线的潜在差异不在“生成器 + 成本损失”，而在 frozen high-dimensional posterior calibration、exact binary two-stage SUC、proper-score hard non-inferiority 和 exact fallback 的组合。

## 9.2 已实现模型

场景重加权器使用共享 temporal CNN（Conv1d 10→32→64）编码每个场景，DeepSets mean/max pooling 建立集合上下文，共享 MLP 输出 logits；最后一层零初始化，因此初始输出严格为均匀权重。模型对场景排列等变。

Transport 对同一 zone-hour 的所有场景使用共同的单调函数：T(x)=σ(a·logit(x)+b)，其中 a∈[0.9,1.1]、b∈[−0.15,0.15]，并精确保留 x=0 和 x=1。固定 KMeans assignment 把 100 个场景压缩为 K=20；簇概率为权重和，代表场景为 transport 后的加权重心。

逐日运行门槛为 ESS≥50、归一化熵≥0.85、平均 transport≤0.02，并检查 NaN 与压缩失败。公开数据结构已包含 full scenarios、probabilities、SUC scenarios、SUC probabilities、ESS、entropy、transport cost、fallback flag 和 fallback reason。

## 9.3 十区域到六风场映射

区域聚类只读取对应 outer 的底模训练数据，使用 24 小时轨迹相关距离，得到四个双区域组和两个单区域组；每区 120 MW，总容量 1200 MW，映射到 RTS-24 的 buses 3、5、7、16、21、23。

| Outer | 六个区域组（0-based Zone ID） |
|---|---|
| Outer 1 | (0,6), (1,9), (2,8), (3), (4,5), (7) |
| Outer 2 | (0,6), (1,9), (2,8), (3), (4,5), (7) |
| Outer 3 | (0,8), (1,9), (2), (3), (4,5), (6,7) |

## 9.4 真正的两阶段 exact SUC

旧模型只有 binary commitment 属于一阶段；一旦固定 commitment，真实日 dispatch 与场景权重无关，决策损失对校准器产生零梯度。新模型把日前基准出力和上下备用也放入一阶段，场景 recourse 包含上下调节、实际出力、弃风、负荷损失、DC flow 和节点相角。

备用需求为负荷的上 10% 与下 5%；备用容量成本为相应边际能源成本的 10% 与 5%；备用短缺罚金为负荷损失罚金的 50%。Exact MILP 使用 SciPy/HiGHS，目标 relative MIP gap 0.1%，单次时限 600 秒，并保存 incumbent、dual bound、实际 gap、运行时间和终止原因。

可微训练采用固定 commitment 的连续 QP surrogate，加入 10⁻⁴ 强凸项。需要特别说明：当前 QP 是 aggregate copper-plate surrogate，不与含网络约束的 exact MILP 完全等价，因此代理收益必须经 exact 模型复核。

## 9.5 数据隔离与训练流程

旧 150 日作为 PS 开发数据，固定 100 日训练、50 日验证；剩余 181 日按月份与 NWP WS100 分层，冻结 3 个互斥 outer test，每组 50 日，余 31 日不用。每个 outer 的 MM-JDWind 底模训练排除 dev150 和本 outer50，因此每个 seed 使用 431 日；三个 seed 按 34/33/33 合并成 M=100 场景。9 个底模重训均通过审计。

校准器训练 50 epochs、20 epochs proper-score warm-up；decision epoch 使用 8 个完整训练日；在 epoch index 35（自然计数第 36 轮）对 100 个训练日刷新 exact commitment。β∈{0,0.25,0.5,1}，共训练 12 个候选；proper-score 通过 augmented Lagrangian/primal-dual 更新，不是固定加权和。

## 9.6 验证结果：12 个候选全部不安全

| Outer | β | Coverage | 日级 fallback 数 | 主要失败（非穷尽） |
|---:|---:|---:|---:|---|
| 1 | 0 | 0.874667 | 9/50 | coverage、CRPS、ramp 及多项置信门 |
| 1 | 0.25 | 0.866083 | 20/50 | coverage、proper scores |
| 1 | 0.5 | 0.865417 | 19/50 | coverage、proper scores |
| 1 | 1 | 0.859917 | 32/50 | coverage、ACE/score 门槛 |
| 2 | 0 | 0.871333 | 17/50 | coverage、ramp 及多项置信门 |
| 2 | 0.25 | 0.862333 | 19/50 | coverage、proper scores |
| 2 | 0.5 | 0.861000 | 26/50 | coverage、proper scores |
| 2 | 1 | 0.861333 | 30/50 | coverage、proper scores |
| 3 | 0 | 0.866750 | 24/50 | coverage、CRPS、ACE 及多项置信门 |
| 3 | 0.25 | 0.867333 | 25/50 | coverage、proper scores |
| 3 | 0.5 | 0.866917 | 35/50 | coverage、proper scores |
| 3 | 1 | 0.865000 | 36/50 | coverage、proper scores |

所有候选 coverage 都低于 0.88；不少候选还违反 CRPS/ES/VS/ramp/zero-atom Brier 或 ACE 门槛。逐日 fallback 全部由 transport budget 超限触发，没有 ESS、entropy 或 NaN 失败；平均 ESS 为 98.224–99.988。直接证据说明候选经常产生超过 0.02 预算的 transport，而不是权重塌缩；由于 no-decision 与 transport-only 消融尚未完成，不能把超预算严格归因于 decision loss。

[[CUSTOM_FIGURE:ps_gate_matrix|图 12  PS-DFSC 验证门槛矩阵。12 个候选均未进入安全 shortlist，因此没有候选获得 exact validation 发布资格。]]

## 9.7 代理收益为什么不能当成果

每个 outer 的最佳代理 β 分别为 0.5、0.25、0.25，预测平均成本下降约 0.51%、1.02%、2.00%，CVaR90 下降约 3.44%、4.53%、6.44%。这些数值只来自 QP surrogate，且候选全部不安全；它们只能说明模型找到了“通过改变分布来降低代理成本”的方向，不能写成运行成本真实下降。

由于没有安全 shortlist，三个 outer 的 selection JSON 都是 candidate=null，最终锁定 identity fallback。确认阶段 PS-DFSC 的 100 场景、均匀权重、零 transport 与 MM-JDWind baseline 完全一致，所以 realized cost、CVaR 和风险事件的 paired difference 均严格为 0。

## 9.8 150 日 exact confirmation

Cost regret 定义为方法的 realized cost 减去同一真实风电轨迹作为唯一场景时的 perfect-information single-truth exact UC cost。成本为模型成本单位；load shedding、curtailment 与 reserve shortage 为 24 个一小时步长的累计量，可按 MWh 解释。

| 指标 | Outer 1 | Outer 2 | Outer 3 | 汇总/平均 |
|---|---:|---:|---:|---:|
| 成功 incumbent 日 | 50/50 | 50/50 | 50/50 | 150/150 |
| 达 0.1% gap 日 | 41/50 | 42/50 | 42/50 | 125/150 |
| Realized cost | 399,558.78 | 396,337.70 | 430,708.55 | 408,868.34 |
| Cost regret | 28,506.47 | 31,219.47 | 53,528.90 | 37,751.61 |
| CVaR90 | 522,879.89 | 579,601.13 | 681,129.88 | 601,706.17 |
| CVaR95 | 536,426.73 | 607,232.73 | 723,128.67 | 653,539.12 |
| Load shedding | 10.574 | 15.812 | 46.063 | 24.150 |
| Curtailment | 126.726 | 91.767 | 38.654 | 85.716 |
| Reserve shortage | 0 | 0 | 0 | 0 |
| Mean planning time/秒 | 267.12 | 283.26 | 277.55 | 275.98 |
| 最大 gap | 3.5867% | 2.8333% | 3.3739% | — |

所有 150 日均找到 incumbent，但只有 83.33% 达到目标 gap；未达目标的 case 没有静默替换。由于方法等于 identity，上表同时是 PS-DFSC 与 baseline 的结果，不能声称 cost 或 CVaR 改善。

正式审计字段为 primary_mean_cost_test_passed=false、safe_improvement_claim_allowed=false、test_results_used_for_model_selection=false。在 1.25、1.5 和 2.0 倍 worst-case penalty 敏感性下，由于两方法仍逐日相同，成本改善也均为 0%。

## 9.9 确认集概率安全审计

| Outer | CRPS | ES | VS | Ramp-CRPS | Coverage | Winkler | Zero-atom Brier |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.072247 | 1.557037 | 0.038915 | 0.022479 | 0.872667 | 0.575823 | 0.048952 |
| 2 | 0.074574 | 1.598492 | 0.043002 | 0.024231 | 0.886667 | 0.592748 | 0.048718 |
| 3 | 0.075597 | 1.651961 | 0.042660 | 0.021630 | 0.863583 | 0.594451 | 0.061226 |

Outer 2 的 coverage 点估计落入 [0.88,0.92]，但 bootstrap 区间仍未完整落入；outer 1 和 3 连点估计也未通过。汇总 coverage 约为 0.87431。Identity fallback 的含义是“相对底模不退化”，不是“底模已经满足绝对安全校准”。

## 9.10 PS-DFSC 的实际成果与未完成事项

已经完成的成果包括：可排列等变的重加权器、有界单调 transport、固定可微压缩、两层 fallback、十区到六风场训练期映射、真正的两阶段 exact SUC、固定 commitment QP gradient path、outer split audit、9 个底模、12 个校准器、validation locks、confirmation manifest 和 150 日 exact 运行。

尚未形成正式结果的必做消融包括：weights-only、transport-only、无 decision loss、取消 proper-score hard constraint、K=10、六种母线循环平移、强凸系数 10⁻⁵/10⁻³、备用成本与短缺罚金敏感性，以及 relaxed/exact objective gap、commitment Hamming distance、计划 MAE 和代理—exact 改善相关性。部分代码与配置存在，但没有完整正式产物，因此不能列为已经完成的论文证据。

## 9.11 对下一轮的直接启示

当前主要瓶颈不是权重退化，而是底模 coverage 约 0.87、当前 0.02 transport 预算、候选 transport 行为和 QP surrogate 之间存在冲突；现有证据不能单独判断是预算过紧还是 transport 过激。下一轮最有信息量的顺序是：先运行 weights-only，验证不移动 support 时是否能获得小幅安全收益；再研究相对非劣加分层 absolute target 的 coverage 约束；随后用 network-aware differentiable recourse 或固定 exact commitment 的 nodal QP 缩小 surrogate mismatch；最后才考虑端到端微调生成器。任何门槛或预算变更都必须重新预注册并使用新的 outer splits，不能在当前 confirmation test 上调参后继续称为确认性结论。

# 10. 跨实验综合

## 10.1 指标演化

| 模型/阶段 | CRPS | Coverage | 关键比较对象 | 主要解释 |
|---|---:|---:|---|---|
| MS-CADM headline reproduction | 0.107400 | 0.3595 | 论文 0.0873 | 严重欠离散，未复现 headline |
| CR-MS-CADM | 0.084655 | 0.83456 | 受控 baseline 0.109229 | 探索性大幅修复 |
| RAHC C6 | 0.085068 | 0.88325 | CR legacy G0 | coverage 提高但 proper score 恶化 |
| CAA A4 | 0.081404 | 0.84997 | A0 0.082321 | 内部确认的小而稳定改善 |
| MM-JDWind mass | 0.079431 | 0.84526 | no-jump 0.081897 | 混合测度结构稳健有效 |
| STGF | 0.083135 | 0.87219 | time-domain 0.079947 | 固定图频假设失败 |
| PS-DFSC | = identity | ≈0.87431 | MM identity | 安全回退，无成本改善 |

该表不能被读成严格的单一 leaderboard，因为各阶段使用的 split、联合维度和模型形式不同。它更适合展示研究问题如何演化：从严重欠离散，到边际校准，再到边界原子与联合分布，最后进入受约束的决策优化。

## 10.2 五条可泛化规律

### 规律一：coverage 不能单独优化

RAHC 说明 coverage 接近 0.90 可以通过明显扩宽区间实现，但 CRPS、Winkler 和点预测同时恶化。有效校准必须同时报告 reliability 与 sharpness，并设置 proper-score 非劣约束。

### 规律二：边界原子值得原生建模

CAA 的 A2 和 MM 的 mass-preserving 在不同协议下都支持 atom-aware 设计。这条证据比某个 attention block 的小幅消融更稳健，也更有物理解释。

### 规律三：联合建模不能由池化单区实验替代

早期模型把 zone-day 当独立样本，无法评估跨区域相关风险。MM-JDWind 与 STGF-Flow 首次进入十区域联合分布建模；PS-DFSC 进一步把十区域联合场景映射到六风场并接入 network SUC。此前 CR、RAHC、CAA 的 SUC 仍是单区域系统代理。

### 规律四：代理目标必须经 exact decision 审计

PS 的 QP 代理给出可观收益，但安全门槛和 exact 发布流程拒绝了它。任何只报告 relaxation cost 的 decision-focused 结果都可能高估实际价值。

### 规律五：负结果可以减少后续搜索空间

RAHC 排除了无约束 bounded-tail 扩张，STGF 排除了固定图频必然优越，PS 暴露了 absolute coverage gate 与弱校准底模的可行域冲突。这些结论能显著减少下一轮无效训练。

## 10.3 推荐的三层论文策略

第一层主论文：MM-JDWind。突出 mixed measure、mass-preserving jump、native joint generation 与低 NFE；以 no-jump 为核心受控对照，以 DDPM 为强基线。

第二层校准论文或附加章节：CAA atom-aware calibration。简化掉贡献不清楚的复杂 tail gate，围绕“边界原子修复 + proper-score 安全发布”重新设计；在新的外部数据集上确认 absolute calibration。

第三层后续决策论文：PS-DFSC 2.0。保留 fail-closed 框架，先完成所有消融与 surrogate mismatch 审计，再在至少一个安全候选出现后进入 exact comparison。当前版本适合作为方法平台和负结果，不适合作为“成本下降 1%–3%”的论文。

## 10.4 对外表述的红线

- 不得把 CR、RAHC 或小样本 SUC 写成确认性结果。
- 不得说 CAA 达到了 90% coverage；其值约为 0.8500。
- 不得说 MM 显著优于 DDPM 的 CRPS；两者统计持平。
- 不得说 STGF 优于 time-domain；预注册假设 6/6 失败。
- 不得把 PS 的 QP proxy 下降写成 realized exact cost 下降；确认改善为 0。
- 不得把 identity fallback 称为“通过绝对安全门槛”；它只保证相对输入不退化。
- 不得把尚无正式产物的 PS 消融写成已完成。
- 不得把 MM/STGF 相对逐 zone 独立 DDPM 的跨区域联合指标称为完全公平的结构比较。
- 不得把 9 个 outer-seed 或 STGF 的 18 个 evaluation cells 当成 9/18 个独立测试集。

# 11. 工程资产与审计状态

## 11.1 代码与统一入口

Baseline 与传统 SUC 代码位于 repro/ 和 repro_scripts/；PS-DFSC 的新增包位于 ps_dfsc/，统一入口为 repro_scripts/run_ps_dfsc.py。核心公开接口为：

- calibrate(model, base_scenarios, mapping, …) → CalibratedDistribution
- solve_two_stage_suc(wind_by_farm, probabilities, …) → SUCResult
- evaluate_realized(first_stage, observed_wind, …) → SUCResult

CalibratedDistribution 同时保留完整 100 场景分布和供 SUC 使用的 20 场景压缩分布，并携带 ESS、熵、transport cost 与 fallback 原因，便于逐日审计。

## 11.2 权威报告与产物

| 阶段 | 权威说明/审计产物 |
|---|---|
| baseline | FINAL_REPRODUCTION_REPORT.md；REPRODUCTION_AUDIT.md；outputs/full_reproduction/completion_audit.json |
| CR | CR_MSCADM_EXPERIMENT_REPORT.md；outputs/cr_mscadm/ |
| RAHC | RAHC_CR_MSCADM_EXPERIMENT_REPORT.md；outputs/rahc_cr_mscadm/ |
| CAA | CAA_RAHC_EXPERIMENT_REPORT.md；outputs/caa_rahc/current_schema_completion_audit.json |
| MM | MM_JDWIND_RESEARCH_REPORT.md；outputs/mm_jdwind_confirmation_v1/final_analysis/ |
| STGF | STGF_FLOW_RESEARCH_REPORT.md；outputs/stgf_confirmation_v2/final_analysis/completion_audit.json |
| PS | PS_DFSC_IMPLEMENTATION.md；PS_DFSC_PUBLICATION_RUNBOOK.md；outputs/ps_dfsc/confirmation/report.json |

Baseline 完成审计 passed=true，覆盖 11 类检查、17 张图、checkpoints、hash 和 22 个测试。CAA 当前 schema 为 13/13 complete；STGF completion audit 为 complete 且 issues=[]。MM 没有独立 completion_audit，主要依赖 final analysis hashes 与测试记录，这是治理上的缺口。PS 有 split audit、validation locks、pre-confirmation absence audit、lock manifest 和 exact result archives。

## 11.3 已知陈旧文件

README.md 中“last 70 days”等描述属于早期 prototype，不应作为正式 split 依据。FORMAL_REPRODUCTION_STATUS.md 和 FULL_REPRODUCTION.md 保留了历史未完成状态。CAA 的旧 completion_audit.json 也已被 current_schema_completion_audit.json 替代。后续发布前应建立单一 artifact registry，显式标记 authoritative、superseded 和 exploratory。

## 11.4 复现完整性的最终评价

从工程角度，baseline 主实验、六条改进链和 PS exact confirmation 都有可运行代码与大部分审计产物；从论文角度，真正能够支撑强主张的只有经过新 outer split 的受控比较。项目当前不是“所有想法都成功”，而是已经建立了能拒绝不安全候选、保留负结果并追踪数据泄漏风险的研究系统。这种可证伪性本身是后续发表质量的重要保障。

# 12. 下一步工作计划

## 12.1 最高优先级：完成 MM-JDWind 论文闭环

1. 在至少一个不同气候域或公开风电数据集上做外部确认。
2. 增加 extreme ramp、zero-duration、one-duration 与 spatial event 的专项指标。
3. 统一 no-jump 与 mass 的 analytic/finite atom metric 定义，修正结构上不可改善的门槛。
4. 补充计算成本、显存、训练时间、NFE 和采样吞吐。
5. 为 MM 建立独立 completion audit 与冻结 manifest。

## 12.2 第二优先级：简化 CAA

以 A2 atom-only 为主候选，重新设计最小充分方法。只保留能在独立 outer 上证明增益的组件；把 tail gate 改为严格局部、预算受限的 transport，并以 calendar day 为 bootstrap 单位。目标不是强行达到 0.90 coverage，而是在 CRPS、Winkler 和 conditional ACE 上获得 Pareto 改善。

## 12.3 第三优先级：PS-DFSC 可行域诊断

先完成 weights-only 与 no-decision 两个低成本消融。如果 weights-only 也没有安全收益，说明当前 100 场景 support 缺少决策相关极端状态，应回到生成器增强 tail/ramp support；如果 weights-only 有收益而 transport-only 失败，则应收缩或取消 transport。随后加入 network-aware QP，量化 surrogate 与 exact 的 mismatch，再决定是否扩大 transport budget。

## 12.4 运行工程改进

Exact MILP 运行器应每完成一日立即写入原子化 checkpoint，并输出 heartbeat、当前 outer/day、incumbent、dual bound、gap、elapsed time 和 ETA；跨日任务可并行，但单个 solver 的线程数要与并发数协调，避免 CPU oversubscription。GPU 仅承担 neural calibration 和 scenario generation，不应等待 GPU 加速 HiGHS。

# 附录 A  论文报告值与 baseline 复现差异

本附录的“本地复现 MS-CADM”专指 250-step ancestral headline 档案。相对差定义为（本地−论文）/论文；下列六项指标均为越低越好，因此正值表示相对退化。

| 指标 | 论文 MS-CADM | 本地复现 | 绝对差 | 相对差 |
|---|---:|---:|---:|---:|
| MAE | 0.1191 | 0.124248 | +0.005148 | +4.32% |
| RMSE | 0.1645 | 0.177632 | +0.013132 | +7.98% |
| CRPS | 0.0873 | 0.107400 | +0.020100 | +23.02% |
| QS | 0.0441 | 0.054139 | +0.010039 | +22.76% |
| ES | 0.5380 | 0.681950 | +0.143950 | +26.76% |
| VS | 18.1400 | 22.0200 | +3.8800 | +21.39% |

Zone 1 复现值为 MAE 0.118892、CRPS 0.114263、ES 0.726970、VS 21.7558；论文分别为 0.1053、0.0785、0.4703、15.96。Zone 1 的偏差比 pooled headline 更明显。

| 方法 | 论文 CRPS | 本地 CRPS | 论文 ES | 本地 ES | 论文 VS | 本地 VS |
|---|---:|---:|---:|---:|---:|---:|
| RAND（评价集采样） | 0.1692 | 0.168710 | 0.9615 | 0.959265 | 23.21 | 23.1721 |
| QRGBM / QRGBM+ECC 重构 | 0.1036 | 0.085746 | 0.6255 | 0.537024 | 20.09 | 17.1205 |
| WGAN / WGAN-reference | 0.0979 | 0.096574 | 0.6052 | 0.592980 | 19.87 | 18.5964 |
| VAE / VAE-reference | 0.0880 | 0.087727 | 0.5482 | 0.545616 | 17.87 | 17.7175 |
| NF / 条件 RealNVP | 0.0907 | 0.085050 | 0.5671 | 0.532192 | 18.54 | 16.9482 |
| DDPM / WaveNet 重构 | 0.0981 | 0.080505 | 0.5985 | 0.506541 | 19.61 | 15.8709 |
| MS-CADM | 0.0873 | 0.107400 | 0.5380 | 0.681950 | 18.14 | 22.0200 |

该表表明 RAND、VAE-reference、WGAN-reference 与论文较接近，而工程补全后的 QRGBM、RealNVP 与 DDPM 明显强于论文对应行；这会改变相对排名，也是“原文 baseline 细节不足”对结论影响最大的证据之一。

# 附录 B  PS-DFSC 门槛与回退逻辑

## B.1 候选发布门槛

候选必须同时满足：CRPS ratio≤1.005；ES、VS、ramp-CRPS、zero-atom Brier ratio≤1.01；coverage∈[0.88,0.92]；conditional ACE 不高于 baseline；每天 ESS≥50、归一化熵≥0.85、transport≤0.02。Bootstrap 使用 10,000 次、以配对 calendar day 为重采样单位：score ratio 与 ACE 差使用单侧 95% 上界，coverage 使用双侧 95% 区间，且上下端都必须落入 [0.88,0.92]。

选择顺序为：QP 代理筛选；安全候选前三名运行 50 日 exact MILP；只保留 exact J_select 不高于 baseline 的候选；exact J_select 差在 0.1% 以内视为并列，再依次选择 ESS 更高、transport 更小者。当前没有任何候选进入第二步 exact validation。

## B.2 两层回退

- Outer 级回退：验证阶段没有安全候选，则整个 outer 锁定 identity。
- 日级回退：部署日出现 NaN、ESS/熵/transport 越界或压缩失败，则该日返回原场景和均匀概率。

回退不能根据 test 真值触发。当前三个 outer 都发生 outer 级 identity 回退；确认集没有按测试表现选择性回退。

# 附录 C  关键局限与开放问题

| 问题 | 当前状态 | 对结论的影响 |
|---|---|---|
| 单一公开数据域 | 所有确认仍为 GEFCom2014 内部划分 | 不能声称跨域泛化 |
| 早期 pooled zone-day | baseline/CR/RAHC/CAA 非 native joint | 小样本 SUC 只能描述性解释 |
| Coverage 长期低于 0.90 | CAA、MM、PS identity 均约 0.85–0.87 | 绝对可靠性尚未解决 |
| MM ramp-CRPS 弱于 DDPM | 已确认 | 需增强高频/极端 ramp 支持 |
| STGF 固定 basis | 已被 time-domain 显著击败 | 应转向条件动态图或放弃图频 |
| PS surrogate mismatch | 尚未系统量化 | 代理收益不可外推到 exact |
| PS 消融不完整 | 无正式产物 | 不能定位安全失败的最小原因 |
| Exact gap | 125/150 达 0.1% | 成本报告需同时给 gap 与 penalty sensitivity |
| MM completion audit | 缺独立统一审计 | 发布治理需补齐 |

# 附录 D  术语表

| 缩写 | 含义 |
|---|---|
| CRPS | Continuous Ranked Probability Score，边际概率预测 proper score |
| ES | Energy Score，多维联合分布 proper score |
| VS | Variogram Score，强调变量间差分结构 |
| ACE | Absolute Calibration Error；conditional ACE 按预注册条件组汇总 |
| RCM | Random Conditional Masking，随机条件遮蔽 |
| CAA | Constraint/Atom-Aware calibration，本项目的原子感知安全校准路线 |
| MM-JDWind | Mixed-Measure Joint Distribution Wind 场景生成器 |
| STGF | Spatio-Temporal Graph-Frequency flow |
| PS-DFSC | Proper-Score-Constrained Decision-Focused Scenario Calibration |
| SUC | Stochastic Unit Commitment，随机机组组合 |
| ESS | Effective Sample Size，场景权重有效样本数 |
| NFE | Number of Function Evaluations，生成时函数评估次数 |
| Identity fallback | 原场景、均匀权重、零 transport 的精确回退 |

# 参考文献与相关工作

1. Zhang, J., Liu, S., Shi, Z., & Li, Y. Wind power scenario generation via multi-scale condition adaptive diffusion model. Electric Power Systems Research, 255 (2026), 112753. DOI: 10.1016/j.epsr.2026.112753.
2. Zhou, Y., Zhou, Y., Morstyn, T., & Wang, Y. Decision-Focused Scenario Generation and Selection for Efficient and Robust Grid Dispatch. arXiv:2607.05830, 2026. [arXiv 记录](https://arxiv.org/abs/2607.05830)
3. Sun, H., & Liu, A. Diff2SP: Diffusion Models for Correlated Scenario Generation in Stochastic Programming. arXiv:2606.05649, 2026. [arXiv 记录](https://arxiv.org/abs/2606.05649)
4. Ghazanfariharandi, M., & Mieth, R. Value-Oriented Forecast Combinations for Unit Commitment. arXiv:2503.13677, 2025. [arXiv 记录](https://arxiv.org/abs/2503.13677)

[[END_REPORT]]
