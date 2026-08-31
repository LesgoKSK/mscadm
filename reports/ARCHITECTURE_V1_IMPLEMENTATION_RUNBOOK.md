# Architecture v1 实施说明与实验运行手册

> 状态：研究架构与无泄漏数据底座的第一版实现说明  
> 适用对象：组会讲解、代码复核、后续实验预注册  
> 重要结论：当前版本是在**建立公平比较的地基**，并没有提前选定 diffusion、rectified flow 或概率状态空间模型。

---

## 0. 一页读懂

我们现在不是要在某一篇论文的架构上继续“雕花”，而是在回答一个更基础的问题：

> 风电联合场景生成中，当前误差到底主要来自连续生成器、跨时刻状态建模，还是 0/1 边界状态（atom）处理？

为了让后面的答案可信，architecture v1 先完成两件事：

1. **重置数据协议**：已经参与过架构诊断的 300 天不再假装是“未见测试数据”，而是永久标为 `R-SEEN`；剩余 431 天重新分为训练、验证、校准和模型选择四个角色。当地数据不再设置虚假的最终测试集。
2. **建立共同模型底座**：所有候选模型尽量共享 NWP 编码器 `E`、完整 atom 模块 `A`、输出语义、缺失值掩码和评估协议，再逐项替换真正想研究的机制。

当前已经落盘的两个模型只承担基线和负对照作用：

- `R0`：新的、干净的 common-shell 联合时域 Rectified Flow 基线；
- `T0`：在 `R0` 上增加一个**参数匹配但不使用上一时刻状态**的 cell。

必须避免两个误读：

- `R0` **不是**旧 STGF/Time-domain 代码的严格复现；
- `T0` 的 “memoryless” **只修饰新增的 cell**，不代表整个生成网络看不到其他小时。其 RF 主干仍有全 24 小时时间注意力。

本阶段只应运行单元测试和小规模 smoke；正式证据从冻结协议后的 3-seed 实验开始。

---

## 1. 为什么必须重置数据协议

### 1.1 300 天已经不再是“未知数据”

前面的跨模型诊断使用了两个互不重叠的 150 天面板：

- MM-JDWind 与 DDPM 的主诊断面板：150 天；
- STGF/Time-domain 与 DDPM 的次诊断面板：150 天。

这些结果已经影响了我们对后续模型的判断，例如要重点检查 ramp、cross-lag、atom duration 和 NWP regime。即使模型参数没有直接在这些日期上反向传播，只要研究者已经看过这些日期上的结果，并据此决定架构，它们就不再具备“最终盲测”的证据地位。

所以新协议把两组日期的并集永久标记为：

```text
R-SEEN = research-seen / architecture-diagnosis-only
```

它的含义不是普通 validation，也不是可反复查看的 test，而是：

- 可以作为“我们为什么提出下一步问题”的历史诊断证据；
- 不参与新模型拟合；
- 不参与 early stopping 或 checkpoint 选择；
- 不参与后处理校准；
- 不参与架构优选或最终分数汇报。

如果再用这 300 天证明新架构优于旧架构，就会形成“先看答案、再出题、最后仍用原答案判分”的闭环。

### 1.2 旧的全局 forward-fill 存在跨 split 风险

旧数据流程曾在切分之前对目标序列做全局 forward-fill。这样一个缺失点可能从另一个角色、另一个日期继承数值。即使数量不大，也会破坏“训练/选择信息边界可证明”的要求。

architecture v1 把顺序固定为：

```text
只读 NWP 日历
      │
      ▼
先冻结每一天的角色
      │
      ▼
再读取目标与原始缺失位置
      │
      ▼
只在同一 role × zone × calendar_day 内 ffill，再 bfill
      │
      ▼
保存 raw target、filled target 和 observed_mask
```

其中 filled target 只用于保证模型输入数组有限；所有监督损失必须使用 `observed_mask`，不能把填充值当成真实标签。ramp 指标还要使用更严格的掩码：相邻两个小时都原始可见时，该 ramp 才能计分。

### 1.3 为什么当地数据没有 final

GEFCom2014 本地数据共有 731 个完整日。300 天已经成为 `R-SEEN`，剩余 431 天还要承担模型开发所需的四个角色。因此，本地数据已经没有一块既未参与问题形成、又未参与开发决策的诚实 final。

代码对此采取 fail-closed 策略：

```python
protocol.require_final_available()
```

在外部数据注册表尚未冻结时，这个调用会直接报错。不得把 validation、selection 或 `R-SEEN` 临时改名为 final。

---

## 2. 731 天具体如何分配

冻结配置位于 `repro_configs/architecture_v1.json`。完整日期结构为：

| 角色 | 天数 | 可以做什么 | 明确不能做什么 |
|---|---:|---|---|
| `train` | 281 | 参数拟合 | 不能代替 validation/selection |
| `validation` | 50 | early stopping、checkpoint 选择 | 不能选择最终架构 |
| `calibration` | 50 | 冻结模型后的后处理校准 | 不能回头改网络或损失 |
| `selection` | 50 | 执行预注册的优效/非劣/效率门 | 不能反复试规则直到显著 |
| `r_seen` | 300 | 保存为历史架构诊断证据 | 不拟合、不校准、不选模、不为新模型计分 |
| `final` | 0（本地） | 将来仅使用独立冻结的外部数据 | 禁止挪用任一本地角色 |

关系可以写成：

```text
731 个本地完整日
├── 300 天 R-SEEN
│   ├── 150 天 MM-JDWind/DDPM 诊断面板
│   └── 150 天 STGF/Time-domain/DDPM 诊断面板
└── 431 天 architecture-v1 开发池
    ├── train       281
    ├── validation   50
    ├── calibration  50
    └── selection    50

external final：另行寻找、注册和冻结；当前为 0 天
```

431 天使用固定 seed `20260815` 和固定的 `numpy.default_rng(PCG64).permutation` 分配。配置中同时冻结了完整日历、剩余日期、各角色日期及两个 `R-SEEN` 来源的 SHA256。这样做的目的不是“密码学安全”，而是让任何无意换日、漏日或角色漂移都立即失败。

还需注意：当前 431 天是确定性随机角色划分，不是连续时间块划分。日期间可能存在季节相关性，因此正式不确定性估计应以 calendar day 做模型间配对，并额外使用按月聚类的重采样结果做稳健性检查。代码 manifest 已记录每个角色的 `YYYY-MM` block。

---

## 3. 数据实现到底保证了什么

### 3.1 模型看到的一个样本

一个样本是一整天，而不是一个独立小时：

```text
condition: [day, zone=10, hour=24, feature=20]
target:    [day, zone=10, hour=24]
```

20 个条件特征由以下两部分组成：

- 10 个 NWP/派生气象量：`U10, V10, U100, V100, WS10, WS100, WE10, WE100, WD10, WD100`；
- 10 维 zone one-hot。

这让模型一次生成 10 个风电场、24 小时的联合场景，而不是事后拼接 240 个独立预测。

### 3.2 缺失目标的三份表示

每个 split 同时保留：

- `target_raw`：原始目标，缺失位置仍为 NaN；
- `target`：只在同 role、同 zone、同日内填充后的有限数组；
- `raw_missing_mask` / `observed_mask`：指出哪些格子是真实观测。

缺失格子的 atom state 被中性化为 interior，只是为了避免填充值制造虚假的 0/1 状态；它仍必须被 `observed_mask=False` 从 atom loss 和 flow loss 中排除。

### 3.3 标准化边界

- NWP standardizer 只在 `train` 上拟合；
- target standardizer 只使用 `train` 中原始可见的格子；
- validation、calibration、selection 和 `R-SEEN` 均不能贡献均值或方差。

### 3.4 已核实的数据 API

只构造和审核日期协议，不读测试目标：

```python
from architecture_v1.protocol import (
    build_architecture_protocol,
    write_protocol_manifest,
)

protocol = build_architecture_protocol(
    "repro_configs/architecture_v1.json",
    data_dir="Data",
    smoke=False,
)

print(protocol.manifest["full_date_counts"])
write_protocol_manifest(
    protocol,
    "outputs/architecture_v1/protocol.manifest.json",
)
```

构造完整数据 bundle：

```python
from architecture_v1.data import build_architecture_v1_data

bundle = build_architecture_v1_data(
    "Data",
    config_path="repro_configs/architecture_v1.json",
    smoke=False,
)

print(bundle.train.condition.shape)       # (281, 10, 24, 20)
print(bundle.train.target.shape)          # (281, 10, 24)
print(bundle.train.observed_mask.shape)   # (281, 10, 24)
print(bundle.train.ramp_observed_mask.shape)  # (281, 10, 23)
```

`bundle.role("final")` 会在外部 final 未注册时失败，这是预期的安全行为，不是待修 bug。

---

## 4. 共同模型底座：E、A 与连续生成器

为了让“换模型家族”真正有解释力，不能让 diffusion 用一套 NWP 特征、flow 用另一套 atom 后处理、SSM 又换一种输出含义。architecture v1 先定义共同底座：

```text
NWP + zone identity
        │
        ▼
共同编码器 E：JointNWPEncoder
        │
        ├──────────────► 共同 atom 模块 A
        │                 P(y=0), P(0<y<1), P(y=1)
        │                          │
        ▼                          ▼
候选连续/动态机制          有限成员状态 allocation
        │                          │
        └──────────┬───────────────┘
                   ▼
      exact 0 / continuous interior / exact 1
```

### 4.1 共同编码器 E

`JointNWPEncoder` 先把 20 维条件投影到共享隐空间，再交替做：

- 同一 zone 内的 24 小时注意力；
- 同一小时内的 10 zone 注意力；
- 前馈变换。

所以 `E` 本身已经编码跨小时和跨场站信息。后续比较 T0/T1 时，必须记住这不是“完全无时间信息”对“有时间信息”，而是“已有全局条件注意力”基础上，新增状态递推是否还有独立价值。

### 4.2 共同 atom 模块 A

风电功率归一化后存在精确边界值：

- `ZERO_STATE = 0`：精确 0；
- `INTERIOR_STATE = 1`：严格位于 `(0,1)`；
- `ONE_STATE = 2`：精确 1。

`ExactAtomModule` 包括四部分：

1. atom head：从共享 NWP 表征预测三分类概率；
2. state allocation：把解析概率变成有限 ensemble 的状态场；
3. active mask：只有 interior 格子进入连续生成器；
4. reconstruction：0/1 状态严格重建为 0/1，interior latent 才经过 sigmoid。

因此，新模型不再像旧的普通连续生成器那样，给边界位置保留一个小权重的高斯残差。atom 格子的连续 latent 和速度都必须严格为 0；连续 flow loss 只在“原始可见且为 interior”的格子上计算。

### 4.3 atom allocation 的局限必须主动汇报

当前 allocation 语义在代码中明确标记为：

```text
balanced_shared_priority_control
```

它采用 largest-remainder 方式给每个格子分配有限成员数，再用 seeded shared/local priority 保持成员身份和可复现性。它解决的是：

- ensemble 个数有限时，三类数量之和严格等于成员数；
- 实现出来的边际频率尽量接近解析概率；
- 相同 seed 得到完全相同的 allocation。

但它**没有**解决：

- 0 状态在不同小时之间应持续多久；
- 多个场站是否会共同进入/离开 0 状态；
- atom state 对 NWP regime 的动态响应；
- 真正的数据驱动联合状态转移规律。

尤其 shared priority 可能人为制造偏强的正相关。因此生成的 ensemble 应表述为“边际概率质量/成员计数守恒的有限成员集合”，不能写成独立同分布（IID）抽样，也不能把它当成“已经学会了联合 atom state field”。这里的“守恒”只指每个格子的三类成员数之和严格等于 ensemble size，绝不是风电系统中的物理质量守恒。正式报告必须同时保存和检查：

- `analytic_probabilities`：head 输出的解析三分类概率；
- `realized_probabilities`：有限 ensemble 实际实现的比例；
- `states` 与 `active_mask`；
- allocation seed 与 `balanced_shared_priority_control` 标签。

这正是后续 `T1-source` 或概率 SSM 可能真正贡献的地方之一。

---

## 5. R0 到底是什么，不是什么

当前类名：

```python
from architecture_v1.model import R0JointRectifiedFlow
```

R0 是：

- 10 zone × 24 hour 联合时域 Rectified Flow；
- 使用新的无泄漏数据协议；
- 使用共同 `E` 编码器；
- 使用完整 `A` atom head/allocation/mask/reconstruction；
- 连续 transport 只作用于原始可见的 interior 格子；
- velocity 网络对小时和 zone 做联合 axial attention。

R0 不是：

- 旧 STGF/Time-domain 训练脚本的逐行复刻；
- 旧 300 天面板上的再次确认实验；
- 论文历史数字的同义词；
- 已被正式 3-seed 验证的最终基线。

为什么不能把它叫“旧 STGF 复现”？因为它至少改变了数据角色、缺失值处理、atom 输出语义、连续 latent 的有效域以及共同 shell。若以后需要确认历史代码能否重现旧结果，应单列为 `legacy_r0_reproduction` 或“历史复现诊断”，并明确它只回答软件复现问题，不能与新的 architecture-v1 selection 证据混为一谈。

一句组会表述可以是：

> R0 不是为了守住旧论文架构，而是给后续机制比较提供一个干净、联合、可掩码、带精确边界语义的新参照点。

---

## 6. T0 的 “memoryless” 为什么容易误解

当前类名：

```python
from architecture_v1.model import T0MemorylessRectifiedFlow
```

T0 在 R0 的共同 shell 上新增 `ParameterMatchedTemporalCell`，但把 `use_memory=False` 固定下来。该 cell：

- 保留与未来 recurrent T1 相同的参数名称、形状和接口；
- 仍接受 `(input_t, previous_state)`；
- 但不执行 recurrent projection；
- 不把 `h[t-1]` 接入数值计算或 autograd graph；
- 因而上一时刻 hidden state 不可能给当前时刻传递信息。

它的作用是一个公平负对照：未来把同一 cell 切换为 recurrent 模式后，若结果改善，不能简单归因于“多了一层、参数更多、接口不同”。代码同时报告：

- total parameters；
- trainable parameters；
- effective active parameters。

最后一项会扣除 T0 中为参数匹配而保留、但 forward 根本没有调用的 recurrent projection。

最重要的限定是：

```text
T0 新增 cell 不递归
        ≠
整个 T0 网络没有跨时刻信息
```

原因如下：

```text
整条 [10 zone × 24 hour] 条件场
             │
             ▼
E 的 hour/zone axial attention   ← 已能看全日条件
             │
             ▼
T0 的新增 cell                   ← 每小时不读取 h[t-1]
             │
             ▼
RF velocity 的 hour/zone attention ← 仍能联合处理整条 24 小时轨迹
```

因此，T0 应解释为“**没有新增显式递归状态通道**”，而不是“时间独立模型”。未来 T1 检验的是在全轨迹注意力之外，显式递归状态是否提供额外价值。

---

## 7. 已核实的模型 API 和安全约束

### 7.1 建模与损失

```python
import torch
from architecture_v1.model import R0JointRectifiedFlow, T0MemorylessRectifiedFlow

common = dict(
    condition_dim=20,
    zones=10,
    hours=24,
    encoder_dim=64,
    encoder_depth=2,
    flow_dim=96,
    flow_depth=4,
    heads=4,
    ff_multiplier=4,
    dropout=0.0,
    atom_hidden_dim=64,
    atom_initial_probabilities=(0.08, 0.919, 0.001),
    atom_shared_priority_weight=0.5,
    logit_epsilon=1e-4,
)

r0 = R0JointRectifiedFlow(**common)
t0 = T0MemorylessRectifiedFlow(**common)

# condition: [B,10,24,20]
# observation/states/observed_mask: [B,10,24]
atom_terms = r0.atom_loss(
    condition,
    states=states,
    observed_mask=observed_mask,
)
flow_terms = r0.flow_loss(
    condition,
    observation,
    states=states,
    observed_mask=observed_mask,
)
```

`observed_mask` 是必需参数，不是可选装饰。填充位置即便数值有限，也不得进入 atom 或 flow 监督。

### 7.2 从 R0 复制并冻结共同 E/A

```python
t0.load_shared_from(r0, freeze=True)
print(t0.model_spec())
```

权重采用 copy，不是两个模型共享同一块可变 tensor。冻结共同 shell 后，`E/A` 会保持 eval 状态；需要联合微调时可调用：

```python
t0.unfreeze_shared()
```

冻结/解冻属于实验因素，正式比较前必须预先规定，不能根据 selection 表现临时调整。

### 7.3 生成联合场景

```python
batch = r0.sample(
    condition,
    members=100,
    steps=16,
    seed=20260815,
    method="heun",
    member_chunk=25,
)

scenarios = batch.values               # [B,100,10,24]
states = batch.states                   # [B,100,10,24]
analytic_p = batch.atom_statistics.probabilities
realized_p = batch.atom_allocation.realized_probabilities
```

Heun 的每条路径 NFE 为 `2 × steps - 1`，因此 16 steps 对应 31 次 velocity evaluation。`member_chunk` 只用于控制显存；同 seed 下结果应与 chunk 大小无关。代码区分：

- `per_path_nfe`：每条路径的算法 NFE；
- `batched_forward_calls`：考虑分块后实际发生的批量网络调用次数。

效率比较时两者都应记录，不能只报一个含糊的“采样步数”。

### 7.4 训练、checkpoint 与评估 API

当前已经有公共训练骨架，而不是只有模型 forward：

```python
from architecture_v1.training import ArchitectureBatch, ArchitectureTrainer

train_batch = ArchitectureBatch.from_split(bundle.train, [0, 1])
valid_batch = ArchitectureBatch.from_split(bundle.validation, [0, 1])
resolved_config = {
    "scientific_status": "manual_api_example_only",
    "candidate": "R0",
}

trainer = ArchitectureTrainer(
    r0,
    "outputs/architecture_v1/example",
    resolved_config=resolved_config,
    protocol_sha256=bundle.protocol.manifest["protocol_sha256"],
    data_bundle_sha256=bundle.manifest["data_bundle_sha256"],
    training_seed=0,
)

atom_result = trainer.fit_stage(
    "atom", [train_batch], [valid_batch],
    steps=2, learning_rate=1e-3,
)
```

这是底层 API 示例，不是推荐的完整实验脚本。stage 之间必须先校验 hash 并恢复 validation 选出的 `best.pt`；已落盘的 smoke runner 已实现这一顺序，实际使用时应优先调用 runner，避免手工漏掉 checkpoint 恢复。

这个 trainer 会：

- 在 atom stage 训练 `E + A`，在 flow stage 冻结 `E/A` 并训练 flow（T0 还包括新增 cell）；
- 每一步检查 loss、gradient、parameter 和 optimizer state 是否有限；
- 原子写入 `latest_safe.pt` 和 validation 选出的 `best.pt`；
- 给 checkpoint 保存 config/protocol/data-bundle hash sidecar；
- 发生非有限错误时保存 batch、traceback 和 failure manifest，而不是留下一个看似可用的半成品 checkpoint。

场景归档和 mask-aware 评估位于 `architecture_v1.evaluation`：

```python
from architecture_v1.evaluation import (
    write_scenario_archive,
    load_scenario_archive,
    evaluate_archive,
)
```

archive 会保存 scenarios、observations、两个缺失掩码、atom states、解析 zero/one 概率、日期、zone 及完整采样元数据。当前已实现的逐日指标是：

- observed-only level CRPS；
- 两端都 observed 的 ramp CRPS；
- zero Brier、one Brier 和三状态 atom Brier。

cross-lag、lagged variogram/VS 和 atom duration 仍属于正式跨模型诊断需要补齐的指标；不能因为当前 smoke summary 没有它们，就把问题改成只比较 CRPS。

---

## 8. Smoke 与正式 3-seed 实验必须分开

### 8.1 Smoke 回答什么

配置中的 smoke 视图为：

| 项目 | 数值 |
|---|---:|
| train | 8 天 |
| validation | 2 天 |
| calibration | 2 天 |
| selection | 2 天 |
| `R-SEEN` | 2 天，仅做协议/结构检查 |
| model seed | 1 个 |
| training updates | 每个已启用 stage 2 次 |
| ensemble members | 4 |

为了让 CPU smoke 边界明确，实际 runner 还把模型缩为 `encoder_dim=32, encoder_depth=1, flow_dim=32, flow_depth=1`，batch size 为 2，使用 FP32、单线程和 deterministic algorithms。这些覆盖值只是缩短连线测试，不是正式模型设定。

Smoke 只应验证：

- 日期 hash 和角色互斥是否成立；
- 目标文件是否在角色冻结后才读取；
- 缺失掩码是否进入两个损失；
- atom latent/velocity 在边界状态上是否严格为 0；
- 0/1 是否精确重建；
- 相同 seed 和不同 member chunk 是否产生相同场景；
- checkpoint、manifest、scenario archive 的字段和形状是否完整；
- CPU/GPU 上 forward、backward、sampling 是否有限。

Smoke **不能**回答：

- R0 是否比历史模型好；
- T0 是否改善 ramp；
- flow 是否优于 diffusion；
- 结果是否具备统计稳定性。

即使 smoke 视图保留了 2 个 `R-SEEN` 日期，它们也只能用于读取、shape、hash 等非评分检查，不能计算新模型优劣。

当前 smoke runner 默认只做只读 preflight：

```bash
python \
  repro_scripts/run_architecture_v1_smoke.py
```

只有显式加开关才会执行有写入的 CPU smoke：

```bash
python \
  repro_scripts/run_architecture_v1_smoke.py --execute-smoke
```

以上命令假定已经激活包含 NumPy/PyTorch 的项目环境；不要直接使用缺少项目依赖的系统 Python。

默认输出目录为 `outputs/architecture_v1/smoke_v1_seed0`。runner 拒绝覆盖非空目录，并把 `R-SEEN/final` 的拟合、生成和评分全部 hard-deny。执行路径只使用：

- train：两步 atom、两步 R0 flow、两步 T0 flow；
- validation：每步 checkpoint 选择；
- selection：2 天、4 成员、Heun 2 steps（每路径 NFE=3）的场景归档与 mask-aware smoke metric；
- calibration：本轮不使用；
- `R-SEEN/final`：不参与拟合、生成或评分；`R-SEEN` 的 smoke 子集只允许做协议/数组结构审计。

R0/T0 selection sampling 使用同一 atom allocation 和同一初始 noise，作为 common-random-number 配对；这只降低 smoke/未来公平比较中的 Monte Carlo 噪声，不会把有限成员当成统计独立重复。

### 8.2 正式 3-seed 回答什么

正式阶段至少应做到：

1. 在运行前冻结 3 个训练 seed、优化器、最大 epoch/step、early-stop 规则和 checkpoint 规则；
2. 每个候选使用完全相同的日期协议和预处理 manifest；
3. validation 只选择 checkpoint；
4. calibration 只做预先声明的后处理；
5. selection 对 3 个 seed 做配对比较和按月聚类的稳健性分析；
6. 保留单 seed 结果，不能只展示最好的 seed 或只汇报 seed 平均；
7. `R-SEEN` 不进入任何新模型评分；
8. external final 注册前，不出现“最终测试性能”字样。

三 seed 的意义不是机械地把一个数字平均三次，而是确认此前看到的现象是否只是随机初始化、batch 顺序或有限 ensemble allocation 的偶然结果。

当前配置已经把正式 seed 冻结为 `[0, 1, 2]`，并记录一个尚未授权执行的采样模板：100 members、Heun 16 steps、每路径 NFE=31、member chunk=10。它仍只是模板，正式 GPU 训练被明确阻断，直到两件事预注册完成：

1. formal `E/A` 策略：只按 atom label 预训练后冻结，还是为 `E` 增加 observed-interior 辅助/联合 flow 预训练；
2. selection 上各优效、非劣和效率门的数值阈值。

---

## 9. 建议的实验顺序

### 阶段 A：地基验收

先满足以下全部条件：

- 731 天完整且无重复；
- 300 个 `R-SEEN` 与四个开发角色零交集；
- 431 天恰好分成 `281/50/50/50`；
- 本地 final 为 0，访问 final 会 fail closed；
- 所有 hash 可重复；
- 原始 175 个缺失目标均保留在 mask 中；
- ramp mask 仅在相邻两格都原始可见时为真；
- 模型损失改变被 mask 的填充值后保持不变。

任意一项失败都应 **NO-GO**，不得开始正式训练。

### 阶段 B：R0 clean common-shell 的 3-seed 基线

目的不是追求最好分数，而是得到一个可重复的共同参照：

- 三个 seed 都能稳定训练，无 NaN/Inf；
- level、ramp、atom、跨场站和跨时刻指标没有明显单 seed 反转；
- 推理 NFE、批量 forward 次数、显存和耗时完整记录；
- 场景、状态、解析 atom 概率和 realized allocation 可追溯。

若 R0 自身不稳定，先定位数值或协议问题，不应立刻增加更复杂架构。

### 阶段 C：公平控制组

建议按机制逐层推进：

1. `T0`：新增参数匹配 cell，但不加入递归状态；
2. `S0`：如果要研究 trajectory-score adaptation，先做同预算、无状态机制版本；
3. `T1-feature`：显式状态空间只作为条件特征进入生成器；
4. `T1-source`：状态空间成为随机动态源，直接建模场景间状态演化；
5. `T1-shuffle`：打乱时间或 NWP 对齐的负对照。

这里最关键的比较不是简单的 `T1 > R0`，而是：

```text
T1-feature 或 T1-source
          对比
参数、接口和共同 E/A 都尽可能匹配的 T0
          并且
真实时间顺序的收益应明显强于 T1-shuffle
```

只有这样，改善才更可能来自真正的动态状态信息，而不是多了参数、不同初始化或一个更强的普通非线性层。

### 阶段 D：再决定大模型家族

只有阶段 C 证明“显式动态状态”确有独立价值后，再实现：

- `T2_joint_ddpm`：与共同 shell、参数量和采样预算尽量匹配的联合 DDPM；
- `T3_probabilistic_ssm`：不依赖 flow/diffusion 外壳的纯概率状态空间模型。

这一步才回答 diffusion、flow 或 SSM 谁更适合，而不是在 smoke 之后凭印象选一个。

---

## 10. Go / No-Go 门怎么定义

正式阈值必须在查看 selection 结果前写入配置或预注册文档。当前可以先冻结“判定结构”，随后根据业务量纲确定数值阈值。

| 门 | 主要问题 | 至少检查 | GO 的含义 | NO-GO 后做什么 |
|---|---|---|---|---|
| G0 协议门 | 数据证据是否干净 | hash、角色交集、缺失 mask、external final | 全部安全不变量成立 | 停止训练并修协议 |
| G1 可训练门 | R0 是否数值稳定 | 3 seeds、loss、梯度、NaN/Inf、checkpoint | 三个 seed 均完整结束 | 修训练，不加模型复杂度 |
| G2 基线稳健门 | 现象是否非单 seed 偶然 | seed 分布、paired day 差值、month cluster CI | 结论方向基本一致 | 增加诊断，不能宣称机制收益 |
| G3 动态机制优效门 | 显式状态是否改善核心动态 | ramp CRPS、lagged variogram/VS、cross-lag、atom duration | T1 相对 T0 达到预注册优效 | 暂停 SSM 扩展 |
| G4 保真非劣门 | 动态收益是否牺牲基本质量 | level CRPS、coverage/Winkler、atom Brier、边界频率 | 核心基础指标不劣于容忍界 | 查损失权衡或拒绝候选 |
| G5 负对照门 | 收益真来自时间/NWP 结构吗 | T1 对比 T1-shuffle | shuffle 后收益明显衰减 | 机制解释不成立 |
| G6 效率门 | 收益是否值得成本 | 参数、effective params、NFE、forward calls、耗时、显存 | 达到预注册性价比门 | 保留为诊断，不升级主线 |
| G7 家族选择门 | flow/DDPM/SSM 谁最合适 | 同 shell、同数据、同 ensemble、同调参预算 | 多指标与效率共同支持 | 保持未决或缩小问题 |

指标解释建议：

- **level CRPS**：每个时空格点的总体概率预测质量；
- **ramp CRPS**：相邻小时功率变化的概率预测质量；
- **lagged VS / cross-lag**：跨小时、跨场站联合结构是否正确；
- **atom Brier**：0/interior/1 状态概率是否校准；
- **atom duration**：连续处于 0 或 1 的持续时间分布是否真实；
- **NWP regime**：不同风速、风向或天气条件下，误差是否系统性变化。

一个模型不能仅凭某一个漂亮指标过门。例如 ramp 变好但 level CRPS、coverage 和 atom calibration 明显恶化，不应直接升级为主模型。

---

## 11. 最终如何在 flow、diffusion、SSM 之间作选择

当前没有选择，且不应提前选择。后续判断可遵循下面的证据逻辑：

### 情形 1：T1 相对 T0 没有稳定改善

说明在当前数据量和共同编码器下，显式递归/状态机制尚未显示独立价值。此时优先保留更简单的 joint flow，并检查：

- 诊断指标是否真的对动态缺陷敏感；
- 状态定义是否不合适；
- 样本量是否不足；
- atom allocation 是否掩盖了动态状态问题。

不能因为“SSM 理论上适合时序”就继续无限加复杂度。

### 情形 2：T1-feature 改善，但 T1-source 没有额外收益

说明显式状态作为确定性条件特征可能足够。可以把 state encoder 作为共同条件模块，再公平比较 flow 与 DDPM 的连续生成能力。

### 情形 3：T1-source 明显改善 ramp、cross-lag 和 atom duration

说明随机动态源可能是缺失机制。此时纯概率 SSM 或 SSM + flow/diffusion hybrid 值得进入主线，并重点比较状态可解释性、长序列扩展和采样效率。

### 情形 4：matched DDPM 在相同 shell 下稳定优于 flow

若优势通过 3 seeds、非劣门和效率门，且额外采样成本可接受，才有理由选择 diffusion。不能拿一个深度更大、调参更多、采样预算不同的 DDPM 与 R0 直接比较。

### 情形 5：纯概率 SSM 达到非劣且显著更高效

若它在 level/atom 指标非劣，同时在 ramp、lag、duration 或效率上有明确优势，则没有必要为了“生成模型新颖性”强行保留 flow/diffusion 外壳。

换句话说，研究路线是问题驱动的：

```text
先确认缺陷来自哪里
        ↓
再确认哪种机制能修复
        ↓
最后才选择最小且足够的模型家族
```

---

## 12. 组会上建议怎么讲

可以用下面五句话概括：

1. 过去两个 150 天面板已经影响了研究方向，所以合并成 300 天 `R-SEEN`，不再用于新模型评分。
2. 731 个本地日扣除 `R-SEEN` 后只剩 431 天，分成 281/50/50/50 的 train/validation/calibration/selection；真正 final 必须来自外部数据。
3. 新的 R0 是公平实验用的 clean common-shell 联合 flow，不是旧 STGF 的严格复现。
4. T0 只关闭新增 cell 的 `h[t-1]` 通道；整个模型仍有 24 小时联合注意力，因此它是“无新增递归状态”的参数匹配负对照。
5. 先做 smoke 和 R0/T0/T1 的 3-seed 机制诊断，通过预注册门后，才决定 diffusion、flow 还是纯 SSM。

如果被问“为什么现在不直接上一个更复杂的新模型”，可回答：

> 因为当前目标不是证明某个架构足够复杂，而是分离连续生成、显式动态状态和边界状态分配各自的贡献。没有公平控制组，模型变好也无法知道为什么变好。

---

## 13. 当前边界与下一次交付

当前代码已经落盘的底座包括：

- 冻结、可哈希、`R-SEEN` 隔离的日期协议；
- split-first 的目标缺失处理与原始 mask；
- 10-zone × 24-hour × 20-condition 的联合数据表示；
- 共同 NWP 编码器 `E`；
- 完整、精确边界的 atom 模块 `A`；
- R0 与参数匹配 memoryless-cell T0；
- interior-only flow loss、确定性有限成员 allocation、分块无关采样和 NFE 计数；
- 分 stage、有限值 fail-closed、hash 绑定的 trainer/checkpoint/failure bundle；
- 可校验的联合场景 `.npz` archive 与 observed-only level/ramp CRPS、atom Brier；
- 默认 dry-run、必须显式授权执行且无法升级为正式训练的 CPU smoke runner。

截至本说明更新时，完整的有写入 CPU smoke 已先在临时验收目录跑通，并按同一冻结配置持久化到 `outputs/architecture_v1/smoke_v1_seed0/`。它包括：R0 atom 2 步、R0 flow 2 步、复制并冻结共同 `E/A` 后的 T0 flow 2 步、selection 上 D=2/M=4/Heun-2/NFE=3 生成、archive 回读以及 mask-aware metrics；`completion.json` 的 SHA256 为 `7992266ce6c1b24a33a62ae7379106025b3e4f856e1cb296fe19e8f3bb422c3e`。这个结果只证明工程链路可运行，不是研究性能证据，也没有启动 formal 3-seed 训练。

下一次实验性交付应当是：

1. 把已跑通 smoke 的 completion、checkpoint、archive 与 hash 检查固化为回归验收项，后续底座代码变化后必须复跑；
2. 补齐正式选择所需的 cross-lag、lagged VS、atom duration 和 NWP-regime 指标；
3. 冻结 formal `E/A` 策略与所有 Go/No-Go 数值阈值；
4. 使用已登记的 `[0,1,2]` 三 seed 训练 R0 clean common-shell 基线；
5. 再进入 T0、T1-feature、T1-source 和 T1-shuffle。

在上述证据完成前，任何“最终模型选择”都应写成：

> **未决：diffusion、rectified flow 与概率 SSM 均保留为候选。**
