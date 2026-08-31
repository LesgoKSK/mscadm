# Architecture-v1 formal-v2.2.1 工程修订记录

冻结日期：2026-08-18

## 结论

formal-v2.2 在 seed 0 完成 E/A 阶段、尚未执行首个 retained flow optimizer update 时中止。中止原因是 runner 读取不存在的门字段 `preclip_gradient_norm_max`，而冻结门文件中的实际字段为 `preclip_gradient_norm_max_over_all_flow_updates`。

本次中止不是 G0/G1 科学门失败，也没有访问 calibration 或 selection。formal-v2.2 的部分 E/A 权重、优化器状态和 RNG 状态均不得复用。

## v2.2.1 唯一代码修复

runner 将正式 flow 梯度审计的最大范数阈值读取键改为已经冻结的 `preclip_gradient_norm_max_over_all_flow_updates`。

以下内容不变：

- 267/50/50/50 数据角色与 314 日隔离集；
- R0 模型结构和 E/A 定义；
- 三个训练随机种子和三个采样随机种子；
- E/A 与 flow 的优化器、学习率、epoch 上限和早停规则；
- gradient clip 候选值 5.0；
- P0、G0、G1 的所有数值门；
- 共同随机数 member-chunk 审计；
- calibration 与 selection 的封存状态。

## 重新开始规则

formal-v2.2.1 必须重新执行三随机种子 train-only P0、CUDA 容量预检，并从全新初始化开始三随机种子 retained R0。任何 formal-v2.2 权重和状态都不得复用。

冻结文件：

- `repro_configs/architecture_v1_formal_v2_2_1.json`
- `repro_configs/architecture_v1_go_no_go_v2_2_1.json`

