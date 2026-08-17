# SO101 World Model 项目总结

> 盘点基准：`causal-action-v2` 分支，commit `a2ba0d5`，2026-08-15。本文描述的是当前仓库代码、配置和本地 manifest 能够证明的状态；服务器上的数据、checkpoint 与实验输出是否存在另行说明。

## 1. 项目定位

本项目面向 SO101 机械臂视频预测，目标是在给定一段视觉历史和机器人动作后，在 SD3 VAE latent 空间中生成后续视频帧。当前主线不是直接预测 RGB，而是：

1. 把前视相机 RGB 编码成冻结的 SD3 VAE latent；
2. 用 action-conditioned causal DiT 学习 latent 的 Flow Matching velocity field；
3. 推理时从高斯噪声出发，用 Euler 积分逐帧生成 future latent；
4. 可选地通过同一个冻结 VAE 解码成图像，用于定性检查；
5. 通过真实动作与时间打乱动作的成对实验，评估模型是否真正使用动作条件。

仓库中同时保留了一套早期/legacy 调试与真实单 episode 实验代码。正式开发主线位于 `src/causal`、`configs/causal` 和对应的 causal 脚本中；它与 legacy 路径隔离，checkpoint 也有独立的 `causal_v2` 语义。

## 2. 当前实现的功能

### 2.1 数据读取与一致性检查

- 支持读取 LeRobot v3 数据布局：Parquet 中的 `action`、`observation.state`、`timestamp`、`frame_index` 和 episode 信息，以及共享 AV1 视频 shard 中由 metadata 界定的 episode 片段。
- 固定使用前视相机 `observation.images.front`，固定原始采样率 20 Hz。
- 检查 episode 边界、帧号连续性、时间戳单调性、20 Hz 帧间隔，以及表格时间轴和视频 PTS 的相对对齐；默认容差为 5 ms。
- 支持按 episode 范围和 shard ID 并行构建 cache，episode 的 VAE posterior seed 为 `base_seed + episode_id`，不受执行顺序影响。
- cache 采用临时文件加原子替换，已有结果默认跳过，可显式覆盖。

### 2.2 SD3 VAE latent cache

- VAE：`stabilityai/stable-diffusion-3-medium-diffusers` 的 `AutoencoderKL`，从本地目录加载。
- 输入预处理：RGB `uint8` → 最短边缩放到 256 → 256×256 中心裁剪 → `[0,1]` → `[-1,1]`。
- 输出 latent：每帧 `[16,32,32]`。
- 冻结的 latent convention：

  ```text
  z_cache = posterior.sample() * scaling_factor
  ```

  不使用 SD3 的 `shift_factor`。CUDA 正式 cache 使用 BF16，CPU 诊断回退使用 FP32。
- 每个 episode cache 保存 latents、6D actions、6D states、timestamps、frame indices、video timestamps 和完整 provenance metadata。
- 提供 cache 校验和首/中/末帧 VAE 解码检查工具。

### 2.3 数据集与动作条件

正式 causal Dataset 支持两种模式：

- `single_episode`：同一 episode 内用显式时间范围划分 train/val；
- `multi_episode`：通过 manifest 做 episode-level train/val/test 划分，窗口不会跨 episode。

时间窗口支持 `frame_stride ∈ {1,2,4}`。正式配置为 10 个 sampled frames、2 个 history frames、stride 4，因此：

- sampled latent 时间跨度是 `1 + (10-1)×4 = 37` 个 raw frame，约 1.8 秒；
- history 为 raw frame `s` 和 `s+4`；
- 8 个 future slots 参与训练 loss。

动作始终采用 causal alignment，第一个 temporal slot 是显式 NULL，且在 action embedding 后再次乘 validity mask，保证它是严格的零 action embedding，不受 Linear bias 影响。支持两种表示：

- `shifted_sampled`：slot `t` 使用前一个 sampled frame 对应的单个 6D action，effective action dim 为 6；
- `fast_chunk`：slot `t` 使用前一个 sampled frame到当前帧之间的全部 raw 20 Hz actions。stride 4 时一个 chunk 为 `[4,6]`，先逐个 6D action 归一化，再 flatten 成 24D。

动作归一化是固定的 raw-6D、逐维 z-score：

- mean/std 只从 train split 计算一次；
- train 与 val 使用完全相同的统计量；
- Dataset 不会自行计算统计量；
- checkpoint 同时保存数值和 provenance；
- evaluation 只读取 checkpoint 中的统计量，不重新估计。

### 2.4 模型结构

主模型为 action-conditioned、factorized spatial-temporal causal DiT。正式 300k recipe 的主要规格为：

| 项目 | 配置 |
| --- | --- |
| 输入/输出 | `[B,T,16,32,32]` latent velocity |
| Patch | 2×2 Conv patchify，得到每帧 16×16 个 token |
| Hidden size | 384 |
| Depth | 12 个 factorized blocks |
| Attention heads | 6，head dim 64 |
| Action dim | 24（stride 4 `fast_chunk`） |
| FFN | SwiGLU，`mlp_ratio=4.0`，实际内部宽度 1024 |
| Normalization | RMSNorm；Q/K 可选 RMSNorm，正式配置开启 |
| Position encoding | 空间 2D RoPE、时间 1D RoPE |
| Conditioning | sinusoidal tau embedding + Linear action embedding |
| 调制 | AdaLN-Zero shift/scale/gate |
| 时间约束 | temporal attention 使用 causal mask |
| 参数量 | 约 64.43M（正式 24D action 配置） |

每个 factorized block 依次执行：

```text
spatial self-attention（单帧内，非 causal）
→ SwiGLU
→ temporal self-attention（同一空间位置跨时间，causal）
→ SwiGLU
```

两个子块都由 timestep + action condition 做 AdaLN 调制并使用 gated residual。所有 block 的 AdaLN 输出层和最终预测层采用 zero initialization，使网络初始输出接近零并逐步学习 velocity field。

### 2.5 Flow Matching 训练目标

训练采用 OT linear Flow Matching。对 clean latent `z` 采样高斯端点 `ε`，future slot 独立采样 `τ ~ Uniform(0,1)`：

```text
z_τ = (1 - τ) z + τ ε
target velocity = ε - z
```

- history slot 的 `τ=0`，保持 clean；当前 `history_noise_std=0`；
- model forward 输入为 `z_τ`、`τ`、action condition 和 action validity mask；
- loss 是 model velocity 与 `ε-z` 的 MSE；
- 只对 future slots 求平均，history 不参与 loss；
- validation 使用相同 precision 和同一目标，只是固定随机种子。它是对加噪 ground-truth future latent 的 FM objective evaluation，不是自回归 rollout accuracy。

### 2.6 正式训练 recipe

`budget_stride4_chunk_300k.yaml` 对应当前正式主模型 recipe：

| 项目 | 值 |
| --- | --- |
| Seed | 42 |
| Batch size | 8 |
| Training steps | 300,000 |
| Precision | BF16 CUDA autocast |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Betas | `(0.9, 0.99)` |
| Epsilon | `1e-8` |
| Weight decay | `0.002` |
| Gradient clipping | global norm `1.0` |
| Scheduler | linear warmup + cosine decay |
| Warmup | 3%，9,000 updates |
| Minimum LR | base LR 的 70%，即 `7e-5` |
| EMA | 开启，decay `0.9995`，FP32 shadow model |
| Validation | 每 10,000 steps，固定均匀抽取最多 256 个 val windows |
| Checkpoint | best 与 last |
| DataLoader workers | 0 |

BF16 只允许在 CUDA 上运行，forward 与 loss 都位于 `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` 中；backward 直接调用 `loss.backward()`，不使用 GradScaler。FP32 则不进入 autocast。CPU 请求 BF16 会明确失败，不会静默降级。

训练 loop 开始前完成 config、Dataset、action stats、model、optimizer、scheduler 和 EMA 初始化。`elapsed_wall_seconds` 从 `model.train()` 前开始计时，包含训练循环和 validation，但排除上述初始化。日志每 25 step 输出 loss、grad norm、LR 和 elapsed wall time。

项目另有 50k/100k/200k/300k 四档 budget pilot，分别使用 1,500/3,000/6,000/9,000 warmup updates，其余正式 recipe 保持一致。四卡 launcher：

- 要求在 tmux 中运行；
- GPU 0/1/2/3 各跑一个 budget；
- 进程以 `nohup` 后台启动并启用 `PYTHONUNBUFFERED=1`；
- 启动前检查配置、非空输出目录和已有日志，防止覆盖；
- 最后 `wait` 四个进程并汇总退出状态。

### 2.7 Checkpoint 与可复现性

causal checkpoint 版本为 v2，保存：

- raw model state、optimizer state、scheduler state；
- 可选 EMA model state；
- 完整 resolved config，包括 precision；
- train-only action mean/std 及来源；
- data info、step、best validation loss、elapsed wall time；
- 当前 git commit。

评估时 checkpoint config 是 source of truth。外部 config 只能用于 compatibility validation，不能覆盖 checkpoint 的结构、时间、动作或 latent 语义。precision 被当作运行 provenance 而非模型兼容字段，因此历史 FP32 checkpoint 与相同结构的 BF16 请求不会被误判成结构不兼容；缺少 precision 的历史配置按 FP32 解释。

## 3. 推理与评估能力

### 3.1 FM objective evaluation

`scripts/evaluate_causal.py`：

- 支持 val/test split；
- 支持 raw 或 EMA weights，`auto` 优先选择 EMA；
- 使用 checkpoint-owned config/action stats；
- 可限制固定 evaluation windows 和多次 noise draw；
- 输出平均 future-only FM loss。

该指标用于检查训练 objective，不等价于自由 rollout 的长期预测能力。

### 3.2 自回归 latent rollout

`scripts/evaluate_rollout.py` / `src/causal/rollout.py`：

- 从 val/test split 的所有合法位置中确定性、均匀地选择 rollout cases；
- 每个 future step 从确定性的高斯噪声开始；noise seed 由全局 seed、episode、start、noise draw 和 step 哈希得到，因此加长 rollout 不会改变已有前缀；
- 复用统一 Euler sampler，从 `τ=1` 积分到 `τ=0`，默认 10 steps；
- 只有初始 history latent 是 GT，之后每个预测 latent 会反馈为下一步 context；
- future GT 只在完整生成之后读取并计算 metric，代码显式记录 `reference_used_as_model_input=False`；
- 超过训练窗口后保留最近 `num_frames-1` 个 clean/predicted context，再追加一个噪声目标，持续滑窗；
- 可并行多个独立 stochastic streams，但单条 rollout 的时间步仍严格顺序执行；
- 输出逐 rollout-step 和逐 step 聚合的 MSE、RMSE、MAE、relative L2、cosine similarity，并支持基于 MSE、MSE P90 或 relative L2 的连续有效 horizon threshold。

正式 validation rollout 协议在项目工作记录中定义为 `300k_val_R32_N128_D4`：300k EMA、128 个 physical cases、每 case 4 个 noise draws、32 个自回归 steps、stride 4、10 个 Euler steps、batch size 8、seed 20260，共 512 条 stochastic streams。当前 Windows 仓库未包含该评估目录，因而本文不填入无法从本地 artifact 复核的数值结果。

### 3.3 Rollout VAE 可视化

`scripts/visualize_rollout.py`：

- 从已有 rollout summary/CSV 选择 q10/q50/q90 或显式指定 stream；
- selection 使用正式 source metric，rerun 仅用于重新得到 latent，不会重选 case；
- 严格校验 checkpoint step、weights、precision、时间配置、动作配置和 latent convention；
- rerun 后释放 world model GPU 内存，再加载 VAE；
- 以相同的 `z_cache / scaling_factor`、no-shift convention 解码 GT cached latent 与 prediction；
- 输出单帧图、视频和 contact sheet；history 明确标为 shared history，future 区分 GT latent reconstruction 与 Prediction；
- 时间标签使用数据的 20 Hz FPS，不硬编码另一套采样率。

这里的 “GT latent reconstruction” 是 GT cached latent 经 VAE 解码的重建，不是原始 RGB。

### 3.4 Action controllability / action-shuffling negative control

`scripts/evaluate_action_controllability.py` / `src/causal/action_counterfactual.py` 实现 TRUE ACTION 与 TEMPORALLY SHUFFLED ACTION 的成对负对照：

- 复用正式 rollout evaluation 的 checkpoint、split、selected cases、noise draws、seed、Euler steps和 horizon；不会重新随机选 case；
- 显式验证所有 source cases 属于 summary 声明的 split；
- history transition action 保持不变，只修改最后一个 history frame 之后的 future action chunks；
- 每个 chunk 的 `[frame_stride,6]` 内部顺序和数值保持原样，只在 rollout step 维上做 permutation；
- permutation 是由 `shuffle_seed + episode_id + start` 决定的 deterministic derangement，所有位置都没有 fixed point；同一 physical case 的不同 noise draws 使用同一 permutation；
- R=1 无法 derange，会明确报错；随机 retry 达到上限时使用 deterministic cyclic-shift fallback；
- 只 clone actions，原 episode latent 等大对象保持共享引用，且不原地修改原 episode；
- TRUE/SHUFFLE 两个 branch 显式共享完全相同的 `initial_noises`，不是仅依赖“相同 seed”；
- 两个 variant 一起 batch，默认 `pair_batch_size=4`，最大有效 model batch 为 8；支持最后一个 partial batch；
- source TRUE rerun 会对齐 identity、target frame 和 metric；identity/target mismatch hard fail，小的 BF16 数值差异只计 warning。

核心指标定义为：

```text
true_mse_k    = MSE(pred_true_k, gt_factual_k)
shuffle_mse_k = MSE(pred_shuffle_k, gt_factual_k)
delta_mse_k   = shuffle_mse_k - true_mse_k
```

`delta_mse > 0` 表示真实动作比被打乱的动作更符合 factual future。另有 `MSE(pred_true, pred_shuffle)` 衡量 action sensitivity。必须注意：shuffled branch 没有 counterfactual GT，因此它与 factual GT 的差异只是 action-corruption negative control，不是 counterfactual prediction accuracy，也不能单靠 prediction divergence 证明因果正确性。

输出包括：

- `per_rollout_step.csv`：逐 episode/start/draw/step 的 true、shuffle、delta、relative L2、cosine 和 prediction divergence；
- `per_rollout.csv`：每条 stochastic rollout 的 mean/final error、mean divergence、true-better step fraction；
- `per_step.csv`：跨 stochastic streams 的 mean/median/P90/P10 和 true-better rate；
- `per_case.csv`：先在同一 `(episode_id,start)` 内平均 noise draws；
- `summary.json`：总体 true/shuffle/delta、aggregate relative degradation、stochastic/case true-better rate、prediction divergence、rerun warning、协议与解释字段；
- 每 case 的 shuffle audit：permutation、修改 raw action span、fixed points、history unchanged 和 normalized chunk L2 difference。

95% CI 使用 deterministic case-level bootstrap：先将同一 physical case 的 4 个 noise draws 平均，再以 physical case 为 resampling unit。正式协议应使用 128 cases，而不是把 512 个相关 stochastic draws 错当成 512 个独立场景。

### 3.5 Action controllability 可视化

`scripts/visualize_action_controllability.py`：

- 从 physical-case `mean_delta_mse` 选择 q10/q50/q90；
- 每个 case 选择最接近 case mean delta 的代表 draw，平局取较小 draw ID；
- source result tables 会检查 physical case、stochastic identity、draw 范围、step 范围、总行数、duplicate/missing/extra；
- TRUE 与 SHUFFLE rerun 都会核对 target、Gaussian noise、permutation和 metric；身份类不一致 hard fail，BF16 小数值差异 warning only；
- metadata 分开保存 source metrics 与 rerun metrics，不因 rerun 漂移改变 selection；
- contact sheet 四行分别为 Original RGB、GT latent reconstruction、True Action Prediction、Shuffled Action Prediction；共享 history 标明为原始 RGB；
- raw-video reader 延迟加载，降低 Windows 上运行 `--help` 对 PyAV/Pandas/PyArrow 的依赖。

## 4. 使用的数据

### 4.1 数据集身份与内容

代码使用 Hugging Face 数据集 `armnet/armnetbench_v01_lerobot_so101`，本地 manifest 记录的数据集名为 `armnetbench_v01_lerobot_so101`。正式 manifest 期望：

- LeRobot codebase version：v3.0；
- episode IDs：严格为 0..2498，共 2499 个；
- 8 个 task IDs（0..7）；
- 每个 episode 只对应一个 task；
- 20 Hz front RGB；
- 每帧 6D action 和 6D observation state。

### 4.2 正式 split

manifest 使用 seed 42、task-stratified、episode-level 80/10/10 划分。每个 task 内先打乱再分别 floor train/val，余数进入 test；因此全局数目不是简单四舍五入：

| Split | Episodes |
| --- | ---: |
| Train | 1999 |
| Validation | 248 |
| Test | 252 |
| Total | 2499 |

三个 split 无 episode 重叠，union 必须恰好覆盖 manifest 的全部 episode。这样可以避免同一 episode 的相邻窗口同时出现在训练和验证/测试中。

### 4.3 当前本地数据状态

当前 Windows 工作区仅发现：

```text
data/causal/armnetbench_so101_seed42_manifest.json
```

manifest 中的真实路径是 Linux 服务器路径：

```text
dataset: /data/x2227/datasets/armnetbench_v01_lerobot_so101
cache:   /data/x2227/datasets/armnetbench_so101_latent_cache
```

本地没有 episode cache、SD3 VAE 权重、world-model checkpoint 或 rollout/action-eval 输出。也就是说：代码和正式 manifest 已在仓库工作区，原始数据与训练产物仍需在 Linux 数据盘上使用，当前 Windows 目录本身不能直接开始正式训练或复核正式指标。

## 5. Linux GPU 服务器运行条件

代码设计目标明确包含 Linux CUDA 服务器/A100：

- 训练入口支持 `--device cuda`；
- 正式 recipe 使用 CUDA BF16 autocast，适合 A100；
- VAE cache 的 CUDA dtype 也是 BF16；
- 4-GPU budget launcher 是 Bash + tmux + nohup，每张卡启动一个独立单卡训练，不是 DDP；
- rollout 和 action negative-control 支持 batch 并行，时间维仍自回归顺序执行。

但运行前需满足以下条件：

1. 按 `requirements.txt` 准备与正式服务器一致的 Python/CUDA 环境，并确认 NVIDIA driver 支持 CUDA 12.8；
2. 准备完整 LeRobot v3 数据、SD3 VAE 本地权重、2499 个 episode latent cache；
3. 确认 manifest 内的绝对路径与服务器一致；
4. 验证 PyTorch、PyAV、PyArrow、Diffusers 和 VAE 本地加载路径；
5. A100 支持 BF16，但其他 CUDA GPU 是否适合需按硬件能力确认；CPU 只能使用 FP32。

当前 `requirements.txt` 已更新为正式服务器环境的完整版本快照，主要版本包括：

- PyTorch `2.9.1+cu128`、Triton `3.5.1` 和 CUDA 12.8/cuDNN 9.10 对应组件；
- Diffusers `0.39.0`、Hugging Face Hub `1.27.0`、Safetensors `0.8.0`；
- NumPy `2.2.6`、Pandas `2.3.3`、PyArrow `25.0.1`、PyAV `17.1.0`；
- Pillow `12.3.0`、PyYAML `6.0.3`、pytest `9.1.1` 及其传递依赖。

该文件更接近 Linux/A100 服务器的 `pip freeze`，而不是跨平台的最小依赖集合：它包含 NVIDIA CUDA wheel，并且 `packaging` 当前记录为服务器 Conda 构建目录的 `file:///home/conda/...` 路径。因此它可以作为正式服务器版本基线，但复制到新的 Linux 主机或 Windows 时仍应先处理这条本地路径并使用合适的 PyTorch wheel source；不应直接把这套 CUDA requirements 当成 Windows 环境文件。`requirements-dev.txt` 仍保留宽松的 `pytest>=8.0` 开发约束，与正式快照中的 pytest 9.1.1 兼容。

## 6. 测试覆盖与工程保护

当前测试覆盖的主要方面包括：

- attention、DiT shape/causality、Flow Matching；
- cache preprocessing、latent convention、20 Hz 和视频/表格对齐；
- manifest 完整性、split 不重叠、task-stratified split；
- causal action alignment、stride、sampled/chunk equivalence、NULL mask；
- train-only normalization、checkpoint source of truth 和 compatibility；
- FP32/BF16 config、autocast 和 CPU rejection；
- AdamW/scheduler/EMA recipe、budget pilot YAML/launcher/summarizer；
- rollout deterministic noise、sliding window、batch/partial batch、GT 不泄漏；
- rollout selection、VAE decode convention、contact sheet；
- action derangement、future-only action modification、common random numbers、case aggregation、bootstrap；
- action visualization source-table integrity、source/rerun metric 区分、raw RGB frame-index mapping。

## 7. 目录与入口速查

| 路径 | 作用 |
| --- | --- |
| `src/causal/` | 正式 causal model、config、runtime、checkpoint、rollout、counterfactual、visualization |
| `src/causal/data/` | causal single/multi episode Dataset、action condition、normalization、manifest |
| `src/so101_cache/` | LeRobot v3 reader、VAE cache convention 与校验 |
| `src/models/` | DiT、factorized attention、action embedder |
| `src/diffusion/` | Flow Matching batch 与 loss |
| `src/inference/` | checkpoint adapter、action adapter、Euler sampler、legacy/causal inference |
| `configs/causal/` | causal 实验配置与正式 budget recipes |
| `scripts/cache_so101_latents.py` | 构建正式 raw-20Hz SD3 VAE cache |
| `scripts/build_so101_causal_manifest.py` | 构建 2499 episode 正式 split manifest |
| `scripts/train_causal.py` | 正式 causal training 入口 |
| `scripts/evaluate_causal.py` | fixed-window FM objective evaluation |
| `scripts/evaluate_rollout.py` | 自回归 latent rollout evaluation |
| `scripts/visualize_rollout.py` | rollout VAE 可视化 |
| `scripts/evaluate_action_controllability.py` | true vs shuffled action negative control |
| `scripts/visualize_action_controllability.py` | action controllability 四行可视化 |
| `scripts/run_budget_pilot_4gpu.sh` | tmux 内四卡 budget pilot launcher |
| `scripts/summarize_budget_pilot.py` | best/last/训练时长汇总，不自动选择预算 |
| `tests/` | 单元与 regression tests |

## 8. 当前结论

到目前为止，项目已经形成了较完整的 causal latent world model 研发链路：原始 SO101/LeRobot v3 数据校验 → 冻结 SD3 VAE cache → episode-level split → causal action condition → DiT + Flow Matching 训练 → checkpoint provenance → fixed-window objective evaluation → 自回归 latent rollout → VAE 可视化 → action-shuffling controllability negative control。

当前最成熟的正式方案是 300k、EMA、BF16、batch 8、stride 4 `fast_chunk` 的约 64.43M 参数 causal DiT。工程上对 action alignment、train-only normalization、latent no-shift convention、checkpoint source of truth、GT future 不泄漏、common random numbers 和 case-level bootstrap 都有显式实现与测试。

当前主要缺口不在核心训练/评估逻辑，而在 artifact 分发和跨机器环境可移植性：Windows 工作区没有数据/cache/checkpoint/result；服务器依赖版本已经锁定，但 requirements 含 Linux CUDA wheel 和一条服务器本地 `file://` 路径。要在新 Linux A100 服务器上复现，仍需准备外部数据和 VAE 权重，确认 driver/CUDA 兼容性，并处理这条非可移植依赖记录。
