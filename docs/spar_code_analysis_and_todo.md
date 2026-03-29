# SPAR（Structure-Prior Aware Refinement for DAL）代码层工作拆解

## 1) 当前 DAL 代码结构与可插入点

### A. 检测器入口（`mmdet3d/models/detectors/dal.py`）
- `DAL.forward_train()` 目前将特征组织为 `[img_feats, pts_feats, img_feats_bev]`，再送入 `forward_pts_train`，并额外计算图像辅助分类损失。  
- 这意味着：
  - SPAR-P（proposal refinement）和 SPAR-C（classification refinement）最合适落在 `DALHead` 内部，不需要改 detector 主流程。
  - “回归分支不变”可以在 `DALHead.forward_single()` 中严格控制。

### B. 关键头部（`mmdet3d/models/dense_heads/dal_head.py`）
当前 `DALHead.forward_single()` 已天然分为三段：
1. **Dense proposal 阶段**：`dense_heatmap = self.heatmap_head(dense_fuse_feat)`，随后 `extract_proposal(heatmap)` 选 top-K。  
2. **Regression 阶段**：仅用 `query_feat_lidar` 做 `height/center/dim/rot/vel` 预测。  
3. **Classification 阶段**：拼接 `query_feat_lidar + query_feat_img + query_feat_img_bev`，再走 `self.fuse_convs` 与分类头。  

> 这个结构与 SPAR 目标高度匹配：SPAR-P 可插在 dense heatmap 前后，SPAR-C 可插在 classification 融合前后，而 regression 可完全冻结其输入与逻辑。

---

## 2) SPAR-P（Proposal refinement）建议的最小实现

## 2.1 新增模块（DALHead 内）
建议新增参数（默认关闭，不影响 baseline）：
- `use_spar_p=False`
- `spar_p_mid_channels`（默认等于 `hidden_channel`）
- `spar_p_alpha=0.5`

新增子模块：
- `self.spar_p_refiner`：
  - Conv2d(3x3) + BN + ReLU
  - Conv2d(3x3) + BN + ReLU
  - 轻量 residual block（或 1 个 bottleneck）
- `self.spar_p_out`：输出通道为 `num_classes` 的 prior map `M_sp`

## 2.2 前向注入位置
在 `dense_heatmap` 生成后执行：
- `M_sp = self.spar_p_out(self.spar_p_refiner(dense_fuse_feat))`
- `dense_heatmap_refined = dense_heatmap + spar_p_alpha * M_sp`
- proposal 使用 `dense_heatmap_refined.detach().sigmoid()`

训练阶段 heatmap loss 建议默认替换为 refined heatmap（主版本）；保留可选开关以对比“只影响 proposal，不影响 dense loss”的变体。

## 2.3 需要保存的中间量（用于分析）
在返回字典加可选字段（仅调试/可视化阶段开启）：
- `dense_heatmap_raw`
- `dense_heatmap_refined`
- `spar_prior_map`

---

## 3) SPAR-C（Classification refinement）建议的最小实现

## 3.1 新增模块（DALHead 内）
建议新增参数：
- `use_spar_c=False`
- `spar_c_kernel=3`（可做 3/5 消融）
- `spar_c_pool_modes=('max','avg')`
- `spar_c_mlp_channels`（例如 `[hidden_channel, hidden_channel]`）

新增子模块：
- `self.spar_c_ctx_mlp`：用于将局部上下文特征映射到分类融合维度。

## 3.2 局部上下文抽取方法
在 top-K proposal 已经得到后，基于 `bev_feat_lidar` 或 `dense_fuse_feat`：
- 把 `top_proposals_index` 还原为 `(x, y)` 网格坐标。
- 在对应 BEV 特征图上做局部窗口池化（3x3 或 5x5）：
  - max pooling
  - avg pooling
  - concat(max, avg)
- 得到 `f_ctx^(k)`，形状 `[B, C_ctx, K]`。

## 3.3 分类特征拼接
把现有
`[query_feat_lidar, query_feat_img, query_feat_img_bev]`
改为
`[query_feat_lidar, query_feat_img, query_feat_img_bev, query_feat_ctx]`，
再接 `self.fuse_convs` 与分类头。

> 注意：SPAR-C 不改 proposal 生成；M2 实验必须保证 `extract_proposal` 流程与 baseline 一致。

---

## 4) “回归分支不变”在代码中如何保证

为满足论文硬约束，建议明确以下规则：
1. `for task in ['height','center','dim','rot','vel']` 的输入仍然是 `query_feat_lidar`。  
2. 不把任何 SPAR-P / SPAR-C 输出拼进回归输入。  
3. regression 的 loss/target/coder 配置不改。  
4. 在配置和文档中写明 `regression_isolation=True`（实验开关/注释即可）。

---

## 5) 配置层需要新增的工作

在 `configs/dal/` 下新增 3 套配置（先 tiny，再 base）：
- `dal-tiny-spar-p.py`
- `dal-tiny-spar-c.py`
- `dal-tiny-spar-pc.py`

建议方式：复制 `dal-tiny.py` 并仅修改 `pts_bbox_head` 中的 SPAR 参数。  
后续再镜像到 `dal-base-*`。

同时补充实验开关：
- `analysis.save_proposal_stats=True/False`
- `analysis.save_heatmap_vis=True/False`
- `analysis.proposal_topk_list=[50,100,200,300]`

---

## 6) 评估与日志统计代码工作

当前仓库默认输出 nuScenes 官方指标；SPAR 还需要中间统计，建议新增脚本：
- `tools/analysis/spar_eval_proposal.py`
- `tools/analysis/spar_eval_cls.py`
- `tools/analysis/spar_scene_breakdown.py`

至少实现以下统计：
1. proposal recall@K / precision@K。  
2. proposal 已命中条件下 classification accuracy。  
3. small/medium/large，near/mid/far 分桶 AP/Recall。  
4. 误检区域占比（free-space/sparse LiDAR 区域）。

实现方式建议：
- 在 `DALHead` 推理时输出 `top_proposals_index/top_proposals_class/top_scores`（仅 analysis 模式）
- 脚本离线读取预测与 GT 做匹配并统计。

---

## 7) 可视化支持工作

建议新增：
- `tools/visualization/vis_spar_heatmap.py`
- `tools/visualization/vis_spar_proposals.py`
- `tools/visualization/vis_spar_ctx.py`

输出内容：
- raw vs refined heatmap
- top-K proposal 空间分布
- classification context 区域热力图
- 成功/失败案例集（远距、小目标、遮挡）

---

## 8) 分阶段落地顺序（代码实现视角）

### 阶段 A（M0）
- 不改模型，仅跑通 DAL-Tiny baseline。
- 确认 train/eval/日志导出脚本可复现。

### 阶段 B（M1：SPAR-P）
- 只加 SPAR-P 模块与开关。
- 先不做 attention，先验证 recall@K 提升。

### 阶段 C（M2：SPAR-C）
- 只加局部上下文池化与分类拼接。
- 验证“proposal 不变前提下”分类精度提升。

### 阶段 D（M3：SPAR-PC）
- 合并两分支并联合微调。
- 检查 mAP/NDS 增益是否来自 proposal/classification，且 mATE/mASE/mAOE 稳定。

---

## 9) 风险点与规避建议

1. **Top-K 固定策略可能掩盖 SPAR-P 增益**  
   - 加 `threshold + top-K` 消融，避免远距离小目标被挤出。

2. **SPAR-C 若直接拼接高维上下文，易拖慢速度**  
   - 使用低维投影（1x1 Conv/MLP）并控制 `C_ctx`。

3. **误改 regression 分支导致“收益来源不干净”**  
   - 单元测试断言回归输入维度和路径与 baseline 完全一致。

4. **分析脚本与训练代码耦合过深**  
   - 统一把统计字段放到 `analysis_mode` 开关下，默认关闭。

---

## 10) 本次分支状态

- 已创建开发分支：`SPAR`
- 当前提交仅包含“代码级任务分析与实施清单”文档，用于下一步逐项开发与实验。
