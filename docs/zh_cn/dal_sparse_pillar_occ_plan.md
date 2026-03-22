# 基于 DAL 的稀疏 Pillar Occupancy 与反馈优化方案

## 1. 目标

本文档用于沉淀一套以 **DAL** 为基础的研究与实现方案，目标是在现有 BEV 感知框架上，增加面向占据预测（occupancy）的稀疏 pillar 表征，并进一步探索利用 occupancy 结果对检测或 BEV 表征进行反馈优化的训练策略。

整体设计遵循以下原则：

- **优先复用 DAL 现有骨架**，尽量减少对训练、数据流和部署流程的侵入。
- **先验证稀疏 pillar occupancy 本身是否有效**，再逐步叠加反馈分支，避免一次性引入过多变量。
- **兼顾研究可解释性与工程可实现性**，每个阶段都保持清晰的输入、输出、损失函数和评测指标。

## 2. 总体研究问题

围绕如下三个核心问题展开：

1. 能否在 DAL 的 BEV 特征之上构建一个轻量、稳定的稀疏 pillar occupancy 分支？
2. occupancy 分支是否能在较低额外计算开销下，提供对场景几何结构更强的监督？
3. occupancy 预测结果或其中间特征，是否能够反向提升检测任务或整体 BEV 表征质量？

## 3. 基础方案：DAL + 稀疏 Pillar Occupancy Head

### 3.1 输入与特征来源

推荐优先从 DAL 已有的 BEV 特征图出发构建 occupancy head，而不是重新设计一条完整的 3D 体素主干。这样做有几个好处：

- 可以最大化复用现有图像编码与时序融合模块。
- 占据分支与检测分支天然共享 BEV 表征，便于后续研究多任务收益。
- 训练与调试成本更低，更容易定位收益到底来自 occupancy 监督还是来自结构改动。

建议将 occupancy head 的输入定义为：

- 主干网络输出的单尺度或多尺度 BEV feature map。
- 若实现复杂度允许，可额外拼接时间维聚合后的历史 BEV 特征。

### 3.2 稀疏 pillar 表达

“稀疏 pillar occupancy” 的核心思想是：

- 在 BEV 平面上使用规则 pillar 网格。
- 在高度方向上只保留有限个离散 bin，或者采用压缩后的高度编码。
- 只对前景候选区域或有效可见区域计算监督与预测，从而降低密集 3D occupancy 带来的显存与算力成本。

可选的两种建模方式：

#### 方案 A：Pillar + 多高度 bin 分类

对每个 BEV pillar 预测一个长度为 `H` 的 occupancy 向量，每个维度表示对应高度 bin 是否被占据。

优点：

- 与标准 occupancy 定义接近，监督直观。
- 更容易与已有体素或占据标注对齐。

缺点：

- 高度 bin 增加后，类别不平衡会比较明显。

#### 方案 B：Pillar 压缩占据编码

对每个 pillar 只预测：

- 是否存在占据；
- 占据高度范围；
- 或少量统计量（例如最低/最高占据高度、占据概率）。

优点：

- 更轻量。
- 更适合作为检测任务的辅助监督。

缺点：

- 表达能力弱于完整高度离散化。
- 与标准 occupancy 指标对齐时可能需要额外转换。

建议实验顺序为：**先做方案 A，验证后再根据成本切换或补充方案 B**。

## 4. 网络结构建议

### 4.1 最小可行版本

最小实现建议如下：

1. 保持 DAL 主干与检测头不变。
2. 在共享 BEV feature 上新增一个轻量 occupancy head。
3. occupancy head 采用若干层 `Conv-BN-ReLU`，最后输出 `C_occ` 个通道。
4. 将输出 reshape 为 `B x X x Y x H` 或 `B x H x X x Y` 的 occupancy logits。

该版本的价值在于：

- 改动范围小；
- 易于与原始 DAL baseline 直接对比；
- 可以快速验证 occupancy supervision 是否提供稳定收益。

### 4.2 增强版本

在最小版本稳定后，可继续尝试：

- **多尺度 occupancy 预测**：对不同分辨率 BEV 特征分别施加辅助 occupancy loss。
- **时序 occupancy 融合**：使用历史帧 BEV 特征提升遮挡区域占据预测。
- **稀疏监督掩码**：只在可见区域、标注可信区域或前景附近进行损失计算。

## 5. 标注与监督构建

### 5.1 标注来源

如果已有体素化 occupancy GT，可直接映射到 pillar 网格；否则可考虑从如下信息近似生成：

- LiDAR 点云累计投影到 BEV pillar 与高度 bin；
- 3D box 内部区域的弱占据近似；
- 多帧点云融合得到更完整的 pseudo occupancy 标注。

建议优先级如下：

1. **真实 occupancy / voxel GT**（如果数据集提供）；
2. **多帧 LiDAR 融合 pseudo GT**；
3. **3D box 弱监督近似 GT**。

### 5.2 正负样本与掩码

occupancy 监督最容易遇到的问题是正负样本极度不平衡，因此建议配套使用：

- 可见区域掩码；
- 忽略未知区域（unknown）的三值监督；
- 对正样本进行重加权；
- 或使用 focal loss / asymmetric loss。

## 6. 损失函数设计

建议从简单到复杂逐步推进。

### 6.1 基础损失

对于离散高度 bin occupancy，可优先使用：

- **Binary Cross Entropy / BCEWithLogitsLoss**；
- 或 **Focal Loss** 以缓解类别不平衡。

整体损失可写为：

```text
L = L_det + λ_occ * L_occ
```

其中：

- `L_det` 为 DAL 原有检测损失；
- `L_occ` 为 occupancy 分支损失；
- `λ_occ` 为 occupancy 权重，建议从较小值开始搜索，例如 `0.1 / 0.5 / 1.0`。

### 6.2 可选正则项

后续可以加入：

- **空间平滑正则**：鼓励相邻 pillar 的预测更连续；
- **时间一致性正则**：约束相邻帧 occupancy 在 ego-motion 对齐后保持一致；
- **结构先验正则**：例如对地面、高空稀疏区域施加不同权重。

## 7. 反馈优化（Feedback）方案

在确认 occupancy 分支本身有效后，再研究“反馈”机制。建议按风险由低到高分三步走。

### 7.1 Feedback-I：仅特征辅助

使用 occupancy head 的中间特征，通过轻量融合模块回注到检测 head 输入特征中，例如：

- concat 后接 `1x1 conv`；
- attention/gating；
- 残差式特征增强。

特点：

- 风险最低；
- 不依赖离散化后的 occupancy 预测结果；
- 更适合作为第一版 feedback。

### 7.2 Feedback-II：使用 occupancy logits/probability 作为显式先验

将 occupancy 概率图转成显式先验，对检测分支进行调制，例如：

- 作为 spatial prior 与 BEV feature 相乘；
- 作为 query / anchor 的筛选依据；
- 作为前景区域增强掩码。

特点：

- 可解释性强；
- 但如果 occupancy 预测早期不稳定，可能反而伤害检测训练。

### 7.3 Feedback-III：双向联合优化

进一步探索检测结果反向帮助 occupancy，例如：

- 将检测框或目标中心热力图作为 occupancy 分支先验；
- 建立 occupancy 与 detection 的一致性损失。

这一阶段研究价值高，但工程复杂度也最高，建议放在最后。

## 8. 实验阶段划分

建议采用如下里程碑式推进：

### Stage 0：DAL baseline 复现

- 固定训练配置与评测流程；
- 获得可靠的检测 baseline；
- 记录速度、显存、mAP/NDS 等基础指标。

### Stage 1：加入 occupancy head，不做 feedback

目标：回答“occupancy supervision 是否对共享 BEV 表征有帮助”。

需要报告：

- 检测指标变化；
- occupancy 指标；
- 训练稳定性；
- 计算开销变化。

### Stage 2：加入 feature-level feedback

目标：验证“occupancy 中间特征是否能反哺检测”。

需要重点对比：

- 与 Stage 1 相比检测是否进一步提升；
- 是否带来额外不稳定；
- 最优融合位置在哪里。

### Stage 3：加入 explicit occupancy prior feedback

目标：验证“显式 occupancy 预测结果是否能提供更强先验”。

这一阶段建议做更细的消融，因为收益和风险都比较明显。

## 9. 关键评测指标

### 9.1 检测任务

保持 DAL 原任务指标不变，例如：

- mAP；
- NDS；
- 各类别 AP；
- 推理速度与显存占用。

### 9.2 Occupancy 任务

建议至少统计：

- voxel / pillar occupancy IoU；
- precision / recall / F1；
- 按距离分段的 occupancy 指标；
- 按高度 bin 分段的 occupancy 指标。

### 9.3 联合分析

重点观察：

- occupancy 提升是否与检测提升正相关；
- 对远距离、小目标、遮挡目标是否帮助更明显；
- 是否存在“occupancy 更好但 detection 下降”的现象。

## 10. 关键消融实验

建议至少做以下消融：

1. **有无 occupancy 分支**；
2. **不同高度 bin 数**；
3. **不同 occupancy loss 权重**；
4. **BCE vs Focal Loss**；
5. **稀疏监督掩码策略**；
6. **仅共享训练 vs 加 feedback**；
7. **feature feedback vs probability feedback**；
8. **单帧 vs 时序 occupancy**。

## 11. 主要风险与对应缓解

### 风险 1：监督噪声大

若 occupancy GT 来自 pseudo label，多帧累积误差和动态目标误差都会影响训练。

缓解建议：

- 先在静态区域或高置信区域训练；
- unknown 区域不计 loss；
- 从 box 内弱监督做 warm-up 也是可行策略。

### 风险 2：类别极度不平衡

大量空白 pillar 会让模型偏向预测全空。

缓解建议：

- focal loss；
- 正样本重加权；
- hard negative mining；
- 只在 ROI 或可见区域内统计损失。

### 风险 3：feedback 过早引入导致主任务退化

若 occupancy 分支尚未收敛，显式反馈可能污染检测特征。

缓解建议：

- 先做无反馈多任务；
- feedback 模块采用 detach 版本；
- 或在训练后期再开启 feedback。

## 12. 推荐实现顺序

建议严格按照以下顺序落地：

1. 复现并固定 DAL baseline；
2. 增加最小 occupancy head；
3. 建立 occupancy GT 构造与评测脚本；
4. 完成 Stage 1 多任务训练；
5. 若 Stage 1 有正收益，再接入 feature-level feedback；
6. 最后尝试 explicit occupancy prior feedback 与双向一致性约束。

## 13. 结论

从研究和工程权衡来看，**最推荐的起点**是：

- 基于 DAL 共享 BEV 特征；
- 新增一个轻量稀疏 pillar occupancy head；
- 先进行无 feedback 的多任务训练；
- 只有在 occupancy 分支本身稳定且对主任务无明显伤害时，再逐步加入反馈机制。

该路线可以把问题拆成一系列可验证的阶段，便于快速定位收益来源，也更适合后续写成论文实验章节或项目实现计划。
