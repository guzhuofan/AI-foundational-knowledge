# World Model 阅读清单

## 入门必读

### 1. World Models, From Zero to Hero (最全面教程)
https://hackmd.io/@AbdelStark/world-model-from-zero-to-hero

从 Ha & Schmidhuber 2018 的原始论文讲起，一路覆盖到 2025 年的 Dreamer V4、V-JEPA 2、Genie 3、GAIA-2 等最新工作。讲清楚了 World Model 的三个流派：
- 生成式世界模型（Dreamer 系列，agent in imagination）
- JEPA 系列（LeCun 路线，在表征空间预测而非像素空间）
- 视频生成式世界模型（Genie、GAIA、Oasis 等）

### 2. World Models 原始交互式博客 (David Ha, 2018)
https://worldmodels.github.io

必读经典。带 GIF 演示和可交互元素，直观展示 VAE → MDN-RNN → Controller 三件套架构。

### 3. Quanta Magazine: World Models, an Old Idea in AI, Mount a Comeback (2025.09)
https://www.quantamagazine.org/world-models-an-old-idea-in-ai-mount-a-comeback-20250902

非技术性的概念介绍，从 1943 年 Kenneth Craik 的原始想法讲到 2025 年的前沿。20分钟读完。

---

## 深入技术细节

### 4. World Models (The Long Version) — ADGEfficiency
https://adgefficiency.com/blog/world-models

对 2018 论文的逐段技术拆解，把 VAE 的 latent space、MDN-RNN 的混合密度网络、CMA-ES 控制器都讲透了。

### 5. Reproducing "World Models" — Is training the RNN really needed?
https://ctallec.github.io/world-models

PyTorch 复现 + 批判性实验，探讨是否真的需要训练 RNN。带代码。

---

## 动手实战

### 6. World-Model-2018 — Modern PyTorch Implementation (推荐首选)
https://github.com/BrunooCS/World-Model-2018

最干净的 PyTorch 实现，5 个 Jupyter Notebook：
- 1-Rollouts — 收集数据
- 2-Vision (VAE) — 训练 VAE 视觉模型
- 3-Memory (rnn-mdn) — 训练 MDN-RNN 记忆模型
- 4-Controller (C) — 训练 CMA-ES 控制器
- 5-Videos — 可视化结果

### 7. 原始 TensorFlow 实现 — David Ha
https://github.com/hardmaru/WorldModelsExperiments

配套工程笔记: https://blog.otoro.net/2018/06/09/world-models-experiments

---

## 论文阅读清单

### 8. World Models Reading List (2025)
https://medium.com/@graison/world-models-reading-list-the-papers-you-actually-need-in-2025-882f02d758a9

按主题分类，每篇附 Why / Look for / Notebook prompt。

### 9. Awesome World Models (持续更新的论文收集)
https://github.com/leofan90/Awesome-World-Models

### 10. Schmidhuber 的 World Model Boom 页面
https://people.idsia.ch/~juergen/world-model-boom.html

从发明者视角梳理 1990-2026 的世界模型历史脉络。

### 11. ICLR 2025 Workshop: World Models — Understanding, Modelling and Scaling
https://iclr.cc/virtual/2025/workshop/24000

学术前沿，含 Schmidhuber、Jeff Clune、Jakob Foerster 等 keynote 视频。

---

## 建议学习路径

1. 先读 Quanta Magazine 文章建立概念直觉 (~20分钟)
2. 读 From Zero to Hero 前半部分，理解三件套架构 (~1小时)
3. 边看 World Models 原始博客 + 跑 PyTorch Notebooks (~半天)
4. 按 Reading List 往下追 Dreamer → V-JEPA → Genie 系列论文

核心概念：World Model = 让 Agent 在自己的"梦里"学会做事，而不是在真实环境里撞墙。
