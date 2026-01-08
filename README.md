# COCHA Tech Semantic Shift

用 COCHA 的 decade-level word embeddings 研究科技话语在历史语料中的语义漂移，重点关注：
- 科技概念：**nuclear**, **GMO**, **smartphone + social media**, **AI**
- 指标层（Layer A，可比层）：总体情绪（正负） + 统一 concern 维度，并做 **baseline-adjusted**
- 解释层（Layer B，涌现层）：每个科技在每个 decade 的近邻语义主题聚类，用来解释 Layer A 的突变与迁移

本仓库只管理代码、配置和小体积输出；COCHA embedding 等大文件未上传

---

## 目录结构

COCHA_PROJECT/
README.md
.gitignore
requirements.txt            # 可选：用于复现环境
notebooks/                  # Jupyter notebooks（分析与作图）
src/                        # 可复用的核心代码（读取、对齐、指标、聚类）
configs/                    # 词表与参数（seeds/concerns/plot settings）
results/                    # 输出图表/表格（建议仅放小文件）
