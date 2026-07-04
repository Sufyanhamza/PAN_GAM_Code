import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import seaborn as sns

# ==========================================
# 1. 全局样式设置 (Nature 风格)
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.dpi'] = 300
plt.rcParams['svg.fonttype'] = 'none'

# 定义配色 (Nature Palette)
COLORS = {
    'red': '#E64B35',    # Pan-GAM
    'gray': '#7F7F7F',   # LMM (GWAS)
    'ci_fill': '#E64B35' # 阴影颜色
}

def plot_figure_6():
    # 创建画布
    fig = plt.figure(figsize=(10, 4.5))
    gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.25)

    # ==========================================
    # Figure 6a: Effect of sample size downsampling
    # 描述: Pan-GAM (red) maintains >80% TPR at 30% sample size.
    # LMM (grey) degrades rapidly below 50%.
    # ==========================================
    ax_a = fig.add_subplot(gs[0])

    # 模拟数据
    # X轴: 样本量百分比 (从100%降到10%)
    sample_sizes = np.array([100, 80, 60, 50, 40, 30, 20, 10])
    
    # Y轴: TPR (True Positive Rate)
    # Pan-GAM: 即使在30%样本量，TPR依然维持在0.82左右 (>0.80)
    tpr_gam = np.array([0.96, 0.95, 0.93, 0.91, 0.88, 0.82, 0.65, 0.40])
    
    # LMM: 在50%以下迅速下降
    tpr_lmm = np.array([0.90, 0.85, 0.75, 0.65, 0.40, 0.20, 0.10, 0.05])

    # 绘制曲线
    ax_a.plot(sample_sizes, tpr_gam, 'o-', color=COLORS['red'], label='Pan-GAM', linewidth=2, markersize=5)
    ax_a.plot(sample_sizes, tpr_lmm, 's--', color=COLORS['gray'], label='LMM (GWAS)', linewidth=1.5, markersize=5)

    # 关键标注: ">80% TPR"
    ax_a.axvline(30, color='black', linestyle=':', alpha=0.3)
    ax_a.annotate('>80% TPR\nat 30% Data', 
                  xy=(30, 0.82), xytext=(45, 0.90),
                  arrowprops=dict(arrowstyle="->", color=COLORS['red']),
                  color=COLORS['red'], fontsize=8, fontweight='bold')

    # 关键标注: "Degrades rapidly"
    ax_a.annotate('Rapid degradation\n(<50% sample)', 
                  xy=(40, 0.40), xytext=(20, 0.50),
                  arrowprops=dict(arrowstyle="->", color=COLORS['gray']),
                  color=COLORS['gray'], fontsize=8)

    # 样式调整
    ax_a.set_xlabel('% Sample Size')
    ax_a.set_ylabel('True Positive Rate (TPR)')
    ax_a.set_title('Sensitivity vs. Sample Size', fontsize=10, fontweight='bold')
    ax_a.invert_xaxis()  # 习惯上从100降到0
    ax_a.set_ylim(0, 1.05)
    ax_a.legend(frameon=False)
    
    # 去除右侧和上侧边框
    sns.despine(ax=ax_a)
    # 添加子图编号 'a'
    ax_a.text(-0.15, 1.05, 'a', transform=ax_a.transAxes, fontsize=12, fontweight='bold')

    # ==========================================
    # Figure 6b: Impact of missing phenotypic data
    # 描述: Accuracy vs Missing Labels (0-20%).
    # Pan-GAM retains >90% accuracy even with 20% missing metadata.
    # ==========================================
    ax_b = fig.add_subplot(gs[1])

    # 模拟数据
    # X轴: 缺失数据百分比
    missing_rates = np.array([0, 5, 10, 15, 20])
    
    # Y轴: 模型准确率 (Model Accuracy)
    # Pan-GAM: 0%缺失时0.95 -> 20%缺失时0.91 (>0.90)
    acc_gam_mean = np.array([0.95, 0.945, 0.938, 0.925, 0.912])
    # 95% 置信区间 (CI)
    acc_gam_ci = np.array([0.01, 0.012, 0.015, 0.018, 0.02])

    # 绘制均值线
    ax_b.plot(missing_rates, acc_gam_mean, 'o-', color=COLORS['red'], label='ML-imputed Pan-GAM', linewidth=2)
    
    # 绘制阴影区域 (95% CI)
    ax_b.fill_between(missing_rates, 
                      acc_gam_mean - acc_gam_ci, 
                      acc_gam_mean + acc_gam_ci, 
                      color=COLORS['ci_fill'], alpha=0.15, label='95% Confidence Interval')

    # 关键标注: ">90% Accuracy"
    ax_b.axhline(0.90, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax_b.annotate('Retains >90% Accuracy\n(20% Missing)', 
                  xy=(20, 0.912), xytext=(10, 0.85),
                  arrowprops=dict(arrowstyle="->", color=COLORS['red']),
                  color=COLORS['red'], fontsize=8, fontweight='bold')

    # 样式调整
    ax_b.set_xlabel('% Missing Phenotypic Labels')
    ax_b.set_ylabel('Model Accuracy')
    ax_b.set_title('Robustness to Missing Metadata', fontsize=10, fontweight='bold')
    ax_b.set_ylim(0.80, 1.0) # 聚焦于高准确率区域
    ax_b.set_xticks(missing_rates)
    ax_b.legend(frameon=False, loc='lower left')
    
    # 去除右侧和上侧边框
    sns.despine(ax=ax_b)
    # 添加子图编号 'b'
    ax_b.text(-0.15, 1.05, 'b', transform=ax_b.transAxes, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()

# 执行绘图
if __name__ == "__main__":
    plot_figure_6()
