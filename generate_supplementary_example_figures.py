# 伪代码实现
import matplotlib.pyplot as plt
import numpy as np

# 性能指标 (示例数据)
metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'AUC']
pan_gam = [94.2, 93.2, 95.5, 93.7, 95.1, 0.97]
card = [75.3, 48.9, 98.8, 96.2, 79.5, 0.82]
whole_pan = [88.4, 82.5, 92.8, 86.7, 90.1, 0.93]

# 创建雷达图
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
pan_gam += pan_gam[:1]  # 闭合图形
card += card[:1]
whole_pan += whole_pan[:1]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
ax.plot(angles, pan_gam, 'o-', linewidth=2, label='Pan-GAM (ours)', color='#E41A1C')
ax.plot(angles, card, 's--', linewidth=2, label='CARD Database', color='#377EB8')
ax.plot(angles, whole_pan, 'd-.', linewidth=2, label='Whole Pan-genome', color='#4DAF4A')

ax.fill(angles, pan_gam, alpha=0.1, color='#E41A1C')
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], metrics, size=12)
ax.tick_params(axis='x', pad=20)
ax.set_rlabel_position(0)
plt.yticks([70,80,90], ["70%","80%","90%"], color="grey", size=10)
plt.ylim(70,100)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
plt.title('Comprehensive Performance Comparison by Method', size=14, pad=20)
plt.tight_layout()
plt.savefig('Figure4_performance_radar.png', dpi=300, bbox_inches='tight')

# 伪代码实现
import seaborn as sns
import pandas as pd

# 准确率数据 (示例)
data = {
    'Antibiotic': ['Methicillin', 'Carbapenems', 'Erythromycin', 'Ciprofloxacin', 'Gentamicin', 'Tetracycline'],
    'Asia→Asia': [95.2, 94.8, 93.5, 96.1, 92.7, 94.3],
    'Asia→North America': [93.8, 92.1, 91.4, 94.5, 90.2, 91.8],
    'Asia→Europe': [92.5, 91.3, 90.8, 93.7, 89.5, 90.6],
    'Performance Drop (%)': [2.7, 3.5, 2.7, 2.4, 3.2, 3.7]
}
df = pd.DataFrame(data).set_index('Antibiotic')

# 创建热图
plt.figure(figsize=(12, 8))
sns.heatmap(df[['Asia→Asia', 'Asia→North America', 'Asia→Europe']], 
            annot=True, fmt='.1f', cmap='RdYlGn', 
            vmin=85, vmax=97, cbar_kws={'label': 'Accuracy (%)'})
plt.title('Geographic Generalizability of Pan-GAM Predictions', fontsize=14, pad=20)
plt.ylabel('Antibiotic Class', fontsize=12)
plt.xlabel('Training→Testing Population', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('Figure5_geographic_generalization.png', dpi=300, bbox_inches='tight')

# 伪代码实现
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# A) Pan-GAM识别的关键基因 (mecA例子)
genes = ['mecA', 'mecR1', 'mecI', 'IS431', 'blaZ', 'fusA']
odds_ratios = [245.0, 18.3, 15.7, 12.9, 1.2, 0.8]
p_values = [1.2e-150, 3.5e-15, 2.8e-12, 1.5e-10, 0.25, 0.67]
colors = ['#E41A1C' if p < 0.001 else '#999999' for p in p_values]

ax1.barh(genes, odds_ratios, color=colors)
ax1.set_xscale('log')
ax1.set_title('A) Pan-GAM: mecA Cassette Identification', fontsize=12)
ax1.set_xlabel('Odds Ratio (log scale)', fontsize=10)
ax1.axvline(x=1, color='k', linestyle='--', alpha=0.3)

# B) 传统GWAS假阳性结果
genes_gwas = ['mecA', 'spa', 'clfA', 'fnbA', 'icaA', 'sarA']
p_values_gwas = [1.2e-150, 2.3e-25, 4.5e-20, 8.7e-18, 3.2e-15, 5.6e-12]
ax2.barh(genes_gwas, -np.log10(p_values_gwas), color='#377EB8')
ax2.set_title('B) Standard GWAS: Spurious Associations', fontsize=12)
ax2.set_xlabel('-log10(p-value)', fontsize=10)

# C) 临床相关性
resistance_levels = ['Susceptible', 'Intermediate', 'Resistant']
mecA_expression = [0.2, 1.8, 245.3]
error = [0.1, 0.3, 15.2]
ax3.bar(resistance_levels, mecA_expression, yerr=error, capsize=5, color='#4DAF4A')
ax3.set_yscale('log')
ax3.set_title('C) mecA Expression vs Resistance Level', fontsize=12)
ax3.set_ylabel('Relative Expression (log scale)', fontsize=10)

# D) 时间趋势
years = [2018, 2019, 2020, 2021, 2022, 2023]
mrsa_prevalence = [35.2, 38.7, 42.1, 45.8, 48.3, 51.6]
pan_gam_accuracy = [89.2, 90.5, 91.8, 92.7, 93.5, 94.2]
ax4.plot(years, mrsa_prevalence, 'o-', linewidth=2, label='MRSA Prevalence (%)', color='#E41A1C')
ax4.set_ylabel('Prevalence (%)', color='#E41A1C', fontsize=10)
ax4.tick_params(axis='y', labelcolor='#E41A1C')
ax42 = ax4.twinx()
ax42.plot(years, pan_gam_accuracy, 's--', linewidth=2, label='Pan-GAM Accuracy (%)', color='#377EB8')
ax42.set_ylabel('Accuracy (%)', color='#377EB8', fontsize=10)
ax42.tick_params(axis='y', labelcolor='#377EB8')
ax4.set_title('D) Rising Resistance vs Pan-GAM Performance', fontsize=12)
ax4.set_xlabel('Year', fontsize=10)
ax4.legend(loc='upper left')
ax42.legend(loc='upper right')

plt.tight_layout()
plt.savefig('Figure6_mechanism_discovery.png', dpi=300, bbox_inches='tight')
