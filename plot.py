"""
EEG情绪识别实验 - 结果可视化
生成所有结题报告所需图表
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ============================================================
# 中文字体设置
# ============================================================
CHINESE_FONTS = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei',
                 'STHeiti', 'Arial Unicode MS', 'DejaVu Sans']

font_set = False
for font_name in CHINESE_FONTS:
    try:
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams['axes.unicode_minus'] = False
        fig_test = plt.figure()
        fig_test.text(0.5, 0.5, '测试')
        plt.close(fig_test)
        font_set = True
        print(f"使用字体: {font_name}")
        break
    except Exception:
        continue

if not font_set:
    print("警告: 未找到中文字体，图表标题将使用英文")
    USE_CHINESE = False
else:
    USE_CHINESE = True

# ============================================================
# 配置
# ============================================================
DATA_DIR = 'results/data'
OUTPUT_DIR = 'results/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DIMENSIONS = ['valence', 'arousal', 'dominance']
DIM_NAMES = {
    'valence': '效价 (Valence)' if USE_CHINESE else 'Valence',
    'arousal': '唤醒度 (Arousal)' if USE_CHINESE else 'Arousal',
    'dominance': '支配度 (Dominance)' if USE_CHINESE else 'Dominance'
}

COLORS = {
    'valence': '#4C72B0',
    'arousal': '#DD8452',
    'dominance': '#55A868',
    'primary': '#4C72B0',
    'secondary': '#DD8452',
    'accent': '#55A868',
}


def load_results():
    """加载实验结果"""
    filepath = f'{DATA_DIR}/full_model_all_results.json'
    if not os.path.exists(filepath):
        print(f"错误: 找不到 {filepath}")
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_fold_history(dim, subject_id, fold_idx):
    """加载单个fold的训练历史"""
    filepath = f'{DATA_DIR}/full_model_{dim}_s{subject_id:02d}_fold{fold_idx}.json'
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_subject_results(dim, subject_id):
    """加载单个受试者的结果"""
    filepath = f'{DATA_DIR}/full_model_{dim}_s{subject_id:02d}_results.json'
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================
# 图1: 三维度准确率柱状图
# ============================================================
def plot_dimension_accuracy(results):
    """三个情绪维度的平均准确率对比"""
    fig, ax = plt.subplots(figsize=(8, 6))

    dims = []
    means = []
    stds = []
    colors = []

    for dim in DIMENSIONS:
        if dim in results:
            dims.append(DIM_NAMES[dim])
            means.append(results[dim]['mean_accuracy'] * 100)
            stds.append(results[dim]['std_accuracy'] * 100)
            colors.append(COLORS[dim])

    bars = ax.bar(dims, means, yerr=stds, capsize=8,
                  color=colors, edgecolor='white', linewidth=1.5,
                  width=0.5, alpha=0.85, error_kw={'linewidth': 2})

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 1,
                f'{mean:.2f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('准确率 (%)' if USE_CHINESE else 'Accuracy (%)', fontsize=13)
    ax.set_title('各情绪维度分类准确率' if USE_CHINESE else
                 'Classification Accuracy by Emotion Dimension',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='随机基线 (50%)')
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=12)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/dimension_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 图1: 三维度准确率柱状图 → dimension_accuracy.png")


# ============================================================
# 图2: 各受试者准确率分布
# ============================================================
def plot_subject_accuracy(results):
    """32个受试者在三个维度上的准确率"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    for idx, dim in enumerate(DIMENSIONS):
        if dim not in results:
            continue

        ax = axes[idx]
        accs = [a * 100 for a in results[dim]['subject_accuracies']]
        subjects = list(range(1, len(accs) + 1))
        mean_acc = np.mean(accs)

        bars = ax.bar(subjects, accs, color=COLORS[dim], alpha=0.8,
                      edgecolor='white', linewidth=0.5)
        ax.axhline(y=mean_acc, color='red', linestyle='--', linewidth=1.5,
                   label=f'均值: {mean_acc:.2f}%')
        ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)

        ax.set_ylabel('准确率 (%)' if USE_CHINESE else 'Accuracy (%)', fontsize=11)
        ax.set_title(DIM_NAMES[dim], fontsize=13, fontweight='bold')
        ax.set_ylim(40, 100)
        ax.legend(fontsize=10, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        max_idx = np.argmax(accs)
        min_idx = np.argmin(accs)
        ax.annotate(f'{accs[max_idx]:.1f}%',
                    xy=(subjects[max_idx], accs[max_idx]),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=8, color='green', fontweight='bold')
        ax.annotate(f'{accs[min_idx]:.1f}%',
                    xy=(subjects[min_idx], accs[min_idx]),
                    xytext=(0, -12), textcoords='offset points',
                    ha='center', fontsize=8, color='red', fontweight='bold')

    axes[-1].set_xlabel('受试者编号' if USE_CHINESE else 'Subject ID', fontsize=12)
    axes[-1].set_xticks(range(1, 33))
    axes[-1].set_xticklabels([f's{i:02d}' for i in range(1, 33)],
                              rotation=45, fontsize=8)

    plt.suptitle('各受试者分类准确率' if USE_CHINESE else
                 'Classification Accuracy per Subject',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/subject_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 图2: 各受试者准确率分布 → subject_accuracy.png")


# ============================================================
# 图3: 混淆矩阵
# ============================================================
def plot_confusion_matrices(results):
    """三个维度的混淆矩阵"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    class_labels = ['低 (Low)', '高 (High)'] if USE_CHINESE else ['Low', 'High']

    for idx, dim in enumerate(DIMENSIONS):
        if dim not in results:
            continue

        ax = axes[idx]
        cm = np.array(results[dim]['confusion_matrix'])

        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues',
                       vmin=0, vmax=1)

        for i in range(2):
            for j in range(2):
                color = 'white' if cm_norm[i, j] > 0.5 else 'black'
                ax.text(j, i, f'{cm_norm[i, j]:.2%}\n({cm[i, j]})',
                        ha='center', va='center', fontsize=11,
                        color=color, fontweight='bold')

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(class_labels, fontsize=11)
        ax.set_yticklabels(class_labels, fontsize=11)
        ax.set_xlabel('预测标签' if USE_CHINESE else 'Predicted', fontsize=11)
        ax.set_ylabel('真实标签' if USE_CHINESE else 'Actual', fontsize=11)
        ax.set_title(DIM_NAMES[dim], fontsize=13, fontweight='bold')

    plt.suptitle('混淆矩阵' if USE_CHINESE else 'Confusion Matrices',
                 fontsize=15, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 图3: 混淆矩阵 → confusion_matrices.png")


# ============================================================
# 图4: 训练曲线（选取代表性受试者）
# ============================================================
def plot_training_curves(results):
    """展示代表性受试者的训练过程"""
    dim = 'valence'
    if dim not in results:
        dim = DIMENSIONS[0]

    accs = results[dim]['subject_accuracies']
    sorted_idx = np.argsort(accs)
    representative = [
        sorted_idx[0] + 1,
        sorted_idx[len(sorted_idx) // 2] + 1,
        sorted_idx[-1] + 1
    ]
    labels_repr = [
        '低准确率受试者' if USE_CHINESE else 'Low Acc Subject',
        '中等准确率受试者' if USE_CHINESE else 'Medium Acc Subject',
        '高准确率受试者' if USE_CHINESE else 'High Acc Subject'
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax_idx, (s_id, label) in enumerate(zip(representative, labels_repr)):
        ax = axes[ax_idx]

        history = load_fold_history(dim, s_id, 1)
        if history is None:
            continue

        epochs = history['epochs']
        ax.plot(epochs, history['train_loss'], 'b-', linewidth=1.5,
                label='训练损失' if USE_CHINESE else 'Train Loss', alpha=0.8)

        ax2 = ax.twinx()
        ax2.plot(epochs, [a * 100 for a in history['train_acc']],
                 'r--', linewidth=1.5,
                 label='训练准确率' if USE_CHINESE else 'Train Acc', alpha=0.7)
        ax2.plot(epochs, [a * 100 for a in history['test_acc']],
                 'g-', linewidth=2,
                 label='测试准确率' if USE_CHINESE else 'Test Acc', alpha=0.9)

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('损失' if USE_CHINESE else 'Loss', fontsize=11, color='blue')
        ax2.set_ylabel('准确率 (%)' if USE_CHINESE else 'Accuracy (%)',
                       fontsize=11, color='green')
        ax2.set_ylim(30, 100)

        subject_acc = accs[s_id - 1] * 100
        ax.set_title(f's{s_id:02d} ({label})\nAcc={subject_acc:.1f}%',
                     fontsize=11, fontweight='bold')

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  loc='center right', fontsize=8)

        ax.spines['top'].set_visible(False)

    plt.suptitle('训练过程曲线 (Valence, Fold 1)' if USE_CHINESE else
                 'Training Curves (Valence, Fold 1)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 图4: 训练曲线 → training_curves.png")


# ============================================================
# 图5: 准确率箱线图
# ============================================================
def plot_accuracy_boxplot(results):
    """三个维度的准确率分布箱线图"""
    fig, ax = plt.subplots(figsize=(8, 6))

    data = []
    labels = []
    for dim in DIMENSIONS:
        if dim in results:
            data.append([a * 100 for a in results[dim]['subject_accuracies']])
            labels.append(DIM_NAMES[dim])

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    widths=0.4, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red',
                                   markersize=8),
                    medianprops=dict(color='black', linewidth=2),
                    flierprops=dict(marker='o', markerfacecolor='gray',
                                    markersize=5))

    for patch, dim in zip(bp['boxes'], DIMENSIONS):
        patch.set_facecolor(COLORS.get(dim, '#4C72B0'))
        patch.set_alpha(0.7)

    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5,
               label='随机基线 (50%)' if USE_CHINESE else 'Random Baseline (50%)')

    ax.set_ylabel('准确率 (%)' if USE_CHINESE else 'Accuracy (%)', fontsize=13)
    ax.set_title('各维度准确率分布' if USE_CHINESE else
                 'Accuracy Distribution by Dimension',
                 fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=12)

    for i, d in enumerate(data):
        mean_val = np.mean(d)
        ax.annotate(f'{mean_val:.1f}%',
                    xy=(i + 1, mean_val),
                    xytext=(25, 5), textcoords='offset points',
                    fontsize=11, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/accuracy_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 图5: 准确率箱线图 → accuracy_boxplot.png")


# ============================================================
# 图6: 三维度雷达图
# ============================================================
def plot_radar_chart(results):
    """三个维度的性能雷达图"""
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    dims_available = [d for d in DIMENSIONS if d in results]
    if len(dims_available) < 3:
        print("⚠️ 雷达图需要三个维度，跳过")
        return

    categories = [DIM_NAMES[d] for d in dims_available]
    values = [results[d]['mean_accuracy'] * 100 for d in dims_available]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values_closed = values + [values[0]]
    angles_closed = angles + [angles[0]]

    ax.plot(angles_closed, values_closed, 'o-', linewidth=2.5,
            color=COLORS['primary'], markersize=10)
    ax.fill(angles_closed, values_closed, alpha=0.2, color=COLORS['primary'])

    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_rticks([20, 40, 50, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '50%', '60%', '80%', '100%'], fontsize=9)

    for angle, value in zip(angles, values):
        ax.annotate(f'{value:.1f}%',
                    xy=(angle, value),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=12, fontweight='bold', color=COLORS['primary'])

    ax.set_title('多维度情绪识别性能' if USE_CHINESE else
                 'Multi-dimension Emotion Recognition',
                 fontsize=14, fontweight='bold', pad=25)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 图6: 雷达图 → radar_chart.png")


# ============================================================
# 图7: 受试者准确率热力图
# ============================================================
def plot_subject_heatmap(results):
    """32个受试者 × 3个维度 的准确率热力图"""
    fig, ax = plt.subplots(figsize=(14, 5))

    dims_available = [d for d in DIMENSIONS if d in results]
    n_subjects = len(results[dims_available[0]]['subject_accuracies'])

    heatmap_data = []
    for dim in dims_available:
        heatmap_data.append([a * 100 for a in results[dim]['subject_accuracies']])
    heatmap_data = np.array(heatmap_data)

    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto',
                   vmin=45, vmax=90)

    ax.set_xticks(range(n_subjects))
    ax.set_xticklabels([f's{i+1:02d}' for i in range(n_subjects)],
                        rotation=45, fontsize=8)
    ax.set_yticks(range(len(dims_available)))
    ax.set_yticklabels([DIM_NAMES[d] for d in dims_available], fontsize=12)

    for i in range(len(dims_available)):
        for j in range(n_subjects):
            val = heatmap_data[i, j]
            color = 'white' if val < 55 or val > 80 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                    fontsize=7, color=color, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('准确率 (%)' if USE_CHINESE else 'Accuracy (%)', fontsize=11)

    ax.set_title('受试者-维度 准确率热力图' if USE_CHINESE else
                 'Subject-Dimension Accuracy Heatmap',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('受试者' if USE_CHINESE else 'Subject', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/subject_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 图7: 受试者热力图 → subject_heatmap.png")


# ============================================================
# 图8: 综合结果总表
# ============================================================
def plot_summary_table(results):
    """生成结果总表图"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    dims_available = [d for d in DIMENSIONS if d in results]

    col_labels = ['情绪维度' if USE_CHINESE else 'Dimension',
                  '平均准确率' if USE_CHINESE else 'Mean Acc',
                  '标准差' if USE_CHINESE else 'Std Dev',
                  '最高' if USE_CHINESE else 'Max',
                  '最低' if USE_CHINESE else 'Min',
                  '受试者数' if USE_CHINESE else 'Subjects']

    table_data = []
    for dim in dims_available:
        accs = results[dim]['subject_accuracies']
        table_data.append([
            DIM_NAMES[dim],
            f'{results[dim]["mean_accuracy"]*100:.2f}%',
            f'±{results[dim]["std_accuracy"]*100:.2f}%',
            f'{max(accs)*100:.2f}%',
            f'{min(accs)*100:.2f}%',
            str(len(accs))
        ])

    all_means = [results[d]['mean_accuracy'] * 100 for d in dims_available]
    table_data.append([
        '总平均' if USE_CHINESE else 'Average',
        f'{np.mean(all_means):.2f}%',
        '-', '-', '-',
        str(len(results[dims_available[0]]['subject_accuracies']))
    ])

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#4C72B0')
            cell.set_text_props(color='white', fontweight='bold')
        elif row == len(table_data):
            cell.set_facecolor('#E8E8E8')
            cell.set_text_props(fontweight='bold')
        else:
            cell.set_facecolor('#F7F7F7' if row % 2 == 0 else 'white')

    ax.set_title('实验结果总表' if USE_CHINESE else 'Experiment Results Summary',
                 fontsize=15, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 图8: 结果总表 → summary_table.png")


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("📊 生成实验结果图表")
    print("=" * 60)

    results = load_results()
    if results is None:
        print("无法加载结果文件，请检查路径")
        return

    print("\n结果概览:")
    for dim in DIMENSIONS:
        if dim in results:
            acc = results[dim]['mean_accuracy'] * 100
            std = results[dim]['std_accuracy'] * 100
            print(f"  {DIM_NAMES[dim]}: {acc:.2f}% ± {std:.2f}%")

    print(f"\n生成图表到 {OUTPUT_DIR}/")
    print("-" * 60)

    plot_dimension_accuracy(results)
    plot_subject_accuracy(results)
    plot_confusion_matrices(results)
    plot_training_curves(results)
    plot_accuracy_boxplot(results)
    plot_radar_chart(results)
    plot_subject_heatmap(results)
    plot_summary_table(results)

    print("-" * 60)
    print(f"✅ 所有图表已保存到 {OUTPUT_DIR}/")
    print("\n生成的图表:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.png'):
            size = os.path.getsize(f'{OUTPUT_DIR}/{f}') / 1024
            print(f"  📈 {f} ({size:.0f} KB)")


if __name__ == '__main__':
    main()