# data-analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder


def load_and_explore_data(file_path):
    """
    加载数据并进行基本探索

    Args:
        file_path: 数据集文件路径

    Returns:
        pandas DataFrame: 加载的数据集
    """
    # 加载数据
    df = pd.read_csv(file_path)

    # 输出基本信息
    print("数据集形状:", df.shape)
    print("\n数据集前几行:")
    print(df.head())

    # 检查数据类型和缺失值
    print("\n数据集信息:")
    print(df.info())

    # 数据描述性统计
    print("\n数据描述性统计:")
    print(df.describe())

    return df


def analyze_target_distribution(df, output_dir):
    """
    分析目标变量(diabetes)分布不均衡情况

    Args:
        df: 数据集DataFrame
        output_dir: 输出目录
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 分析目标变量分布
    diabetes_counts = df['diabetes'].value_counts()
    total = len(df)

    print("\n目标变量分布:")
    for label, count in diabetes_counts.items():
        percentage = count / total * 100
        print(f"糖尿病: {label}, 数量: {count}, 比例: {percentage:.2f}%")

    # 创建饼图
    plt.figure(figsize=(10, 6))
    plt.pie(diabetes_counts, labels=['Non-diabetic', 'Diabetic'],
            autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'],
            explode=(0, 0.1), shadow=True)
    plt.title('Distribution of Diabetic vs Non-diabetic Samples')
    plt.axis('equal')  # 确保饼图是圆形

    # 保存饼图
    plt.savefig(os.path.join(output_dir, 'diabetes_distribution_pie.png'))
    plt.close()

    # 创建柱状图
    plt.figure(figsize=(10, 6))
    sns.countplot(x='diabetes', data=df, palette=['#66b3ff', '#ff9999'])
    plt.title('Distribution of Diabetic vs Non-diabetic Samples')
    plt.xlabel('Diabetes (0=No, 1=Yes)')
    plt.ylabel('Count')

    # 添加数量和百分比标签
    for i, count in enumerate(diabetes_counts):
        percentage = count / total * 100
        plt.text(i, count + 50, f'{count}\n({percentage:.1f}%)',
                 ha='center', va='bottom', fontweight='bold')

    # 设置y轴上限以容纳标签
    plt.ylim(0, max(diabetes_counts) * 1.15)

    # 保存柱状图
    plt.savefig(os.path.join(output_dir, 'diabetes_distribution_bar.png'))
    plt.close()



def analyze_features_by_target(df, output_dir):
    """
    按目标变量分析各特征的分布

    Args:
        df: 数据集DataFrame
        output_dir: 输出目录
    """
    # 数值型特征
    numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']

    # 分析数值型特征
    for feature in numerical_features:
        plt.figure(figsize=(12, 6))

        # 创建密度图
        plt.subplot(1, 2, 1)
        for target_value in [0, 1]:
            subset = df[df['diabetes'] == target_value]
            sns.kdeplot(subset[feature], label=f'Diabetes={target_value}')

        plt.title(f'Distribution of {feature} by Diabetes Status')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()

        # 创建箱线图
        plt.subplot(1, 2, 2)
        sns.boxplot(x='diabetes', y=feature, data=df)
        plt.title(f'Boxplot of {feature} by Diabetes Status')
        plt.xlabel('Diabetes (0=No, 1=Yes)')
        plt.ylabel(feature)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{feature}_by_diabetes.png'))
        plt.close()

    # 分类特征
    categorical_features = ['gender', 'hypertension', 'heart_disease', 'smoking_history']

    # 处理分类特征
    for feature in categorical_features:
        plt.figure(figsize=(12, 6))

        # 创建带百分比的堆叠条形图
        cross_tab = pd.crosstab(df[feature], df['diabetes'])
        cross_tab_pct = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100

        cross_tab_pct.plot(kind='bar', stacked=True, colormap='coolwarm')
        plt.title(f'Relationship between {feature} and Diabetes (Percentage)')
        plt.xlabel(feature)
        plt.ylabel('Percentage')
        plt.legend(title='Diabetes', labels=['Non-diabetic', 'Diabetic'])
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{feature}_diabetes_pct.png'))
        plt.close()

        # 创建计数图
        plt.figure(figsize=(12, 6))
        sns.countplot(x=feature, hue='diabetes', data=df)
        plt.title(f'Relationship between {feature} and Diabetes (Count)')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.legend(title='Diabetes', labels=['Non-diabetic', 'Diabetic'])
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{feature}_diabetes_count.png'))
        plt.close()




def analyze_correlations(df, output_dir):
    """
    分析特征之间的相关性

    Args:
        df: 数据集DataFrame
        output_dir: 输出目录
    """
    # 准备数据 - 将分类变量转换为数值
    df_encoded = df.copy()

    # 对分类变量进行编码
    categorical_features = ['gender', 'smoking_history']
    label_encoder = LabelEncoder()

    for feature in categorical_features:
        df_encoded[feature] = label_encoder.fit_transform(df[feature])

    # 计算相关系数
    correlation_matrix = df_encoded.corr()

    # 绘制热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()

    # 绘制与目标变量的相关性条形图
    diabetes_correlations = correlation_matrix['diabetes'].sort_values(ascending=False).drop('diabetes')

    plt.figure(figsize=(12, 8))
    diabetes_correlations.plot(kind='barh', colormap='coolwarm')
    plt.title('Correlation of Features with Diabetes')
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diabetes_correlations.png'))
    plt.close()




def analyze_feature_distributions(df, output_dir):
    """
    分析各特征的分布情况，绘制分布曲线图，并计算最大值、最小值

    Args:
        df: 数据集DataFrame
        output_dir: 输出目录
    """
    # 创建特征分布信息文件
    with open(os.path.join(output_dir, 'feature_distribution_stats.txt'), 'w', encoding='utf-8') as f:
        f.write("特征分布统计信息\n")
        f.write("==============\n\n")

        # 数值型特征
        f.write("数值型特征:\n")
        numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']

        for feature in numerical_features:
            # 计算基本统计量
            min_val = df[feature].min()
            max_val = df[feature].max()
            mean_val = df[feature].mean()
            median_val = df[feature].median()
            std_val = df[feature].std()

            # 写入统计信息
            f.write(f"\n{feature}:\n")
            f.write(f"  最小值: {min_val}\n")
            f.write(f"  最大值: {max_val}\n")
            f.write(f"  平均值: {mean_val:.2f}\n")
            f.write(f"  中位数: {median_val:.2f}\n")
            f.write(f"  标准差: {std_val:.2f}\n")

            # 分位数
            f.write("  分位数:\n")
            for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
                f.write(f"    {q * 100}%: {df[feature].quantile(q):.2f}\n")

            # 绘制分布曲线图
            plt.figure(figsize=(10, 6))

            # 绘制直方图和KDE
            sns.histplot(df[feature], kde=True, color='skyblue')

            # 添加垂直线表示平均值和中位数
            plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            plt.axvline(median_val, color='green', linestyle=':', label=f'Median: {median_val:.2f}')

            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{feature}_distribution.png'))
            plt.close()

        # 分类特征
        f.write("\n\n分类特征:\n")
        categorical_features = ['gender', 'hypertension', 'heart_disease', 'smoking_history', 'diabetes']

        for feature in categorical_features:
            # 计算值计数
            value_counts = df[feature].value_counts()

            # 写入统计信息
            f.write(f"\n{feature}:\n")
            f.write(f"  唯一值数量: {df[feature].nunique()}\n")
            f.write("  值分布:\n")

            for value, count in value_counts.items():
                percentage = count / len(df) * 100
                f.write(f"    {value}: {count} ({percentage:.2f}%)\n")

            # 绘制分布图
            plt.figure(figsize=(10, 6))
            sns.countplot(x=feature, data=df, palette='viridis')

            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Count')

            # 添加数量和百分比标签
            for i, count in enumerate(value_counts):
                percentage = count / len(df) * 100
                plt.text(i, count + 10, f'{count}\n({percentage:.1f}%)',
                         ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{feature}_distribution.png'))
            plt.close()


def save_summary_report(df, output_dir):
    """
    保存数据分析的总结报告

    Args:
        df: 数据集DataFrame
        output_dir: 输出目录
    """
    with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w', encoding='utf-8') as f:
        # 基本信息
        f.write("糖尿病预测数据集分析摘要\n")
        f.write("=" * 50 + "\n\n")

        # 数据集信息
        f.write("数据集信息:\n")
        f.write(f"样本数量: {df.shape[0]}\n")
        f.write(f"特征数量: {df.shape[1] - 1}\n\n")

        # 特征最大值、最小值
        f.write("特征值范围:\n")
        for col in df.columns:
            if df[col].dtype.kind in 'ifc':  # 整数、浮点数或复数
                f.write(f"{col}: 最小值={df[col].min()}, 最大值={df[col].max()}\n")
            else:
                f.write(f"{col}: 类别数={df[col].nunique()}\n")
        f.write("\n")

        # 目标变量分布
        diabetes_counts = df['diabetes'].value_counts()
        total = len(df)

        f.write("目标变量分布:\n")
        for label, count in diabetes_counts.items():
            percentage = count / total * 100
            f.write(f"糖尿病: {label}, 数量: {count}, 比例: {percentage:.2f}%\n")

        # 计算不平衡比例
        imbalance_ratio = diabetes_counts[0] / diabetes_counts[1]
        f.write(f"\n类别不平衡比例 (非糖尿病:糖尿病): {imbalance_ratio:.2f}:1\n\n")

        # 重要特征
        f.write("重要特征分析:\n")

        # 计算相关系数
        df_encoded = df.copy()
        categorical_features = ['gender', 'smoking_history']
        label_encoder = LabelEncoder()

        for feature in categorical_features:
            df_encoded[feature] = label_encoder.fit_transform(df[feature])

        correlations = df_encoded.corr()['diabetes'].sort_values(ascending=False).drop('diabetes')

        f.write("与糖尿病相关性最强的特征:\n")
        for feature, corr in correlations.items():
            f.write(f"- {feature}: {corr:.4f}\n")

        # 数值特征的统计
        f.write("\n按糖尿病状态的主要数值特征统计:\n")
        numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']

        for feature in numerical_features:
            f.write(f"\n{feature}:\n")
            stats_0 = df[df['diabetes'] == 0][feature].describe()
            stats_1 = df[df['diabetes'] == 1][feature].describe()

            f.write(
                f"  非糖尿病患者 - 均值: {stats_0['mean']:.2f}, 标准差: {stats_0['std']:.2f}, 最小值: {stats_0['min']:.2f}, 最大值: {stats_0['max']:.2f}\n")
            f.write(
                f"  糖尿病患者 - 均值: {stats_1['mean']:.2f}, 标准差: {stats_1['std']:.2f}, 最小值: {stats_1['min']:.2f}, 最大值: {stats_1['max']:.2f}\n")

            # t检验，判断两组之间是否有显著差异
            t_stat, p_val = stats.ttest_ind(df[df['diabetes'] == 0][feature].dropna(),
                                            df[df['diabetes'] == 1][feature].dropna())
            significance = "有" if p_val < 0.05 else "无"
            f.write(f"  两组差异显著性: p值={p_val:.4f} ({significance}统计学显著差异)\n")




def main(data_file='dataset/diabetes_prediction_dataset.csv', output_dir='data-analysis-result'):
    """
    主函数，执行整个数据分析流程

    Args:
        data_file: 数据文件路径
        output_dir: 输出目录
    """
    # 加载并探索数据
    df = load_and_explore_data(data_file)

    # 分析目标变量分布
    analyze_target_distribution(df, output_dir)

    # 分析特征分布
    analyze_feature_distributions(df, output_dir)

    # 按目标变量分析特征
    analyze_features_by_target(df, output_dir)

    # 分析相关性
    analyze_correlations(df, output_dir)

    # 保存分析摘要
    save_summary_report(df, output_dir)

    print(f"\n分析完成，结果已保存到 {output_dir} 目录")


if __name__ == "__main__":
    # 导入scipy.stats用于假设检验
    from scipy import stats

    # 执行主函数
    main()