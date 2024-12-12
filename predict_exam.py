import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shap
import optuna
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def optimize_model(model_name, X_train, y_train, X_test, y_test):
    def objective(trial):
        if model_name == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0)
            }
            model = xgb.XGBRegressor(**params, random_state=42)
        
        elif model_name == 'AdaBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1)
            }
            model = AdaBoostRegressor(**params, random_state=42)
        
        elif model_name == 'CatBoost':
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
                'depth': trial.suggest_int('depth', 3, 10)
            }
            model = CatBoostRegressor(**params, random_state=42, verbose=False)
        
        elif model_name == 'Random Forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
            }
            model = RandomForestRegressor(**params, random_state=42)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse

    # Run optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    # Get best parameters
    best_params = study.best_params
    print(f"Best parameters for {model_name}:", best_params)
    return best_params

def train_with_tolerance(X_train, y_train, X_test, y_test, model, tolerance=0.0):
    """
    使用容差训练模型，但用原始标准评估
    
    Args:
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
        model: 模型
        tolerance: 训练时使用的容差值
    """
    # 创建模型的深拷贝
    model_copy = type(model)(**model.get_params())
    
    # 使用容差调整训练标签
    y_train_adjusted = y_train.astype(float).copy()  # 确保是浮点数类型
    
    # 首次训练得到初始预测
    model_copy.fit(X_train, y_train)
    y_train_pred = model_copy.predict(X_train)
    
    # 在容差范围内的预测视为正确
    errors = np.abs(y_train - y_train_pred)
    within_tolerance = np.where(errors <= tolerance, True, False)
    y_train_adjusted[within_tolerance] = y_train_pred[within_tolerance]
    
    # 使用调整后的标签重新训练模型
    model_copy.fit(X_train, y_train_adjusted)
    
    # 使用原始标准评估模型
    y_pred = model_copy.predict(X_test)
    original_metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }
    
    return model_copy, y_pred, original_metrics

def plot_comparison(tolerance_results, output_path):
    plt.figure(figsize=(15, 8))
    tolerances = list(tolerance_results.keys())
    models = list(tolerance_results[0].keys())
    
    for i, model in enumerate(models):
        rmse_values = [tolerance_results[tolerance][model]['RMSE'] for tolerance in tolerances]
        r2_values = [tolerance_results[tolerance][model]['R2'] for tolerance in tolerances]
        
        plt.subplot(2, len(models), i + 1)
        plt.plot(tolerances, rmse_values, marker='o', linestyle='-', label='RMSE')
        plt.title(f'{model} RMSE')
        plt.xlabel('Tolerance')
        plt.ylabel('RMSE')
        plt.xticks(tolerances)
        plt.grid(True)
        
        plt.subplot(2, len(models), i + 1 + len(models))
        plt.plot(tolerances, r2_values, marker='o', linestyle='-', label='R²', color='orange')
        plt.title(f'{model} R²')
        plt.xlabel('Tolerance')
        plt.ylabel('R²')
        plt.xticks(tolerances)
        plt.grid(True)
    
    plt.tight_layout()
    new_output_path = output_path.replace('model_comparison', 'tolerance_comparison')
    plt.savefig(new_output_path, bbox_inches='tight', dpi=300)
    print(f"Comparison plot saved as {new_output_path}")
    plt.show()
    plt.close()

def plot_tolerance_results(tolerance_results, output_path):
    plt.figure(figsize=(15, 8))
    models = list(tolerance_results[0].keys())
    
    for i, model in enumerate(models):
        plt.subplot(2, 2, i + 1)
        plt.plot(tolerance_results[2][model]['y_test'], label='True', marker='o')
        plt.plot(tolerance_results[2][model]['y_pred'], label='Predicted', marker='x')
        plt.title(f'{model} Predictions (Tolerance=2)')
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Tolerance results plot saved as {output_path}")
    plt.show()
    plt.close()

def plot_model_comparisons(results_df, output_path):
    """
    绘制不同模型在不同容差下的RMSE和R²分数趋势图，并标记最优点。

    Args:
        results_df: 包含模型、容差、RMSE和R²的DataFrame
        output_path: 保存图像的路径
    """
    # 设置图形风格
    plt.figure(figsize=(12, 10))
    
    # 获取模型名称和容差级别
    models = results_df['Model'].unique()
    tolerances = results_df['Tolerance'].unique()
    
    # 绘制 RMSE 图
    plt.subplot(2, 1, 1)
    for model in models:
        model_data = results_df[results_df['Model'] == model]
        plt.plot(model_data['Tolerance'], model_data['RMSE'], marker='o', label=model)
        
        # 找到最优点
        min_rmse_idx = model_data['RMSE'].idxmin()
        min_rmse = model_data.loc[min_rmse_idx]
        plt.annotate(f"{min_rmse['RMSE']:.2f}", 
                     (min_rmse['Tolerance'], min_rmse['RMSE']),
                     textcoords="offset points", xytext=(0,10), ha='center',
                     arrowprops=dict(arrowstyle='->', color='red'))

    plt.title('RMSE Trend with Different Tolerances')
    plt.xlabel('Tolerance Level')
    plt.ylabel('RMSE')
    plt.xticks(tolerances)
    plt.legend()
    plt.grid(True)
    
    # 绘制 R² 图
    plt.subplot(2, 1, 2)
    for model in models:
        model_data = results_df[results_df['Model'] == model]
        plt.plot(model_data['Tolerance'], model_data['R2'], marker='o', label=model)
        
        # 找到最优点
        max_r2_idx = model_data['R2'].idxmax()
        max_r2 = model_data.loc[max_r2_idx]
        plt.annotate(f"{max_r2['R2']:.2f}", 
                     (max_r2['Tolerance'], max_r2['R2']),
                     textcoords="offset points", xytext=(0,10), ha='center',
                     arrowprops=dict(arrowstyle='->', color='blue'))

    plt.title('R2 Score Trend with Different Tolerances')
    plt.xlabel('Tolerance Level')
    plt.ylabel('R2 Score')
    plt.xticks(tolerances)
    plt.legend()
    plt.grid(True)
    
    # 调整布局并保存图像
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_predictions_vs_actual(tolerance_results, output_path):
    """
    绘制四种算法在容差为2时的预测与实际值对比图。

    Args:
        tolerance_results: 包含预测结果的字典
        output_path: 保存图像的路径
    """
    plt.figure(figsize=(14, 10))
    models = ['XGBoost', 'AdaBoost', 'CatBoost', 'Random Forest']
    
    for i, model in enumerate(models):
        plt.subplot(2, 2, i + 1)
        y_test = tolerance_results[2][model]['y_test']
        y_pred = tolerance_results[2][model]['y_pred']
        
        plt.scatter(y_test, y_pred, alpha=0.6, label='Predictions')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
        
        plt.title(f'Prediction vs Actual for {model} (Tolerance ±2)')
        plt.xlabel('Actual Final Score')
        plt.ylabel('Predicted Final Score')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_shap_summary(model, X, model_name, output_path):
    """
    生成并保存SHAP summary图。

    Args:
        model: 训练好的模型
        X: 特征数据
        model_name: 模型名称
        output_path: 保存路径
    """
    plt.figure(figsize=(12, 8))
    
    try:
        if model_name == 'AdaBoost':
            # 对于AdaBoost使用KernelExplainer
            background = shap.sample(X, 100)  # 使用100个样本作为背景数据
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X[:100])  # 为了加快计算，只使用100个样本
        else:
            # 对于其他树模型使用TreeExplainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        
        # 绘制SHAP summary plot
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[0], X.iloc[:100] if model_name == 'AdaBoost' else X, 
                            show=False)
        else:
            shap.summary_plot(shap_values, X.iloc[:100] if model_name == 'AdaBoost' else X, 
                            show=False)
            
        plt.title(f'SHAP Summary for {model_name}')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Successfully saved SHAP plot for {model_name}")
    except Exception as e:
        print(f"Error in plot_shap_summary for {model_name}: {str(e)}")
    finally:
        plt.close()

def analyze_pass_prediction(tolerance_results, output_path):
    """
    分析四种算法对及格预测的效果。
    
    Args:
        tolerance_results: 包含预测结果的字典
        output_path: 保存图像的路径
    """
    plt.figure(figsize=(15, 10))
    models = ['XGBoost', 'AdaBoost', 'CatBoost', 'Random Forest']
    metrics = {'Model': [], 'Precision': [], 'Recall': [], 'F1': []}
    
    for i, model in enumerate(models):
        plt.subplot(2, 2, i + 1)
        y_test = tolerance_results[2][model]['y_test']
        y_pred = tolerance_results[2][model]['y_pred']
        
        # 转换为及格/不及格的二分类问题
        y_test_pass = (y_test >= 59.5).astype(int)
        y_pred_pass = (y_pred >= 59.5).astype(int)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_test_pass, y_pred_pass)
        
        # 计算评估指标
        precision = precision_score(y_test_pass, y_pred_pass)
        recall = recall_score(y_test_pass, y_pred_pass)
        f1 = f1_score(y_test_pass, y_pred_pass)
        
        # 存储指标
        metrics['Model'].append(model)
        metrics['Precision'].append(precision)
        metrics['Recall'].append(recall)
        metrics['F1'].append(f1)
        
        # 绘制混淆矩阵热力图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Fail', 'Pass'],
                   yticklabels=['Fail', 'Pass'],
                   annot_kws={'size': 16},  # 设置数字大小
                   )
        plt.title(f'{model}\nPrecision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}')
        plt.xlabel('Predicted', fontsize=14)  # 增加横坐标标签大小
        plt.ylabel('Actual', fontsize=14)     # 增加纵坐标标签大小
        plt.xticks(fontsize=14)  # 增加横坐标刻度标签大小
        plt.yticks(fontsize=14)  # 增加纵坐标刻度标签大小
    
    plt.tight_layout()
    plt.savefig(output_path + '_confusion_matrices.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 绘制性能指标对比柱状图
    metrics_df = pd.DataFrame(metrics)
    plt.figure(figsize=(12, 6))
    ax = metrics_df.set_index('Model')[['Precision', 'Recall', 'F1']].plot(kind='bar')
    plt.title('Pass/Fail Prediction Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 设置x轴标签水平显示
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path + '_metrics_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    return metrics_df

def train_and_evaluate_models(data, output_prefix, drop_columns=None):
    print(f"\n处理数据集: {output_prefix}")
    print(f"数据集大小: {data.shape}")
    
    # 添加特征名称映射
    feature_name_mapping = {
        'BasicRequired': 'Discipline foundation compulsory course',
        'IndividualBonus': 'Extra Reward',
        'GeneralElective': 'General public elective course',
        'Lab': 'Lab Performance',
        'IndividualPenalty': 'Extra Penalty',
        'Homework': 'Homework',
        'ClassPerformance': 'Classroom Performance',
        'SelfStudy': 'Self-study Performance',
        'GeneralRequired': 'General education compulsory course'
    }
    
    # 重命名特征列
    data = data.rename(columns=feature_name_mapping)
    
    # 特征选择
    if drop_columns:
        data = data.drop(columns=drop_columns)
    
    # Exclude 'Name' from features
    feature_columns = [col for col in data.columns if col not in ['Name', 'Final']]
    X = data[feature_columns]
    y = data['Final']
    
    # 数据预处理
    X = X.fillna(X.mean())
    
    # 检查数据类型
    print("Feature data types before conversion:")
    print(X.dtypes)
    
    # 如果有非数值列，行转换或删除
    for col in X.columns:
        if X[col].dtype == 'object':
            # 将分类数据转换为数值编码
            X[col] = X[col].astype('category').cat.codes
    
    # 确保目标变量是数值类型
    if y.dtype == 'object':
        y = y.astype('category').cat.codes
    
    print("Feature data types after conversion:")
    print(X.dtypes)
    print("Target data type:", y.dtype)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=43
    )
    
    # 优化所有模型
    best_xgb_params = optimize_model('XGBoost', X_train, y_train, X_test, y_test)
    best_adaboost_params = optimize_model('AdaBoost', X_train, y_train, X_test, y_test)
    best_catboost_params = optimize_model('CatBoost', X_train, y_train, X_test, y_test)
    best_rf_params = optimize_model('Random Forest', X_train, y_train, X_test, y_test)
    
    # 定义优化后的模型
    models = {
        'XGBoost': xgb.XGBRegressor(**best_xgb_params, random_state=42),
        'AdaBoost': AdaBoostRegressor(**best_adaboost_params, random_state=42),
        'CatBoost': CatBoostRegressor(**best_catboost_params, random_state=42, verbose=False),
        'Random Forest': RandomForestRegressor(**best_rf_params, random_state=42)
    }
    
    # 存储不同容差的结果
    tolerance_results = {tolerance: {} for tolerance in range(0, 6)}
    
    results_table = []

    # 训练和评估
    for name, model in models.items():
        try:
            print(f"\n训练 {name} 模型...")
            
            # 训练基础模型（用于SHAP值计算）
            model.fit(X_train, y_train)
            
            # 生成SHAP图
            try:
                shap_output_path = f'2/basic_models_{name}_shap_summary.png'
                plot_shap_summary(model, X_test, name, shap_output_path)
                print(f"Generated SHAP summary plot for {name}")
            except Exception as e:
                print(f"Error generating SHAP plot for {name}: {str(e)}")
            
            # 容差训练和评估
            # 没有容差的结果
            trained_model, y_pred, metrics = train_with_tolerance(X_train, y_train, X_test, y_test, model, tolerance=0)
            tolerance_results[0][name] = metrics
            results_table.append([name, 0, metrics['RMSE'], metrics['R2']])
            print(f"{name} 容差 0 评估完成: RMSE={metrics['RMSE']:.4f}, R²={metrics['R2']:.4f}")
            
            # 不同容差的结果
            for tolerance in range(1, 6):
                trained_model, y_pred, metrics = train_with_tolerance(X_train, y_train, X_test, y_test, model, tolerance=tolerance)
                tolerance_results[tolerance][name] = metrics
                results_table.append([name, tolerance, metrics['RMSE'], metrics['R2']])
                if tolerance == 2:
                    tolerance_results[tolerance][name]['y_test'] = y_test
                    tolerance_results[tolerance][name]['y_pred'] = y_pred
                print(f"{name} 容差 {tolerance} 评估完成: RMSE={metrics['RMSE']:.4f}, R²={metrics['R2']:.4f}")
            
        except Exception as e:
            print(f"Error with {name}: {str(e)}")
    
    # 生成对比图
    plot_comparison(tolerance_results, '2/model_comparison.png')
    
    plot_tolerance_results(tolerance_results, '2/tolerance_comparison.png')
    
    results_df = pd.DataFrame(results_table, columns=['Model', 'Tolerance', 'RMSE', 'R2'])
    results_df.to_csv('2/tolerance_results.csv', index=False)
    print("Results table saved as '2/tolerance_results.csv'")
    
    # 绘制模型比较图
    plot_model_comparisons(results_df, '2/comprehensive_model_comparison.png')
    
    # 绘制预测与实际值对比图
    plot_predictions_vs_actual(tolerance_results, '2/prediction_vs_actual.png')
    
    # 添加及格预测分析
    metrics_df = analyze_pass_prediction(tolerance_results, '2/pass_prediction')
    metrics_df.to_csv('2/pass_prediction_metrics.csv', index=False)
    print("Pass prediction analysis completed and saved")
    
    return tolerance_results

def main():
    try:
        # 确保路径存在
        os.makedirs('2', exist_ok=True)
        
        # 合并数据集
        data1 = pd.read_excel('2/summary_table.xlsx')
        data2 = pd.read_excel('2/summary_table2.xlsx')
        combined_data = pd.concat([data1, data2], ignore_index=True)
        
        print("数据集1形状:", data1.shape)
        print("数据集2形状:", data2.shape)
        print(f"合并后的数据集大小: {combined_data.shape}")
        
        # 使用全部特征进行训练和评估
        results_all = train_and_evaluate_models(combined_data, 'all_features')
        
    except Exception as e:
        print(f"错误：{str(e)}")

if __name__ == "__main__":
    main() 