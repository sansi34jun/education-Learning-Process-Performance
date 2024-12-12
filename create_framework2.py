from graphviz import Digraph
import os

def create_research_framework():
    # 设置 Graphviz 的路径
    os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"

    dot = Digraph(comment='Research Framework')
    dot.attr(rankdir='LR', size='55,55', dpi='300', fontname='Arial')  # 保持原有尺寸
    
    # 定义节点样式
    dot.attr('node', shape='none', fontname='Arial Bold', fontsize='32')  # 增加字体大小并加粗

    # 数据收集和预处理
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='Data collection and preprocessing', fontsize='32', fontname='Arial Bold')  # Removed style='filled'
        c.node('course_performance', '''<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
            <TR><TD><FONT POINT-SIZE="48">📊</FONT></TD></TR>
            <TR><TD><FONT POINT-SIZE="32" FACE="Arial Bold">Course Performance</FONT></TD></TR>
            </TABLE>>''')
        c.node('completed_courses_grades', '''<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
            <TR><TD><FONT POINT-SIZE="48">📚</FONT></TD></TR>
            <TR><TD><FONT POINT-SIZE="32" FACE="Arial Bold">Completed Courses Grades</FONT></TD></TR>
            </TABLE>>''')

    # 特征选择
    dot.node('feature_selection', '''<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
        <TR><TD><FONT POINT-SIZE="48">🔍</FONT></TD></TR>
        <TR><TD><FONT POINT-SIZE="32" FACE="Arial Bold">Feature selection</FONT></TD></TR>
        </TABLE>>''')

    # 机器学习方法
    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='Machine Learning Methods', fontsize='32', fontname='Arial Bold')
        methods = {
            'xgboost': ('🚀', 'XGBoost'),
            'adaboost': ('🔄', 'AdaBoost'),
            'catboost': ('🐱', 'CatBoost'),
            'random_forest': ('🌳', 'Random Forest')
        }
        for key, (emoji, name) in methods.items():
            c.node(key, f'''<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                <TR><TD><FONT POINT-SIZE="48">{emoji}</FONT></TD></TR>
                <TR><TD><FONT POINT-SIZE="32" FACE="Arial Bold">{name}</FONT></TD></TR>
                </TABLE>>''')

    # 容差对比
    dot.node('tolerance_comparison', '''<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
        <TR><TD><FONT POINT-SIZE="48">⚖️</FONT></TD></TR>
        <TR><TD><FONT POINT-SIZE="32" FACE="Arial Bold">Tolerance Comparison (0-5)</FONT></TD></TR>
        </TABLE>>''')

    # SHAP 分析
    dot.node('shap_analysis', '''<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
        <TR><TD><FONT POINT-SIZE="48">🏆</FONT></TD></TR>
        <TR><TD><FONT POINT-SIZE="32" FACE="Arial Bold">SHAP Analysis</FONT></TD></TR>
        </TABLE>>''')

    # 训练和测试
    dot.node('train_test', '''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" STYLE="rounded" BGCOLOR="lightsalmon">
        <TR><TD><FONT POINT-SIZE="48">🔬</FONT></TD></TR>
        <TR><TD><FONT POINT-SIZE="32" FACE="Arial Bold">Train &amp; Test</FONT></TD></TR>
        </TABLE>>''')

    # 考试成绩
    dot.node('exam_scores', '''<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
        <TR><TD><FONT POINT-SIZE="48">📝</FONT></TD></TR>
        <TR><TD><FONT POINT-SIZE="32" FACE="Arial Bold">Exam Scores</FONT></TD></TR>
        </TABLE>>''')

    # 评估指标
    dot.node('evaluation', '''<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
        <TR><TD><FONT POINT-SIZE="48">📊</FONT></TD></TR>
        <TR><TD><FONT POINT-SIZE="32" FACE="Arial Bold">Evaluation Metrics</FONT></TD></TR>
        <TR><TD><FONT POINT-SIZE="48">📉</FONT> <FONT POINT-SIZE="32" FACE="Arial Bold">RMSE</FONT></TD></TR>
        <TR><TD><FONT POINT-SIZE="48">📈</FONT> <FONT POINT-SIZE="32" FACE="Arial Bold">R2</FONT></TD></TR>
        </TABLE>>''')

    # 可视化分析
    dot.node('visualization_impact', '''<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
        <TR><TD><FONT POINT-SIZE="48">🔍</FONT></TD></TR>
        <TR><TD><FONT POINT-SIZE="32" FACE="Arial Bold">Visualization of Impact on Exam Scores</FONT></TD></TR>
        </TABLE>>''')

    # 添加边
    dot.edge('course_performance', 'feature_selection')
    dot.edge('completed_courses_grades', 'feature_selection')
    dot.edge('feature_selection', 'xgboost')
    dot.edge('feature_selection', 'adaboost')
    dot.edge('feature_selection', 'catboost')
    dot.edge('feature_selection', 'random_forest')
    dot.edge('xgboost', 'tolerance_comparison')
    dot.edge('adaboost', 'tolerance_comparison')
    dot.edge('catboost', 'tolerance_comparison')
    dot.edge('random_forest', 'tolerance_comparison')
    dot.edge('tolerance_comparison', 'train_test')
    dot.edge('shap_analysis', 'train_test')
    dot.edge('train_test', 'exam_scores')
    dot.edge('exam_scores', 'evaluation')
    dot.edge('evaluation', 'visualization_impact')

    # 添加位置约束
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('course_performance')
        s.node('completed_courses_grades')
    
    # 调整布局
    dot.attr(rank='same', rankdir='LR')

    # 渲染图形
    dot.render('research_framework', format='png', cleanup=True, engine='dot')
    print("Research framework graph has been created: research_framework.png")

if __name__ == "__main__":
    create_research_framework()
