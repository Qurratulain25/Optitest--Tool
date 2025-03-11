# from flask import Flask, render_template, request, redirect, url_for, session
# from flask_session import Session
# import os
# import pandas as pd
# from werkzeug.utils import secure_filename

# # Import custom modules
# from data_preprocessing import preprocess_data
# from mapping import map_criteria, map_alternatives, renormalize_criteria
# from fahp import calculate_fahp
# from visualization import plot_quartile_distribution

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Import default config
# from config import ALTERNATIVE_MAPPING, CRITERIA_LIST

# app = Flask(__name__)
# app.secret_key = "some_random_secret_key"
# app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['SESSION_TYPE'] = 'filesystem'
# app.config['SESSION_PERMANENT'] = False
# app.config['SESSION_FILE_DIR'] = os.path.join(os.getcwd(), 'flask_session')
# Session(app)

# # Ensure required directories exist
# for folder in [app.config['UPLOAD_FOLDER'], 'static']:
#     os.makedirs(folder, exist_ok=True)
#     os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

# @app.route('/')
# def home():
#     return redirect(url_for('dashboard'))

# @app.route('/dashboard', methods=['GET', 'POST'])
# def dashboard():
#     if request.method == 'POST':
#         file = request.files.get('file')
#         if not file:
#             return "No file provided", 400

#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)

#         df = pd.read_excel(filepath)
#         csv_path = os.path.splitext(filepath)[0] + ".csv"
#         df.to_csv(csv_path, index=False)
#         session['csv_path'] = csv_path
#         session['uploaded_filename'] = filename

#         numeric_cols = df.select_dtypes(include=[float, int]).columns
#         categorical_cols = df.select_dtypes(include=[object]).columns
#         numeric_count = len(numeric_cols)
#         categorical_count = len(categorical_cols)
#         missing_values = df.isnull().sum().sum()
#         rows, cols = df.shape
#         df_head = df.head().to_html(classes="dataframe w-full divide-y divide-gray-200", index=False)

#         return render_template(
#             'dashboard.html',
#             uploaded=True,
#             filename=filename,
#             numeric_count=numeric_count,
#             categorical_count=categorical_count,
#             missing_values=missing_values,
#             total_rows=rows,
#             total_columns=cols,
#             df_head=df_head
#         )

#     return render_template('dashboard.html', uploaded=False)

# @app.route('/preprocessing', methods=['GET', 'POST'])
# def preprocessing_page():
#     if 'csv_path' not in session:
#         return redirect(url_for('dashboard'))

#     csv_path = session['csv_path']
#     df = pd.read_csv(csv_path)
#     numeric_cols = df.select_dtypes(include=['number']).columns
#     categorical_cols = df.select_dtypes(include=['object']).columns

#     if request.method == 'POST':
#         missing_method = request.form.get('missing_numeric_method', 'none')
#         df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

#         if missing_method == 'mean':
#             df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
#         elif missing_method == 'drop':
#             df.dropna(inplace=True)

#         df.to_csv(csv_path, index=False)
#         session['preprocessing_done'] = True

#     numeric_count = len(numeric_cols)
#     categorical_count = len(categorical_cols)
#     missing_values = df.isnull().sum().sum()
#     rows, cols = df.shape

#     return render_template(
#         'preprocessing.html',
#         updated=session.get('preprocessing_done', False),
#         numericCols=numeric_count,
#         categoricalCols=categorical_count,
#         missingValues=missing_values,
#         totalRows=rows,
#         totalColumns=cols
#     )

# @app.route('/criteria_alternatives', methods=['GET', 'POST'])
# def criteria_alternatives_page():
#     if request.method == 'GET':
#         return render_template('criteria_alternatives.html',
#                                default_criteria=CRITERIA_LIST,
#                                default_alternatives=list(ALTERNATIVE_MAPPING.keys()))

#     confirmed_criteria = request.form.getlist('criteria_select')
#     confirmed_alternatives = request.form.getlist('alternative_select')

#     user_weights = {crit: request.form.get(f'weight_0_{j}', "equally_important") for j, crit in enumerate(confirmed_criteria)}

#     fahp_wts, fahp_summary = calculate_fahp(user_weights, confirmed_criteria)
#     # Store FAHP weights in a formatted way instead of debug logs
#     fahp_details = "<ul>"
#     for crit, weight in zip(confirmed_criteria, fahp_wts):
#         fahp_details += f"<li><strong>{crit}:</strong> {weight:.4f}</li>"
#     fahp_details += "</ul>"

#     # Save clean FAHP details in session (not debug logs)
#     session['fahp_details'] = fahp_details
   



#     csv_path = session.get('csv_path')
#     if not csv_path:
#         return redirect(url_for('dashboard'))
#     df = pd.read_csv(csv_path)

#     df, _ = map_criteria(df)
#     df = map_alternatives(df, 'label', confirmed_alternatives)

#     df['FAHP_Score'] = df[confirmed_criteria].dot(fahp_wts)
#      # Ensure graph is saved and session stores its path
#     graph_path = "fa_hpscore_plot.png"
#     plot_quartile_distribution(df, output_path=f"static/{graph_path}")
#     session['graph_path'] = graph_path

#     df['FAHP_Score'] = (df['FAHP_Score'] - df['FAHP_Score'].mean()) / (df['FAHP_Score'].std() + 1e-8)

#     q1_threshold = df['FAHP_Score'].quantile(0.20)
#     q2_threshold = df['FAHP_Score'].quantile(0.50)
#     q3_threshold = df['FAHP_Score'].quantile(0.80)

#     df['Quartile'] = df['FAHP_Score'].apply(lambda score: "Q1" if score <= q1_threshold else "Q2" if score <= q2_threshold else "Q3" if score <= q3_threshold else "Q4")

#     quartile_counts = df['Quartile'].value_counts()

#     q4_weight = 1.0
#     q3_weight = 0.75 if quartile_counts.get('Q4', 0) > quartile_counts.get('Q3', 0) else 0.85
#     q2_weight = 0.5 if quartile_counts.get('Q3', 0) > quartile_counts.get('Q2', 0) else 0.65

#     optimized_size = (q4_weight * quartile_counts.get('Q4', 0) +
#                       q3_weight * quartile_counts.get('Q3', 0) +
#                       q2_weight * quartile_counts.get('Q2', 0))

#     original_size = len(df)
#     optimized_size = min(optimized_size, original_size)
#     reduction_pct = ((original_size - optimized_size) / original_size) * 100
#     coverage_pct = (optimized_size / original_size) * 100

#     ranking_table = df[['label', 'FAHP_Score', 'Quartile']].sort_values('FAHP_Score', ascending=False)\
#                     .to_html(classes="results-table min-w-full divide-y divide-gray-200", index=False)

#     session.update({
#         'selected_criteria': confirmed_criteria,
#         'selected_alternatives': confirmed_alternatives,
#         'weights_table': ranking_table,
#         'original_size': original_size,
#         'optimized_size': optimized_size,
#         'reduction_pct': reduction_pct,
#         'coverage_pct': coverage_pct
#     })

#     return redirect(url_for('results'))

# @app.route('/results')
# def results():
#     return render_template('results.html', **session)

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template, request, redirect, url_for, session
from flask_session import Session
import os
import pandas as pd
from werkzeug.utils import secure_filename

# Import custom modules
from data_preprocessing import preprocess_data
from mapping import map_criteria, map_alternatives, renormalize_criteria
from fahp import calculate_fahp
from visualization import plot_quartile_distribution

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Import default config
from config import ALTERNATIVE_MAPPING, CRITERIA_LIST

app = Flask(__name__)
app.secret_key = "some_random_secret_key"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_FILE_DIR'] = os.path.join(os.getcwd(), 'flask_session')
Session(app)

# Ensure required directories exist
for folder in [app.config['UPLOAD_FOLDER'], 'static']:
    os.makedirs(folder, exist_ok=True)
    os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

@app.route('/')
def home():
    return redirect(url_for('dashboard'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return "No file provided", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        df = pd.read_excel(filepath)
        csv_path = os.path.splitext(filepath)[0] + ".csv"
        df.to_csv(csv_path, index=False)
        session['csv_path'] = csv_path
        session['uploaded_filename'] = filename

        numeric_cols = df.select_dtypes(include=[float, int]).columns
        categorical_cols = df.select_dtypes(include=[object]).columns
        numeric_count = len(numeric_cols)
        categorical_count = len(categorical_cols)
        missing_values = df.isnull().sum().sum()
        rows, cols = df.shape
        df_head = df.head().to_html(classes="dataframe w-full divide-y divide-gray-200", index=False)

        return render_template(
            'dashboard.html',
            uploaded=True,
            filename=filename,
            numeric_count=numeric_count,
            categorical_count=categorical_count,
            missing_values=missing_values,
            total_rows=rows,
            total_columns=cols,
            df_head=df_head
        )

    return render_template('dashboard.html', uploaded=False)

@app.route('/preprocessing', methods=['GET', 'POST'])
def preprocessing_page():
    if 'csv_path' not in session:
        return redirect(url_for('dashboard'))

    csv_path = session['csv_path']
    df = pd.read_csv(csv_path)
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    if request.method == 'POST':
        missing_method = request.form.get('missing_numeric_method', 'none')
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        if missing_method == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif missing_method == 'drop':
            df.dropna(inplace=True)

        df.to_csv(csv_path, index=False)
        session['preprocessing_done'] = True

    numeric_count = len(numeric_cols)
    categorical_count = len(categorical_cols)
    missing_values = df.isnull().sum().sum()
    rows, cols = df.shape

    return render_template(
        'preprocessing.html',
        updated=session.get('preprocessing_done', False),
        numericCols=numeric_count,
        categoricalCols=categorical_count,
        missingValues=missing_values,
        totalRows=rows,
        totalColumns=cols
    )

@app.route('/criteria_alternatives', methods=['GET', 'POST'])
def criteria_alternatives_page():
    if request.method == 'GET':
        return render_template('criteria_alternatives.html',
                               default_criteria=CRITERIA_LIST,
                               default_alternatives=list(ALTERNATIVE_MAPPING.keys()))

    confirmed_criteria = request.form.getlist('criteria_select')
    confirmed_alternatives = request.form.getlist('alternative_select')

    user_weights = {crit: request.form.get(f'weight_0_{j}', "equally_important") for j, crit in enumerate(confirmed_criteria)}

    fahp_wts, fahp_summary = calculate_fahp(user_weights, confirmed_criteria)

    # Store FAHP weights in a formatted way (NOT DEBUG LOGS)
    fahp_details = "<ul>"
    for crit, weight in zip(confirmed_criteria, fahp_wts):
        fahp_details += f"<li><strong>{crit}:</strong> {weight:.4f}</li>"
    fahp_details += "</ul>"

    # Save clean FAHP details in session (not debugging output)
    session['fahp_details'] = fahp_details

    csv_path = session.get('csv_path')
    if not csv_path:
        return redirect(url_for('dashboard'))
    df = pd.read_csv(csv_path)

    df, _ = map_criteria(df)
    df = map_alternatives(df, 'label', confirmed_alternatives)

    df['FAHP_Score'] = df[confirmed_criteria].dot(fahp_wts)

    # **Quartile Assignment**
    df['FAHP_Score'] = (df['FAHP_Score'] - df['FAHP_Score'].mean()) / (df['FAHP_Score'].std() + 1e-8)
    q1_threshold = df['FAHP_Score'].quantile(0.20)
    q2_threshold = df['FAHP_Score'].quantile(0.50)
    q3_threshold = df['FAHP_Score'].quantile(0.80)

    df['Quartile'] = df['FAHP_Score'].apply(
        lambda score: "Q1" if score <= q1_threshold else "Q2" if score <= q2_threshold else "Q3" if score <= q3_threshold else "Q4"
    )

    quartile_counts = df['Quartile'].value_counts()

    q4_weight = 1.0
    q3_weight = 0.75 if quartile_counts.get('Q4', 0) > quartile_counts.get('Q3', 0) else 0.85
    q2_weight = 0.5 if quartile_counts.get('Q3', 0) > quartile_counts.get('Q2', 0) else 0.65

    optimized_size = (q4_weight * quartile_counts.get('Q4', 0) +
                      q3_weight * quartile_counts.get('Q3', 0) +
                      q2_weight * quartile_counts.get('Q2', 0))

    original_size = len(df)
    optimized_size = min(optimized_size, original_size)
    reduction_pct = ((original_size - optimized_size) / original_size) * 100
    coverage_pct = (optimized_size / original_size) * 100

    ranking_table = df[['label', 'FAHP_Score', 'Quartile']].sort_values('FAHP_Score', ascending=False)\
                    .to_html(classes="results-table min-w-full divide-y divide-gray-200", index=False)

    # **Now generate the graph AFTER quartile assignment**
    graph_path = "fa_hpscore_plot.png"
    plot_quartile_distribution(df, output_path=f"static/{graph_path}")
    session['graph_path'] = graph_path

    session.update({
        'selected_criteria': confirmed_criteria,
        'selected_alternatives': confirmed_alternatives,
        'weights_table': ranking_table,
        'original_size': original_size,
        'optimized_size': optimized_size,
        'reduction_pct': reduction_pct,
        'coverage_pct': coverage_pct
    })

    return redirect(url_for('results'))

@app.route('/results')
def results():
    return render_template('results.html', **session)

if __name__ == '__main__':
    app.run(debug=True)
