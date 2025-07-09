from flask import Flask, render_template, request, send_file, url_for
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import uuid

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PLOT_FOLDER'] = 'static/plots'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PLOT_FOLDER'], exist_ok=True)

# ---------- Dataset Analysis ----------
def analyze_dataset(df, preview_name='preview'):
    return {
        'original_shape': df.shape,
        'columns': list(df.columns),
        'data_types': df.dtypes.astype(str).to_dict(),
        'missing_values': df.isna().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'numeric_stats': {
            col: {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std()
            } for col in df.select_dtypes(include='number').columns
        },
        'text_stats': {
            col: {
                'unique_values': df[col].nunique(),
                'top_value': df[col].mode()[0] if not df[col].mode().empty else None,
                'top_freq': df[col].value_counts().max() if not df[col].empty else 0
            } for col in df.select_dtypes(include='object').columns
        },
        preview_name: df.head(5).to_dict(orient='records')
    }

# ---------- Data Cleaning ----------
def clean_csv(df):
    changes = {
        'columns_renamed': {},
        'missing_values_filled': {},
        'duplicates_removed': 0,
        'text_standardized': []
    }

    # 1. Clean column names
    new_cols = {}
    for col in df.columns:
        clean = re.sub(r'[^a-z0-9]+', '_', col.strip().lower()).strip('_')
        if clean != col:
            new_cols[col] = clean
    if new_cols:
        df.rename(columns=new_cols, inplace=True)
        changes['columns_renamed'] = new_cols

    # 2. Fill missing values
    for col in df.columns:
        missing = df[col].isna().sum()
        if missing > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_val = df[col].median()
                df[col].fillna(fill_val, inplace=True)
                changes['missing_values_filled'][col] = {
                    'count': missing,
                    'type': 'numeric',
                    'filled_with': f'median ({fill_val:.2f})'
                }
            else:
                df[col].fillna('unknown', inplace=True)
                changes['missing_values_filled'][col] = {
                    'count': missing,
                    'type': 'text',
                    'filled_with': "'unknown'"
                }

    # 3. Remove duplicates
    dups = df.duplicated().sum()
    if dups > 0:
        df.drop_duplicates(inplace=True)
        changes['duplicates_removed'] = dups

    # 4. Standardize text columns
    for col in df.select_dtypes(include='object').columns:
        if not df[col].str.strip().str.lower().equals(df[col]):
            df[col] = df[col].str.strip().str.lower()
            changes['text_standardized'].append(col)

    return df, changes

# ---------- Dashboard Plot Generation ----------
def generate_dashboard_plots(df):
    paths = []
    sns.set(style="whitegrid")

    # Null values bar chart
    nulls = df.isna().sum()
    if nulls.sum() > 0:
        plt.figure(figsize=(10, 4))
        nulls[nulls > 0].plot(kind='bar', color='salmon')
        plt.title('Missing Values per Column')
        plt.tight_layout()
        path = f"{app.config['PLOT_FOLDER']}/nulls_{uuid.uuid4().hex}.png"
        plt.savefig(path)
        paths.append(path)
        plt.close()

    # Correlation heatmap
    num_df = df.select_dtypes(include='number')
    if not num_df.empty:
        plt.figure(figsize=(8, 6))
        sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        path = f"{app.config['PLOT_FOLDER']}/corr_{uuid.uuid4().hex}.png"
        plt.savefig(path)
        paths.append(path)
        plt.close()

    # Numeric histograms
    for col in num_df.columns[:2]:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True, color='skyblue')
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
        path = f"{app.config['PLOT_FOLDER']}/hist_{col}_{uuid.uuid4().hex}.png"
        plt.savefig(path)
        paths.append(path)
        plt.close()

    # Categorical bar charts
    cat_cols = df.select_dtypes(include='object').columns[:2]
    for col in cat_cols:
        plt.figure(figsize=(6, 4))
        df[col].value_counts().head(10).plot(kind='bar', color='green')
        plt.title(f'Top Categories in {col}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        path = f"{app.config['PLOT_FOLDER']}/bar_{col}_{uuid.uuid4().hex}.png"
        plt.savefig(path)
        paths.append(path)
        plt.close()

    return paths

# ---------- Flask Routes ----------
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')

        if not file or file.filename == '':
            return render_template('index.html', error="Please upload a CSV file.")

        if not file.filename.lower().endswith('.csv'):
            return render_template('index.html', error="Only CSV files are allowed.")

        try:
            df = pd.read_csv(file)

            original_analysis = analyze_dataset(df, preview_name='preview_original')
            cleaned_df, changes = clean_csv(df)
            cleaned_analysis = analyze_dataset(cleaned_df, preview_name='preview_cleaned')

            cleaned_filename = f"cleaned_{file.filename}"
            cleaned_path = os.path.join(app.config['UPLOAD_FOLDER'], cleaned_filename)
            cleaned_df.to_csv(cleaned_path, index=False)

            # Generate dashboard plots
            plot_paths = generate_dashboard_plots(cleaned_df)

            return render_template('result.html',
                                   original_filename=file.filename,
                                   cleaned_filename=cleaned_filename,
                                   original_analysis=original_analysis,
                                   cleaned_analysis=cleaned_analysis,
                                   changes=changes,
                                   plot_paths=plot_paths)

        except Exception as e:
            return render_template('index.html', error=f"Error: {str(e)}")

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
