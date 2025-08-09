from flask import Flask, request, render_template, send_file, redirect, url_for
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        
        try:
            # Read CSV
            df = pd.read_csv(file)

            # Get parameters from form
            eps = float(request.form.get('eps', 0.5))
            min_samples = int(request.form.get('min_samples', 5))

            # Select relevant columns and drop NA
            X = df[['Day', 'Night']].dropna()

            # Scale and cluster
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            db = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = db.fit_predict(X_scaled)

            df_clean = X.copy()
            df_clean['cluster'] = clusters

            # Calculate cluster statistics
            cluster_count = len(set(clusters)) - (1 if -1 in clusters else 0)
            noise_points = list(clusters).count(-1)

            # Plot clusters
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(df_clean['Day'], df_clean['Night'], 
                                c=df_clean['cluster'], cmap='viridis', s=50)
            ax.set_xlabel("Day Noise Level (dB)")
            ax.set_ylabel("Night Noise Level (dB)")
            ax.set_title(f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})")
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Cluster ID')

            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
            plt.close(fig)
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()

            # Save clustered data
            csv_data = df_clean.to_csv(index=False)

            return render_template('result.html', 
                                plot_url=plot_url, 
                                csv_data=csv_data,
                                cluster_count=cluster_count,
                                noise_points=noise_points,
                                eps_value=eps,
                                min_samples=min_samples)

        except Exception as e:
            return f"An error occurred: {str(e)}"

    return render_template('index.html')

@app.route('/download')
def download():
    csv_data = request.args.get('data')
    if not csv_data:
        return "No data to download"
    return send_file(
        io.BytesIO(csv_data.encode()),
        mimetype='text/csv',
        download_name='noise_clusters.csv',
        as_attachment=True
    )

if __name__ == '__main__':
    app.run(debug=True)