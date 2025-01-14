# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score

# Load Dataset
file_path = './Dataset-terbaru.csv'
dataset = pd.read_csv(file_path)

# Data Cleaning
def clean_numeric_column(column):
    return column.str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)

dataset['Tanggal'] = pd.to_datetime(dataset['Tanggal'], format='%d/%m/%Y')
dataset['Terakhir'] = clean_numeric_column(dataset['Terakhir'])
dataset['Pembukaan'] = clean_numeric_column(dataset['Pembukaan'])
dataset['Tertinggi'] = clean_numeric_column(dataset['Tertinggi'])
dataset['Terendah'] = clean_numeric_column(dataset['Terendah'])
dataset['Perubahan%'] = dataset['Perubahan%'].str.replace('%', '').str.replace(',', '.').astype(float)
dataset = dataset.drop(columns=['Vol.'], errors='ignore')

# Aggregate Monthly Data
dataset['Month'] = dataset['Tanggal'].dt.to_period('M')
monthly_data = dataset.groupby('Month').agg({
    'Terakhir': 'mean',
    'Pembukaan': 'mean',
    'Tertinggi': 'mean',
    'Terendah': 'mean',
    'Perubahan%': 'mean'
}).reset_index()

# Clustering with K-Means
optimal_k = 3  # Jumlah cluster
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
features = monthly_data.drop(columns=['Month']).values
monthly_data['Cluster'] = kmeans.fit_predict(features)

# Evaluasi Performa Model
dbs = davies_bouldin_score(features, kmeans.labels_)  # Davies-Bouldin Score
ss = silhouette_score(features, kmeans.labels_)  # Silhouette Score
inertia = kmeans.inertia_  # Inertia
chs = calinski_harabasz_score(features, kmeans.labels_)  # Calinski-Harabasz Index

# Cetak Hasil Evaluasi Performa
print("\nEvaluasi Performa Model:")
print(f"Davies-Bouldin Score: {dbs}")
print(f"Silhouette Score: {ss}")
print(f"Inertia (Sum of Squared Distances): {inertia}")
print(f"Calinski-Harabasz Index: {chs}")

# Menentukan Bulan untuk Analisis
print("Pilih bulan yang tersedia untuk analisis:")
print(monthly_data['Month'].astype(str).tolist())

# Input dari pengguna
selected_month = input("Masukkan bulan yang ingin dianalisis (format: YYYY-MM): ")

# Filter Data untuk Bulan yang Dipilih
selected_data = monthly_data[monthly_data['Month'].astype(str) == selected_month]

if selected_data.empty:
    print(f"Bulan {selected_month} tidak ditemukan dalam dataset.")
else:
    # Analisis Pergerakan
    cluster_label = selected_data['Cluster'].iloc[0]
    if cluster_label == 0:
        movement = "Pergerakan harga cenderung **turun**."
    elif cluster_label == 1:
        movement = "Pergerakan harga cenderung **naik**."
    else:
        movement = "Pergerakan harga cenderung **sideways**."

    print(f"\nAnalisis Bulan {selected_month}:")
    print(movement)

    # Visualisasi Scatter Plot Cluster
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=monthly_data['Pembukaan'],
    y=monthly_data['Terakhir'],
    hue=monthly_data['Cluster'],
    palette='viridis',
    s=100
)
plt.title('K-Means Clustering of Bitcoin Monthly Data', fontsize=16)
plt.xlabel('Average Opening Price', fontsize=12)
plt.ylabel('Average Closing Price', fontsize=12)
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Filter Data Harian untuk Bulan yang Dipilih
daily_data = dataset[dataset['Tanggal'].dt.to_period('M') == selected_month]

# Visualisasi Data Harian
plt.figure(figsize=(12, 6))
plt.plot(daily_data['Tanggal'], daily_data['Pembukaan'], label='Pembukaan', marker='o')
plt.plot(daily_data['Tanggal'], daily_data['Tertinggi'], label='Tertinggi', marker='o')
plt.plot(daily_data['Tanggal'], daily_data['Terendah'], label='Terendah', marker='o')
plt.plot(daily_data['Tanggal'], daily_data['Terakhir'], label='Penutupan', marker='o')
plt.title(f'Pergerakan Harian Bitcoin untuk Bulan {selected_month}', fontsize=16)
plt.xlabel('Tanggal', fontsize=12)
plt.ylabel('Harga', fontsize=12)
plt.legend(title='Harga Harian')
plt.grid(True)
plt.show()

# Tampilkan Data Harian
print("\nDetail Data Harian:")
print(daily_data[['Tanggal', 'Pembukaan', 'Tertinggi', 'Terendah', 'Terakhir']])
