import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skfuzzy import cluster
from scipy.spatial import ConvexHull
import io

st.set_page_config(page_title="Clustering Produksi Daging", layout="wide")
st.title("📊 Aplikasi Clustering Fuzzy C-Means Produksi Daging di Indonesia")

# === Upload File ===
uploaded_file = st.file_uploader("Unggah file CSV produksi daging", type="csv")

if uploaded_file:
    # === Load dan tampilkan data awal ===
    df = pd.read_csv(uploaded_file)
    st.subheader("📌 Data Awal")
    st.dataframe(df)

    clicked = st.button("Clean Data")
    if clicked:
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        st.success(f"Data telah dibersihkan ({len(df) - len(df)} baris dihapus)")
        st.subheader("✅ Data Setelah Pembersihan")
        st.dataframe(df)

    # === FCM Clustering ===
    features = df.columns[1:]
    data = df[features].T.values
    n_clusters = st.number_input("Jumlah Cluster", 2, 10, 3, 1)

    cntr, u, _, _, _, _, _ = cluster.cmeans(data, c=n_clusters, m=2, error=0.005, maxiter=1000)
    labels = u.argmax(axis=0)
    df['Cluster'] = labels

    st.subheader("📥 Hasil Clustering")
    df.insert(0, 'No', np.arange(1, len(df) +1))
    st.dataframe(df, use_container_width=True, hide_index=True)

    # === Visualisasi Clustering ===
    st.subheader("📈 Visualisasi Clustering")

    selected_features = st.multiselect(
        "Pilih dua fitur untuk divisualisasikan:",
        list(features),
        default=list(features[:2])
    )

    if len(selected_features) == 2:
        feature1, feature2 = selected_features
        x_idx = list(features).index(feature1)
        y_idx = list(features).index(feature2)

        fig, ax = plt.subplots()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for i in range(n_clusters):
            cluster_data = data.T[labels == i]
            points = cluster_data[:, [x_idx, y_idx]]
            ax.scatter(points[:, 0], points[:, 1], label=f'Cluster {i+1}', alpha=0.6)

            # Polygon cluster
            if len(points) >= 3:
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    ax.plot(points[simplex, 0], points[simplex, 1], color=colors[i])
                ax.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=colors[i], alpha=0.2)

        # Tambahkan nomor provinsi
        for idx, row in enumerate(data.T):
            ax.text(row[x_idx], row[y_idx], str(idx + 1), fontsize=8, color='black',
                    ha='center', va='center')

        # Plot centroid
        ax.scatter(cntr[:, x_idx], cntr[:, y_idx], c='black', marker='^', s=200, label='Centroid')
        ax.set_title("Visualisasi Cluster")
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Pilih tepat dua fitur untuk menampilkan visualisasi.")
        
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button(
        label="📥 Unduh Visualisasi PNG",
        data=buf.getvalue(),
        file_name="visualisasi_cluster.png",
        mime="image/png"
    )

else:
    st.info("Silakan unggah file CSV terlebih dahulu.")
