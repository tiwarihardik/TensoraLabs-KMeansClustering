import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


st.title('TensoraLabs - KMeans Clustering')
st.write('Upload your dataset, select features, and perform KMeans clustering.')


if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_columns' not in st.session_state:
    st.session_state.X_columns = None
if 'clusters' not in st.session_state:
    st.session_state.clusters = None


file = st.file_uploader("Upload your CSV file", type=["csv"])
if file:
    df = pd.read_csv(file).dropna()  
    st.write("Preview of your data:")
    st.write(df.head())

    target = st.selectbox('Select column to cluster:', df.columns)
    features = st.multiselect('Select features for clustering', df.columns)

    if len(features) >= 2: 
        k_ = st.slider("Number of clusters (k)", min_value=2, max_value=10, value=3)

        if st.button('Run KMeans Clustering'):
            X = df[features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = KMeans(n_clusters=k_, random_state=42)
            df['Cluster'] = model.fit_predict(X_scaled)
            st.session_state.model = model
            st.write(df[['Cluster'] + features].head())
            if len(features) == 2:
                plt.figure(figsize=(8, 5))
                sns.scatterplot(x=features[0], y=features[1], hue='Cluster', data=df, palette='tab10')
                plt.title(f'KMeans Clustering with {k_} clusters')
                st.pyplot(plt)
                sil_score = silhouette_score(X_scaled, model.labels_)
                st.write("Silhouette Score (Accuracy): ", sil_score)

                if sil_score >= 0.5:
                    st.write("The clustering has good separation between clusters.")
                else:
                    st.write("The clustering has poor separation between clusters.")
                st.balloons()

            st.success("Clustering completed!")
if st.session_state.model:
    st.header("🔮 Make Predictions")

    user_input = {}
    for feature in features:
        if pd.api.types.is_numeric_dtype(df[feature]):
            user_input[feature] = st.number_input(f"{feature}:")
        else:
            user_input[feature] = st.selectbox(f"{feature}:", df[feature].unique())

    if st.button("📍 Predict Cluster"):
        input_df = pd.DataFrame([user_input])
        input_scaled = st.session_state.model.transform(input_df[features])
        pred_cluster = st.session_state.model.predict(input_scaled)[0]

        st.success(f"Predicted Cluster: {pred_cluster}")
