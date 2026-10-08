import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

MODEL_PATH = "weather_model.pkl"
DATA_PATH = "data/weather.csv"

st.set_page_config(
    page_title="AI Weather Prediction",
    page_icon="🌦️",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def compare_models():
    df = load_data()
    features = [
        "Temperature_C", "Humidity_pct", "WindSpeed_kmh",
        "Pressure_hPa", "CloudCover_pct"
    ]
    X, y = df[features], df["Weather"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, max_depth=8
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=42
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7)
    }

    scores = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        scores.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred) * 100
        })

    return pd.DataFrame(scores).sort_values("Accuracy", ascending=False)

model = load_model()
df = load_data()

st.title("🌦️ AI Weather Prediction System")
st.write("A Machine Learning weather classification project built with Python, Scikit-learn and Streamlit.")

st.info("💡 Enter weather conditions below and let the trained Random Forest model predict the likely weather condition.")

st.sidebar.header("📍 Location")
city = st.sidebar.text_input("City", "Ghaziabad")
st.sidebar.caption("Location is currently used as a display label. The prediction is based on the weather values you enter.")

st.sidebar.header("📊 Dataset")
st.sidebar.write(f"Records: **{len(df)}**")
st.sidebar.write(f"Features: **5**")
st.sidebar.write(f"Classes: **{df['Weather'].nunique()}**")

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Data & Charts", "🤖 Model Comparison"])

with tab1:
    st.subheader(f"Weather prediction for {city}")

    col1, col2, col3 = st.columns(3)
    with col1:
        temperature = st.number_input(
            "Temperature (°C)", min_value=-10.0, max_value=50.0,
            value=28.0, step=0.5
        )
        humidity = st.number_input(
            "Humidity (%)", min_value=0.0, max_value=100.0,
            value=65.0, step=1.0
        )
    with col2:
        wind_speed = st.number_input(
            "Wind Speed (km/h)", min_value=0.0, max_value=100.0,
            value=12.0, step=0.5
        )
        pressure = st.number_input(
            "Atmospheric Pressure (hPa)", min_value=950.0, max_value=1080.0,
            value=1012.0, step=0.5
        )
    with col3:
        cloud_cover = st.number_input(
            "Cloud Cover (%)", min_value=0.0, max_value=100.0,
            value=45.0, step=1.0
        )

    if st.button("🔮 Predict Weather", type="primary", use_container_width=True):
        input_data = pd.DataFrame([{
            "Temperature_C": temperature,
            "Humidity_pct": humidity,
            "WindSpeed_kmh": wind_speed,
            "Pressure_hPa": pressure,
            "CloudCover_pct": cloud_cover
        }])

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        confidence = probabilities.max() * 100

        icons = {"Sunny": "☀️", "Cloudy": "☁️", "Rainy": "🌧️"}

        st.success(
            f"{icons.get(prediction, '🌦️')} Predicted Weather: **{prediction}**"
        )
        st.metric("Model Confidence", f"{confidence:.1f}%")

        result = pd.DataFrame({
            "Weather": model.classes_,
            "Probability (%)": probabilities * 100
        }).sort_values("Probability (%)", ascending=False)

        st.bar_chart(result.set_index("Weather")["Probability (%)"])

        st.subheader("Prediction probabilities")
        st.dataframe(
            result.style.format({"Probability (%)": "{:.1f}%"}),
            use_container_width=True,
            hide_index=True
        )

with tab2:
    st.subheader("📊 Weather Dataset Overview")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    st.subheader("Weather class distribution")
    counts = df["Weather"].value_counts()
    st.bar_chart(counts)

    st.subheader("Average weather measurements by class")
    numeric_cols = [
        "Temperature_C", "Humidity_pct", "WindSpeed_kmh",
        "Pressure_hPa", "CloudCover_pct"
    ]
    averages = df.groupby("Weather")[numeric_cols].mean().round(2)
    st.dataframe(averages, use_container_width=True)

    st.subheader("Feature relationship")
    feature = st.selectbox("Choose a feature", numeric_cols)
    st.scatter_chart(df, x=feature, y="Temperature_C")

with tab3:
    st.subheader("🤖 Compare Machine Learning Algorithms")
    st.write("The models below are trained and evaluated using the same train/test split.")

    scores = compare_models()
    st.dataframe(
        scores.style.format({"Accuracy": "{:.2f}%"}),
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(scores.set_index("Model")["Accuracy"])

    st.subheader("🌳 Random Forest Feature Importance")
    importance = pd.DataFrame({
        "Feature": model.feature_names_in_,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    st.bar_chart(importance.set_index("Feature")["Importance"])

st.divider()
st.caption("AICTE Machine Learning & AI Internship Project | Python • Pandas • Scikit-learn • Streamlit")
