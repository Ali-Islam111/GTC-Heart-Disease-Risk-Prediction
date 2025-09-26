import streamlit as st
import joblib
import pandas as pd

# Load the pipeline
pipeline = joblib.load("final_tuned_pipeline.pkl")

st.title("🫀 Heart Disease Predictor")
st.write("Enter your Data to predict the risk of heart disease:")

# User Input layout
col1, col2 = st.columns(2)

# Column 1
age = col1.number_input("Age", min_value=1, max_value=100, value=54)
sex = col1.selectbox("Sex", options=["Female", "Male"], index=1)    # keep 0/1 mapping
cp = col1.selectbox("Chest Pain Type",options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],index=1)  # mapped 0..3
restecg = col1.selectbox("Resting ECG",options=["Normal", "ST-T Wave Abnormality", "Left ventricular hypertrophy"],index=1)  # mapped 0..2
exang = col1.selectbox("Exercise Induced Angina", options=["No", "Yes"], index=0)  # 0/1
slope = col1.selectbox("Slope", options=["Up sloping", "Flat", "Down Sloping"], index=1)  # 0..2

# Column 2
ca = col2.selectbox("Number of major vessels (0..3)", options=[0, 1, 2, 3], index=0)
thal = col2.selectbox("Thal", options=["normal", "fixed defect", "reversible"], index=1)  # 1,2,3
trestbps = col2.number_input("Resting blood pressure", min_value=50, max_value=250, value=130)
chol = col2.number_input("Cholesterol", min_value=50, max_value=600, value=246)
thalach = col2.number_input("Max heart rate achieved", min_value=50, max_value=260, value=150)
oldpeak = col2.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# Mapping dictionaries
sex_map = {"Female": 0, "Male": 1}
exang_map = {"No": 0, "Yes": 1}
cp_map = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}
restecg_map = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left ventricular hypertrophy": 2
}
slope_map = {"Up sloping": 0, "Flat": 1, "Down Sloping": 2}
thal_map = {"normal": 1, "fixed defect": 2, "reversible": 3}

# Apply mappings
sex_val = sex_map[sex]
exang_val = exang_map[exang]
cp_val = cp_map[cp]
restecg_val = restecg_map[restecg]
slope_val = slope_map[slope]
thal_val = thal_map[thal]

# Build DataFrame and ensure categorical columns are ints
input_data = pd.DataFrame([{
    "age": age,
    "sex": sex_val,
    "cp": cp_val,
    "restecg": restecg_val,
    "exang": exang_val,
    "slope": slope_val,
    "ca": ca,
    "thal": thal_val,
    "trestbps": trestbps,
    "chol": chol,
    "thalach": thalach,
    "oldpeak": oldpeak
}])

# Cast categorical columns to int (safer for ColumnTransformer expectations)
for col in ["sex", "cp", "restecg", "exang", "slope", "ca", "thal"]:
    input_data[col] = input_data[col].astype(int)

# Optional sanity check: confirm pipeline's required columns exist
required_cols = ['age','trestbps','chol','thalach','oldpeak','sex','cp','restecg','exang','slope','ca','thal']
missing = [c for c in required_cols if c not in input_data.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
else:
    if st.button("Predict"):
        prob = pipeline.predict_proba(input_data)[0,1]
        pred = pipeline.predict(input_data)[0]
        if pred == 1:
            st.error("Prediction Result :")
            st.markdown("<h2 style='text-align: center; color: red;'>Heart Disease Detected 🤒</h2>", unsafe_allow_html=True)
        else:
            st.success("Prediction Result:")
            st.markdown("<h2 style='text-align: center; color: green;'>No Heart Disease 👍</h2>", unsafe_allow_html=True)
        st.write(f"Probability of disease: {prob:.2f}")


# ---- Data exploration / visualization section ----
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

@st.cache_data
def load_data(path="02_heart_disease_preprocessed.csv"):
    df = pd.read_csv(path)
    return df

def ensure_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    return missing

# Load
st.header("📊 Explore Heart Disease Data")
df = load_data()

# Basic info
st.write(f"Loaded `{len(df)}` rows and `{len(df.columns)}` columns.")
if st.checkbox("Show raw data"):
    st.dataframe(df.head(100))

# Set up filters
st.sidebar.header("Explore filters")
min_age = int(df['age'].min()) if 'age' in df.columns else 0
max_age = int(df['age'].max()) if 'age' in df.columns else 120
age_range = st.sidebar.slider("Age range", min_value=min_age, max_value=max_age,
                              value=(min_age, max_age))

sex_options = None
if 'sex' in df.columns:
    sex_unique = sorted(df['sex'].dropna().unique().tolist())
    # if encoded as 0/1 show labels otherwise show values
    if set(sex_unique).issubset({0,1}):
        sex_display = {0: "Female", 1: "Male"}
        sex_options = st.sidebar.multiselect("Sex", options=[0,1],
                                             default=[0,1],
                                             format_func=lambda x: sex_display.get(x, str(x)))
    else:
        sex_options = st.sidebar.multiselect("Sex", options=sex_unique, default=sex_unique)

# subset df
df_sub = df.copy()
if 'age' in df_sub.columns:
    df_sub = df_sub[(df_sub['age'] >= age_range[0]) & (df_sub['age'] <= age_range[1])]
if sex_options is not None and 'sex' in df_sub.columns:
    df_sub = df_sub[df_sub['sex'].isin(sex_options)]

st.write(f"Using {len(df_sub)} rows after filters.")

# Visualization chooser
viz = st.selectbox("Choose visualization", [
    "Target distribution",
    "Age distribution by target",
    "Chest Pain type vs Disease rate",
    "Cholesterol distribution by target",
    "Correlation heatmap"
    
])

# 1) Target distribution
if viz == "Target distribution":
    if 'target' not in df_sub.columns:
        st.error("Column 'target' not found in dataset.")
    else:
        vc = df_sub['target'].value_counts().sort_index()
        labels = [str(idx) for idx in vc.index]
        fig = px.bar(x=labels, y=vc.values, labels={'x':'Target','y':'Count'},
                     title="Target distribution (0 = no disease, 1 = disease)")
        st.plotly_chart(fig, use_container_width=True)

# 2) Age distribution by target
elif viz == "Age distribution by target":
    if not {'age','target'}.issubset(df_sub.columns):
        st.error("Columns 'age' and/or 'target' missing.")
    else:
        fig = px.histogram(df_sub, x='age', color='target', nbins=25, barmode='overlay',
                           labels={'target':'Target'}, title="Age distribution by target")
        st.plotly_chart(fig, use_container_width=True)

# 3) Chest Pain type vs Disease rate
elif viz == "Chest Pain type vs Disease rate":
    if 'cp' not in df_sub.columns or 'target' not in df_sub.columns:
        st.error("Columns 'cp' and/or 'target' missing.")
    else:
        # compute disease rate per cp
        grp = df_sub.groupby('cp')['target'].agg(['count','mean']).reset_index().rename(columns={'mean':'disease_rate'})
        # display both counts and rate
        fig = go.Figure()
        fig.add_trace(go.Bar(x=grp['cp'].astype(str), y=grp['count'], name='count', yaxis='y1'))
        fig.add_trace(go.Line(x=grp['cp'].astype(str), y=grp['disease_rate'], name='disease rate', yaxis='y2', marker=dict(size=8)))
        fig.update_layout(title="Chest pain type: count and disease rate",
                          xaxis_title="cp",
                          yaxis=dict(title='count'),
                          yaxis2=dict(title='disease rate', overlaying='y', side='right', tickformat=".0%"))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(grp)

# 4) Cholesterol distribution by target
elif viz == "Cholesterol distribution by target":
    if 'chol' not in df_sub.columns or 'target' not in df_sub.columns:
        st.error("Columns 'chol' and/or 'target' missing.")
    else:
        fig = px.violin(df_sub, x='target', y='chol', box=True, points='outliers',
                        labels={'target':'Target','chol':'Cholesterol'},
                        title="Cholesterol distribution by target (violin + box)")
        st.plotly_chart(fig, use_container_width=True)

# 5) Correlation heatmap
elif viz == "Correlation heatmap":
    # pick numeric columns that likely exist
    numeric_cols = df_sub.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Not enough numeric columns for correlation.")
    else:
        corr = df_sub[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation matrix (numeric columns)")
        st.plotly_chart(fig, use_container_width=True)


