import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="HR Prediction App", layout="wide")

# =====================================================
# FIXED DATA PATH
# =====================================================
DATA_PATH = r"D:\Guvi-Projects\Project 3\Employee-Attrition - Employee-Attrition.csv"

# =====================================================
# LOAD PICKLES
# =====================================================
@st.cache_resource
def load_pickle(path):
    return pickle.load(open(path, "rb"))

attrition_model = load_pickle("attrition_model.pkl")
attrition_scaler = load_pickle("scaler.pkl")

performance_model = load_pickle("performance_model.pkl")
performance_scaler = load_pickle("performance_scaler.pkl")

promotion_model = load_pickle("promotion_model.pkl") 
promotion_scaler = load_pickle("promotion_scaler.pkl")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_resource
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# =====================================================
# COLUMNS USED IN MODEL
# =====================================================
numeric_cols = [
    'Age', 'MonthlyIncome', 'DistanceFromHome', 'TotalWorkingYears',
    'YearsAtCompany', 'YearsInCurrentRole', 'PercentSalaryHike',
    'JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance'
]

# =====================================================
# DUMMY ENCODING FOR ATTRITION
# =====================================================
def get_feature_columns(df):
    df_enc = df.copy()
    df_enc['OverTime'] = df_enc['OverTime'].map({'No': 0, 'Yes': 1})
    df_enc['Gender'] = df_enc['Gender'].map({'Female': 0, 'Male': 1})

    df_enc = pd.get_dummies(df_enc, columns=['JobRole', 'MaritalStatus', 'BusinessTravel'], drop_first=True)

    df_enc = df_enc.drop(columns=['Attrition'])
    return df_enc.columns.tolist()

attrition_feature_cols = get_feature_columns(df[
    numeric_cols + ['OverTime', 'Gender', 'JobRole', 'MaritalStatus', 'BusinessTravel', 'Attrition']
])

# =====================================================
# BUILD ATTRITION INPUT
# =====================================================
def build_attrition_input(data_dict):
    df_input = pd.DataFrame([data_dict])

    df_input['OverTime'] = df_input['OverTime'].map({'No': 0, 'Yes': 1})
    df_input['Gender'] = df_input['Gender'].map({'Female': 0, 'Male': 1})

    df_input = pd.get_dummies(df_input, columns=['JobRole', 'MaritalStatus', 'BusinessTravel'], drop_first=True)
    df_input = df_input.reindex(columns=attrition_feature_cols, fill_value=0)

    df_input[numeric_cols] = attrition_scaler.transform(df_input[numeric_cols])
    return df_input

# =====================================================
# SIDEBAR MENU
# =====================================================
menu = st.sidebar.radio(
    "📌 Navigation",
    ["🏠 Home", "🔮 Predict Attrition", "⭐ Performance Prediction", "🎓 Promotion Prediction"]
)

# =====================================================
# HOME PAGE
# =====================================================
if menu == "🏠 Home":
    st.title("📊 Employee Insights Dashboard")

    st.dataframe(df, use_container_width=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🔴 High-Risk Employees")
        if st.checkbox("Show High Attrition"):
            st.dataframe(df[df["Attrition"] == "Yes"], use_container_width=True)

    with col2:
        st.subheader("🙂 High Job Satisfaction")
        if st.checkbox("Show High Satisfaction"):
            st.dataframe(df[df["JobSatisfaction"] >= 4], use_container_width=True)

    with col3:
        st.subheader("⭐ High Performance")
        if st.checkbox("Show High Performers"):
            st.dataframe(df[df["PerformanceRating"] >= 4], use_container_width=True)

# =====================================================
# ATTRITION PREDICTION PAGE
# =====================================================
elif menu == "🔮 Predict Attrition":

    st.title("🔮 Employee Attrition Prediction")

    st.markdown("### 📘 Employee Dataset Preview")
   

    st.dataframe(
        df[["Attrition", "Age", "YearsAtCompany", "MonthlyIncome",
            "DistanceFromHome", "YearsInCurrentRole", "TotalWorkingYears",
            "PercentSalaryHike","JobSatisfaction","EnvironmentSatisfaction","WorkLifeBalance",
            "OverTime","Gender","JobRole","MaritalStatus","BusinessTravel"]],
        use_container_width=True
    )

    st.markdown("---")

    # --------------------------------------------------
    # CLEAN ALIGNED INPUT SECTION
    # --------------------------------------------------
    st.markdown("""
    <div style='padding:20px; background:white; border-radius:12px; 
    box-shadow:0px 4px 10px rgba(0,0,0,0.05); margin-top:10px;'>
    <h2>🧾 Enter Employee Details</h2>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # BASIC WORK SECTION
    st.subheader("📌 Basic Work Information")
    age = st.number_input("Age", 18, 60)
    years_company = st.number_input("Years At Company", 0, 40)
    income = st.number_input("Monthly Income", 1000, 60000)

    st.write("")

    st.subheader("📌 Experience Details")
    distance = st.number_input("Distance From Home", 1, 30)
    years_role = st.number_input("Years In Current Role", 0, 20)
    total_work_years = st.number_input("Total Working Years", 0, 40)

    st.write("")

    st.subheader("📌 Salary & Growth")
    salary_hike = st.number_input("Percent Salary Hike", 1, 100)

    st.write("")

    st.subheader("📌 Satisfaction Scores")
    job_sat = st.slider("Job Satisfaction", 1, 4)
    env_sat = st.slider("Environment Satisfaction", 1, 4)
    worklife = st.slider("Work-Life Balance", 1, 4)

    st.write("")

    st.subheader("📌 Work Style & Personal Info")
    overtime = st.selectbox("OverTime", ["No", "Yes"])
    gender = st.selectbox("Gender", ["Female", "Male"])

    st.write("")

    st.subheader("📌 Job Attributes")
    jobrole = st.selectbox("Job Role", sorted(df["JobRole"].unique()))
    marital = st.selectbox("Marital Status", sorted(df["MaritalStatus"].unique()))
    travel = st.selectbox("Business Travel", sorted(df["BusinessTravel"].unique()))

    # --------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------
    if st.button("Predict Attrition", use_container_width=True):
        input_dict = {
            "Age": age,
            "MonthlyIncome": income,
            "DistanceFromHome": distance,
            "TotalWorkingYears": total_work_years,
            "YearsAtCompany": years_company,
            "YearsInCurrentRole": years_role,
            "PercentSalaryHike": salary_hike,
            "JobSatisfaction": job_sat,
            "EnvironmentSatisfaction": env_sat,
            "WorkLifeBalance": worklife,
            "OverTime": overtime,
            "Gender": gender,
            "JobRole": jobrole,
            "MaritalStatus": marital,
            "BusinessTravel": travel
        }

        X_attr = build_attrition_input(input_dict)
        pred = attrition_model.predict(X_attr)[0]

        if pred == 1:
            st.error("⚠ Employee is LIKELY to leave!")
        else:
            st.success("✔ Employee is NOT likely to leave.")

        #st.subheader("🔎 Processed Input Data")
        #st.dataframe(X_attr, use_container_width=True)

# =====================================================
# PERFORMANCE PREDICTION
# =====================================================
elif menu == "⭐ Performance Prediction":

    st.title("⭐ Performance Rating Prediction")

    st.dataframe(
        df[["Education", "JobInvolvement", "JobLevel",
            "MonthlyIncome", "YearsAtCompany", "YearsInCurrentRole"]],
        use_container_width=True
    )

    edu = st.selectbox("Education", sorted(df["Education"].unique()))
    involve = st.selectbox("Job Involvement", sorted(df["JobInvolvement"].unique()))
    level = st.selectbox("Job Level", sorted(df["JobLevel"].unique()))
    perf_income = st.number_input("Monthly Income", 1000, 60000)
    yc = st.number_input("Years At Company", 0, 40)
    yr = st.number_input("Years In Current Role", 0, 20)

    if st.button("Predict Performance Rating"):
        X_perf = pd.DataFrame([{
            "Education": edu,
            "JobInvolvement": involve,
            "JobLevel": level,
            "MonthlyIncome": perf_income,
            "YearsAtCompany": yc,
            "YearsInCurrentRole": yr
        }])

        X_scaled = performance_scaler.transform(X_perf)
        pred = performance_model.predict(X_scaled)[0]

        st.success(f"⭐ Predicted Performance Rating = {pred}")

# =====================================================
# PROMOTION PREDICTION
# =====================================================
elif menu == "🎓 Promotion Prediction":

    st.title("🎓 Promotion Likelihood Prediction")

    st.dataframe(
        df[["PerformanceRating", "YearsAtCompany",
            "TrainingTimesLastYear", "JobInvolvement", "JobLevel"]], 
        use_container_width=True
    )

    pr = st.selectbox("Performance Rating", sorted(df["PerformanceRating"].unique()))
    p_years = st.number_input("Years At Company", 0, 40)
    train = st.number_input("Training Times Last Year", 0, 10)
    jinvolve = st.selectbox("Job Involvement", sorted(df["JobInvolvement"].unique()))
    jlevel = st.selectbox("Job Level", sorted(df["JobLevel"].unique()))

    if st.button("Predict Promotion"):
        X_promo = pd.DataFrame([{
            "PerformanceRating": pr,
            "YearsAtCompany": p_years,
            "TrainingTimesLastYear": train, 
            "JobInvolvement": jinvolve,
            "JobLevel": jlevel 
        }])

        X_scaled = promotion_scaler.transform(X_promo)
        pred = promotion_model.predict(X_scaled)[0]

        if pred == 1:
            st.success("🎉 Promotion is LIKELY!")
        else:
            st.error("❌ Promotion is NOT likely.")