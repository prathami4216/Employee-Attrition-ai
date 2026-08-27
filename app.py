"""
============================================================
Employee Attrition Prediction System
Professional HR Dashboard
============================================================
Frontend  : Streamlit
Backend   : Decision Tree Classifier
Author    : Prathami Sawant
============================================================
"""

import streamlit as st
import pandas as pd
import joblib
import os

from predict import (
    predict_employee,
    predict_dataset
)

from utils.preprocess import (
    load_dataset
)

from config import *

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👨‍💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------
# Custom CSS
# ----------------------------------------------------------

st.markdown("""

<style>

.main{
    background:#F5F7FA;
}

h1{
    color:#0E4C92;
}

.metric-card{
    background:white;
    padding:20px;
    border-radius:12px;
    box-shadow:0px 3px 10px rgba(0,0,0,.15);
}

</style>

""", unsafe_allow_html=True)

# ----------------------------------------------------------
# Sidebar
# ----------------------------------------------------------

if os.path.exists(LOGO):
    st.sidebar.image(LOGO, width="stretch")
else:
    st.sidebar.info("Logo not found.")

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Select Page",

    [

        "🏠 Home",

        "🔍 Predict Employee",

        "📂 Batch Prediction",

        "📊 Dashboard",

        "🌳 Decision Tree",

        "📈 Model Performance",

        "ℹ About"

    ]

)

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

try:

    dataset = load_dataset()

except:

    dataset = None

# ----------------------------------------------------------
# HOME PAGE
# ----------------------------------------------------------

if page == "🏠 Home":

    st.title("Employee Attrition Prediction System")

    st.write("---")

    col1,col2,col3,col4 = st.columns(4)

    if dataset is not None:

        col1.metric(

            "Employees",

            len(dataset)

        )

        col2.metric(

            "Features",

            len(dataset.columns)-1

        )

        col3.metric(

            "Target",

            "Attrition"

        )

        yes = (dataset["Attrition"]=="Yes").sum()

        col4.metric(

            "Employees Left",

            yes

        )

    st.write("---")

    st.subheader("Project Overview")

    st.write("""

This project predicts whether an employee is likely to leave the organization using a **Decision Tree Classifier**.

### Features

- Employee Attrition Prediction
- Batch Prediction
- Decision Tree Visualization
- Feature Importance
- Confusion Matrix
- Model Performance
- CSV / Excel Support
- Professional HR Dashboard

""")

    if dataset is not None:

        st.subheader("Dataset Preview")

        st.dataframe(dataset.head(10), width="stretch")
        
 # ----------------------------------------------------------
# EMPLOYEE PREDICTION PAGE
# ----------------------------------------------------------

elif page == "🔍 Predict Employee":

    st.title("Employee Attrition Prediction")

    st.write("Enter employee details below.")

    if dataset is None:

        st.error("Dataset not found.")

        st.stop()

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=65,
                value=30
            )

            gender = st.selectbox(
                "Gender",
                sorted(dataset["Gender"].unique())
            )

            marital = st.selectbox(
                "Marital Status",
                sorted(dataset["Marital_Status"].unique())
            )

            department = st.selectbox(
                "Department",
                sorted(dataset["Department"].unique())
            )

            job_role = st.selectbox(
                "Job Role",
                sorted(dataset["Job_Role"].unique())
            )

            job_level = st.slider(
                "Job Level",
                1,
                5,
                2
            )

            monthly_income = st.number_input(
                "Monthly Income",
                min_value=1000,
                max_value=100000,
                value=8000
            )

            hourly_rate = st.number_input(
                "Hourly Rate",
                min_value=1,
                max_value=150,
                value=60
            )

            years_company = st.number_input(
                "Years at Company",
                min_value=0,
                max_value=40,
                value=5
            )

            years_role = st.number_input(
                "Years in Current Role",
                min_value=0,
                max_value=30,
                value=3
            )

            years_promotion = st.number_input(
                "Years Since Last Promotion",
                min_value=0,
                max_value=20,
                value=1
            )

        with col2:

            worklife = st.slider(
                "Work Life Balance",
                1,
                4,
                3
            )

            satisfaction = st.slider(
                "Job Satisfaction",
                1,
                5,
                3
            )

            performance = st.slider(
                "Performance Rating",
                1,
                5,
                3
            )

            training = st.number_input(
                "Training Hours Last Year",
                min_value=0,
                max_value=200,
                value=20
            )

            overtime = st.selectbox(
                "Overtime",
                sorted(dataset["Overtime"].unique())
            )

            project_count = st.number_input(
                "Project Count",
                min_value=1,
                max_value=20,
                value=5
            )

            avg_hours = st.number_input(
                "Average Working Hours / Week",
                min_value=20,
                max_value=80,
                value=45
            )

            absenteeism = st.number_input(
                "Absenteeism",
                min_value=0,
                max_value=40,
                value=3
            )

            environment = st.slider(
                "Work Environment Satisfaction",
                1,
                4,
                3
            )

            manager = st.slider(
                "Relationship with Manager",
                1,
                4,
                3
            )

            involvement = st.slider(
                "Job Involvement",
                1,
                4,
                3
            )

            distance = st.number_input(
                "Distance From Home",
                min_value=0,
                max_value=100,
                value=10
            )

            companies = st.number_input(
                "Number of Companies Worked",
                min_value=0,
                max_value=20,
                value=2
            )

        predict_button = st.form_submit_button(
            "Predict Attrition"
        )

    if predict_button:

        employee = {

            "Age": age,
            "Gender": gender,
            "Marital_Status": marital,
            "Department": department,
            "Job_Role": job_role,
            "Job_Level": job_level,
            "Monthly_Income": monthly_income,
            "Hourly_Rate": hourly_rate,
            "Years_at_Company": years_company,
            "Years_in_Current_Role": years_role,
            "Years_Since_Last_Promotion": years_promotion,
            "Work_Life_Balance": worklife,
            "Job_Satisfaction": satisfaction,
            "Performance_Rating": performance,
            "Training_Hours_Last_Year": training,
            "Overtime": overtime,
            "Project_Count": project_count,
            "Average_Hours_Worked_Per_Week": avg_hours,
            "Absenteeism": absenteeism,
            "Work_Environment_Satisfaction": environment,
            "Relationship_with_Manager": manager,
            "Job_Involvement": involvement,
            "Distance_From_Home": distance,
            "Number_of_Companies_Worked": companies

        }

        with st.spinner("Predicting..."):

            result = predict_employee(employee)

        st.success("Prediction Completed")

        st.write("---")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Prediction",
            result["prediction"]
        )

        c2.metric(
            "Confidence",
            f"{result['confidence']} %"
        )

        c3.metric(
            "Risk Level",
            f"{result['risk_percentage']} %"
        )

        
        if result["prediction"].lower() == "yes":

            st.error(
                "⚠ Employee is likely to leave the organization."
            )

        else:

            st.success(
                "✅ Employee is likely to stay in the organization."
            )

        st.subheader("Attrition Risk")

        import plotly.graph_objects as go

        gauge_color = (
            "#D64545" if result["confidence"] >= 80 and result["prediction"].lower() == "yes"
            else "#E8A33D" if result["confidence"] >= 60
            else "#3AA655"
        )

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["confidence"],
            number={"suffix": "%"},
            title={"text": f"Confidence — {result['prediction']}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": gauge_color},
                "steps": [
                    {"range": [0, 60], "color": "#EAF4E8"},
                    {"range": [60, 80], "color": "#FCF1DC"},
                    {"range": [80, 100], "color": "#FBE4E4"}
                ]
            }
        ))

        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))

        st.plotly_chart(fig, width="stretch")

        st.subheader("Prediction Probability")

        probability = pd.DataFrame({

            "Class":["No","Yes"],

            "Probability": [round(p*100,2) for p in result["probability"]]

        })

        fig2 = go.Figure(go.Bar(
            x=probability["Class"],
            y=probability["Probability"],
            marker_color=["#3AA655", "#D64545"],
            text=probability["Probability"].astype(str) + "%",
            textposition="outside"
        ))

        fig2.update_layout(
            yaxis_title="Probability (%)",
            height=350,
            margin=dict(l=20, r=20, t=30, b=20)
        )

        st.plotly_chart(fig2, width="stretch")    
# ----------------------------------------------------------
# BATCH PREDICTION
# ----------------------------------------------------------

elif page == "📂 Batch Prediction":

    st.title("Batch Employee Attrition Prediction")

    st.write(
        """
Upload a CSV or Excel file containing employee information.
The system will predict attrition for every employee.
"""
    )

    uploaded_file = st.file_uploader(

        "Choose CSV or Excel File",

        type=["csv", "xlsx"]

    )

    if uploaded_file is not None:

        extension = uploaded_file.name.split(".")[-1].lower()

        if extension == "csv":

            dataframe = pd.read_csv(uploaded_file)

        else:

            dataframe = pd.read_excel(uploaded_file)

        st.subheader("Dataset Preview")

        st.dataframe(

            dataframe.head(),

            width="stretch"

        )

        st.write(f"Total Employees : {len(dataframe)}")

        if st.button("Predict Entire Dataset"):

            with st.spinner("Predicting..."):

                try:

                    if extension == "csv":

                        dataframe.to_csv(
                            "temp_prediction.csv",
                            index=False
                        )

                        results = predict_dataset(
                            "temp_prediction.csv"
                        )

                    else:

                        dataframe.to_excel(
                            "temp_prediction.xlsx",
                            index=False
                        )

                        results = predict_dataset(
                            "temp_prediction.xlsx"
                        )

                    st.success(
                        "Prediction Completed Successfully."
                    )

                    st.subheader("Prediction Results")

                    st.dataframe(

                        results,

                        width="stretch"

                    )

                    # ----------------------------
                    # Summary
                    # ----------------------------

                    st.subheader("Prediction Summary")

                    col1, col2, col3 = st.columns(3)

                    total = len(results)

                    leave = (
                        results["Prediction"]
                        .astype(str)
                        .str.lower()
                        .eq("yes")
                        .sum()
                    )

                    stay = total - leave

                    col1.metric(

                        "Total Employees",

                        total

                    )

                    col2.metric(

                        "Likely To Leave",

                        leave

                    )

                    col3.metric(

                        "Likely To Stay",

                        stay

                    )

                    # ----------------------------
                    # Distribution + Risk Breakdown
                    # ----------------------------

                    import plotly.express as px

                    st.subheader("Prediction Distribution")

                    dist_col1, dist_col2 = st.columns(2)

                    with dist_col1:

                        counts = results["Prediction"].value_counts().reset_index()
                        counts.columns = ["Prediction", "Count"]

                        pie = px.pie(
                            counts,
                            names="Prediction",
                            values="Count",
                            color="Prediction",
                            color_discrete_map={"Yes": "#D64545", "No": "#3AA655"},
                            hole=0.45
                        )

                        pie.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))

                        st.plotly_chart(pie, width="stretch")

                    with dist_col2:

                        results["Risk_Band"] = pd.cut(
                            results["Confidence (%)"],
                            bins=[0, 60, 80, 100],
                            labels=["Low", "Medium", "High"]
                        )

                        risk_counts = results["Risk_Band"].value_counts().reindex(
                            ["Low", "Medium", "High"]
                        ).reset_index()

                        risk_counts.columns = ["Risk", "Count"]

                        bar = px.bar(
                            risk_counts,
                            x="Risk",
                            y="Count",
                            color="Risk",
                            color_discrete_map={
                                "Low": "#3AA655",
                                "Medium": "#E8A33D",
                                "High": "#D64545"
                            }
                        )

                        bar.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))

                        st.plotly_chart(bar, width="stretch")

                    # ----------------------------
                    # Download CSV
                    # ----------------------------

                    csv = results.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(

                        "Download Prediction Report",

                        csv,

                        file_name="employee_predictions.csv",

                        mime="text/csv"

                    )

                except Exception as e:

                    st.error(str(e))   
# ----------------------------------------------------------
# HR DASHBOARD
# ----------------------------------------------------------

elif page == "📊 Dashboard":

    st.title("📊 HR Analytics Dashboard")

    if dataset is None:

        st.error("Dataset not found.")
        st.stop()

    st.write("---")

    # =====================================================
    # KPI CARDS
    # =====================================================

    total_employees = len(dataset)

    attrition_yes = len(
        dataset[dataset["Attrition"] == "Yes"]
    )

    attrition_no = len(
        dataset[dataset["Attrition"] == "No"]
    )

    attrition_rate = round(
        attrition_yes / total_employees * 100,
        2
    )

    avg_income = int(
        dataset["Monthly_Income"].mean()
    )

    avg_age = round(
        dataset["Age"].mean(),
        1
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Employees",
        total_employees
    )

    c2.metric(
        "Left",
        attrition_yes
    )

    c3.metric(
        "Stayed",
        attrition_no
    )

    c4.metric(
        "Attrition %",
        f"{attrition_rate}%"
    )

    c5.metric(
        "Average Salary",
        f"₹ {avg_income:,}"
    )

    st.write("---")

    # =====================================================
    # SECOND ROW
    # =====================================================

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("Attrition Distribution")

        attrition_chart = dataset[
            "Attrition"
        ].value_counts()

        st.bar_chart(attrition_chart)

    with c2:

        st.subheader("Department Wise Employees")

        department_chart = dataset[
            "Department"
        ].value_counts()

        st.bar_chart(department_chart)

    st.write("---")

    # =====================================================
    # THIRD ROW
    # =====================================================

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("Job Role Distribution")

        role_chart = dataset[
            "Job_Role"
        ].value_counts()

        st.bar_chart(role_chart)

    with c2:

        st.subheader("Gender Distribution")

        gender_chart = dataset[
            "Gender"
        ].value_counts()

        st.bar_chart(gender_chart)

    st.write("---")

    # =====================================================
    # FOURTH ROW
    # =====================================================

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("Monthly Income")

        st.line_chart(
            dataset["Monthly_Income"]
        )

    with c2:

        st.subheader("Age Distribution")

        st.line_chart(
            dataset["Age"]
        )

    st.write("---")

    # =====================================================
    # OVERTIME ANALYSIS
    # =====================================================

    st.subheader("Overtime Analysis")

    overtime_chart = (
        dataset
        .groupby("Overtime")["Attrition"]
        .value_counts()
        .unstack(fill_value=0)
    )

    st.bar_chart(overtime_chart)

    st.write("---")

    # =====================================================
    # SATISFACTION
    # =====================================================

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("Job Satisfaction")

        satisfaction = (
            dataset["Job_Satisfaction"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(satisfaction)

    with c2:

        st.subheader("Work Life Balance")

        balance = (
            dataset["Work_Life_Balance"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(balance)

    st.write("---")

    # =====================================================
    # PERFORMANCE
    # =====================================================

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("Performance Rating")

        performance = (
            dataset["Performance_Rating"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(performance)

    with c2:

        st.subheader("Project Count")

        projects = (
            dataset["Project_Count"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(projects)

    st.write("---")

    # =====================================================
    # DEPARTMENT SUMMARY
    # =====================================================

    st.subheader("Department Summary")

    summary = dataset.groupby(

        "Department"

    ).agg(

        Employees=("Department", "count"),

        Average_Salary=("Monthly_Income", "mean"),

        Average_Age=("Age", "mean")

    )

    st.dataframe(

        summary.style.format({

            "Average_Salary": "₹{:,.0f}",

            "Average_Age": "{:.1f}"

        }),

        width="stretch"

    )

    st.write("---")

    # =====================================================
    # RAW DATA
    # =====================================================

    with st.expander("View Complete Dataset"):

        st.dataframe(

            dataset,

            width="stretch",

            height=450

        )
# ----------------------------------------------------------
# MODEL PERFORMANCE
# ----------------------------------------------------------

elif page == "📈 Model Performance":

    st.title("📈 Decision Tree Model Performance")

    import json
    import os

    metrics_file = "models/metrics.json"

    if not os.path.exists(metrics_file):

        st.warning(
            "Train the model first to generate performance metrics."
        )

        st.stop()

    with open(metrics_file, "r") as f:

        metrics = json.load(f)

    st.subheader("Overall Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Accuracy",
        f"{metrics['accuracy']:.2%}"
    )

    c2.metric(
        "Precision",
        f"{metrics['precision']:.2%}"
    )

    c3.metric(
        "Recall",
        f"{metrics['recall']:.2%}"
    )

    c4.metric(
        "F1 Score",
        f"{metrics['f1_score']:.2%}"
    )

    st.write("---")

    # ==========================================
    # Classification Report
    # ==========================================

    st.subheader("Classification Report")

    report = metrics.get("classification_report", "")

    st.code(report)

    st.write("---")

    # ==========================================
    # Confusion Matrix
    # ==========================================

    cm_path = CONFUSION_MATRIX

    if os.path.exists(cm_path):

        st.subheader("Confusion Matrix")

        st.image(
            cm_path,
            width="stretch"
        )

    st.write("---")

    # ==========================================
    # Feature Importance
    # ==========================================

    fi_path = FEATURE_IMPORTANCE

    if os.path.exists(fi_path):

        st.subheader("Feature Importance")

        st.image(
            fi_path,
            width="stretch"
        )

    st.write("---")

    # ==========================================
    # Decision Tree
    # ==========================================

    tree_path = DECISION_TREE

    if os.path.exists(tree_path):

        st.subheader("Decision Tree")

        st.image(
            tree_path,
            width="stretch"
        )

    st.write("---")

    # ==========================================
    # Model Information
    # ==========================================

    st.subheader("Model Details")

    info = {

        "Algorithm":
            "Decision Tree",

        "Criterion":
            "Entropy",

        "Maximum Depth":
            MAX_DEPTH,

        "Minimum Samples Split":
            MIN_SAMPLES_SPLIT,

        "Minimum Samples Leaf":
            MIN_SAMPLES_LEAF,

        "Train/Test Split":
            f"{int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}",

        "Random State":
            RANDOM_STATE

    }

    info_display = {k: str(v) for k, v in info.items()}

    st.table(

        pd.DataFrame(

            info_display.items(),

            columns=["Parameter", "Value"]

        )

    )

    st.write("---")

    # ==========================================
    # Feature List
    # ==========================================

    st.subheader("Input Features Used")

    features = [

        "Age",

        "Gender",

        "Marital_Status",

        "Department",

        "Job_Role",

        "Job_Level",

        "Monthly_Income",

        "Hourly_Rate",

        "Years_at_Company",

        "Years_in_Current_Role",

        "Years_Since_Last_Promotion",

        "Work_Life_Balance",

        "Job_Satisfaction",

        "Performance_Rating",

        "Training_Hours_Last_Year",

        "Overtime",

        "Project_Count",

        "Average_Hours_Worked_Per_Week",

        "Absenteeism",

        "Work_Environment_Satisfaction",

        "Relationship_with_Manager",

        "Job_Involvement",

        "Distance_From_Home",

        "Number_of_Companies_Worked"

    ]

    st.dataframe(

        pd.DataFrame(

            {"Features": features}

        ),

        width="stretch"
    )

# ----------------------------------------------------------
# ABOUT PROJECT
# ----------------------------------------------------------

elif page == "ℹ️ About":

    st.title("ℹ️ Employee Attrition Prediction System")

    st.markdown("---")

    st.header("Project Overview")

    st.write("""
The Employee Attrition Prediction System is a Machine Learning based HR Analytics
application that predicts whether an employee is likely to leave the organization.

The application helps HR departments identify high-risk employees early,
allowing organizations to improve employee retention and reduce turnover costs.

The system has been developed using Python, Streamlit, Scikit-Learn,
Decision Tree Classification, and Data Analytics techniques.
""")

    st.markdown("---")

    st.header("Objectives")

    st.markdown("""

✔ Predict Employee Attrition

✔ Identify High Risk Employees

✔ Improve Employee Retention

✔ Assist HR Decision Making

✔ Analyze Workforce Trends

✔ Visualize Employee Statistics

✔ Explain Decision Tree Predictions

""")

    st.markdown("---")

    st.header("Machine Learning Workflow")

    workflow = """

Employee Dataset

↓

Data Cleaning

↓

Feature Engineering

↓

Encoding

↓

Train Test Split

↓

Decision Tree Training

↓

Model Evaluation

↓

Prediction

↓

Visualization

↓

Deployment

"""

    st.code(workflow)

    st.markdown("---")

    st.header("Technology Stack")

    tech = {

        "Frontend":"Streamlit",

        "Backend":"Python",

        "Machine Learning":"Scikit-Learn",

        "Algorithm":"Decision Tree",

        "Dataset":"IBM HR Analytics",

        "Visualization":"Matplotlib",

        "Data Processing":"Pandas & NumPy",

        "Model Saving":"Joblib"

    }

    tech_display = {k: str(v) for k, v in tech.items()}

    st.table(

        pd.DataFrame(

            tech_display.items(),

            columns=["Technology","Used"]

        )

    )

    st.markdown("---")

    st.header("Dataset Features")

    features = [

        "Age",

        "Gender",

        "Marital Status",

        "Department",

        "Job Role",

        "Job Level",

        "Monthly Income",

        "Hourly Rate",

        "Years at Company",

        "Years in Current Role",

        "Years Since Last Promotion",

        "Work Life Balance",

        "Job Satisfaction",

        "Performance Rating",

        "Training Hours",

        "Overtime",

        "Project Count",

        "Average Hours Worked",

        "Absenteeism",

        "Work Environment Satisfaction",

        "Relationship with Manager",

        "Job Involvement",

        "Distance From Home",

        "Number of Companies Worked"

    ]

    st.dataframe(

        pd.DataFrame(

            {"Features":features}

        ),

        width="stretch"

    )

    st.markdown("---")

    st.header("Model Evaluation Metrics")

    metrics = [

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score",

        "Confusion Matrix",

        "Feature Importance",

        "Decision Tree Visualization"

    ]

    st.table(

        pd.DataFrame(

            {"Metric":metrics}

        )

    )

    st.markdown("---")

    st.header("Project Features")

    st.markdown("""

✔ Single Employee Prediction

✔ Batch Prediction

✔ HR Dashboard

✔ Decision Tree Visualization

✔ Feature Importance

✔ Confusion Matrix

✔ Model Performance Report

✔ Download Prediction Report

✔ CSV Support

✔ Excel Support

✔ Responsive Streamlit UI

✔ Enterprise Level Structure

""")

    st.markdown("---")

    st.header("Future Enhancements")

    st.markdown("""

• Employee Login

• HR Admin Dashboard

• Email Alerts

• PDF Report Generation

• Explainable AI (SHAP)

• Live Database Integration

• Multi Model Comparison

• Salary Trend Analysis

• Department Risk Analysis

• Cloud Deployment

""")

    st.markdown("---")

    st.header("Developer")

    st.info("""

Employee Attrition Prediction System

Machine Learning Project

Algorithm : Decision Tree Classifier

Framework : Streamlit

Language : Python

""")

    st.success("Project Loaded Successfully.")            
