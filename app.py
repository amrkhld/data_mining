import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import itertools

# Page config
st.set_page_config(page_title="Alzheimer's Disease Analysis", layout="wide")

# Title
st.title("🧠 Alzheimer's Disease Prediction & Analysis")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to", ["Data Overview", "Data Cleaning", "Visualizations", "Model Training"]
)


# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("alzheimers_modified_data.csv")
    return df


try:
    df = load_data()

    # Define numeric columns
    numeric_cols = [
        "Age",
        "BMI",
        "Smoking",
        "AlcoholConsumption",
        "PhysicalActivity",
        "DietQuality",
        "SleepQuality",
        "FamilyHistoryAlzheimers",
        "CardiovascularDisease",
        "Diabetes",
        "Depression",
        "HeadInjury",
        "Hypertension",
        "SystolicBP",
        "DiastolicBP",
        "CholesterolTotal",
        "CholesterolLDL",
        "CholesterolHDL",
        "CholesterolTriglycerides",
        "MMSE",
        "FunctionalAssessment",
        "MemoryComplaints",
        "BehavioralProblems",
        "ADL",
        "Confusion",
        "Disorientation",
        "PersonalityChanges",
        "DifficultyCompletingTasks",
        "Forgetfulness",
    ]

    # Data Overview Page
    if page == "Data Overview":
        st.header("📊 Data Overview")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())

        st.subheader("Dataset Preview")
        st.dataframe(df.head(10))

        st.subheader("Dataset Info")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Data Types**")
            st.write(df.dtypes)

        with col2:
            st.write("**Missing Values**")
            missing = df.isnull().sum().sort_values(ascending=False)
            st.write(missing[missing > 0])

        st.subheader("Statistical Summary")
        st.dataframe(df.describe())

    # Data Cleaning Page
    elif page == "Data Cleaning":
        st.header("🧹 Data Cleaning")

        if st.button("Apply Data Cleaning"):
            with st.spinner("Cleaning data..."):
                # Convert to numeric
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                # Impute missing values
                imputer = SimpleImputer(strategy="mean")
                num_cols = df.select_dtypes(include=["float64", "int64"]).columns
                if "Diagnosis" in num_cols:
                    num_cols = num_cols.drop("Diagnosis")
                df[num_cols] = imputer.fit_transform(df[num_cols])

                # Drop unnecessary columns
                df.drop(
                    ["PatientID", "DoctorInCharge", "EducationLevel"],
                    axis=1,
                    inplace=True,
                    errors="ignore",
                )

                # Convert binary columns
                binary_cols = [
                    "Gender",
                    "Smoking",
                    "FamilyHistoryAlzheimers",
                    "CardiovascularDisease",
                    "Diabetes",
                    "Depression",
                    "HeadInjury",
                    "Hypertension",
                    "MemoryComplaints",
                    "BehavioralProblems",
                    "Confusion",
                    "Disorientation",
                    "PersonalityChanges",
                    "DifficultyCompletingTasks",
                    "Forgetfulness",
                    "Diagnosis",
                ]

                for col in binary_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                        df[col] = df[col].apply(lambda x: 1 if x >= 1 else 0)

                # Handle outliers (clip at 5th and 95th percentile)
                for col in numeric_cols:
                    if col in df.columns:
                        lower = df[col].quantile(0.05)
                        upper = df[col].quantile(0.95)
                        df[col] = df[col].clip(lower, upper)

                # Feature engineering
                df["Alcohol_level"] = pd.cut(
                    df["AlcoholConsumption"],
                    bins=[0, 5, 10, 15, 20],
                    labels=["Low", "Moderate", "High", "Very High"],
                )
                df["BMI_category"] = pd.cut(
                    df["BMI"],
                    bins=[0, 18.5, 24.9, 29.9, 100],
                    labels=["Underweight", "Normal", "Overweight", "Obese"],
                )
                df["PA_level"] = pd.cut(
                    df["PhysicalActivity"],
                    bins=[0, 3, 6, 10],
                    labels=["Low", "Moderate", "High"],
                )

                st.success("✅ Data cleaning completed!")
                st.write(f"**Shape after cleaning:** {df.shape}")
                st.write(f"**Missing values:** {df.isnull().sum().sum()}")

    # Visualizations Page
    elif page == "Visualizations":
        st.header("📈 Data Visualizations")

        viz_type = st.selectbox(
            "Select Visualization",
            [
                "Diagnosis Distribution",
                "Age Distribution",
                "MMSE Box Plot",
                "Correlation Heatmap",
                "Smoking vs Diagnosis",
                "Age vs MMSE Scatter",
                "Group Analysis",
            ],
        )

        if viz_type == "Diagnosis Distribution":
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x="Diagnosis", data=df, ax=ax)
            ax.set_title("Distribution of Alzheimer Diagnosis")
            st.pyplot(fig)

        elif viz_type == "Age Distribution":
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(data=df, x="Age", hue="Diagnosis", bins=30, kde=True, ax=ax)
            ax.set_title("Age Distribution by Diagnosis")
            st.pyplot(fig)

        elif viz_type == "MMSE Box Plot":
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x="Diagnosis", y="MMSE", data=df, ax=ax)
            ax.set_title("MMSE Score vs Diagnosis")
            st.pyplot(fig)

        elif viz_type == "Correlation Heatmap":
            fig, ax = plt.subplots(figsize=(16, 12))
            corr = df.select_dtypes(include=[np.number]).corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            ax.set_title("Feature Correlation Heatmap")
            st.pyplot(fig)

        elif viz_type == "Smoking vs Diagnosis":
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x="Diagnosis", y="Smoking", data=df, palette="pastel", ax=ax)
            ax.set_title("Smoking by Diagnosis")
            st.pyplot(fig)

        elif viz_type == "Age vs MMSE Scatter":
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(
                x="Age", y="MMSE", hue="Diagnosis", data=df, palette="coolwarm", ax=ax
            )
            ax.set_title("Age vs MMSE by Diagnosis")
            st.pyplot(fig)

        elif viz_type == "Group Analysis":
            st.subheader("Diagnosis Rate by Groups")

            col1, col2 = st.columns(2)
            with col1:
                st.write("**By Gender:**")
                st.write(df.groupby("Gender")["Diagnosis"].mean() * 100)

                st.write("**By Smoking:**")
                st.write(df.groupby("Smoking")["Diagnosis"].mean() * 100)

            with col2:
                st.write("**By Family History:**")
                st.write(
                    df.groupby("FamilyHistoryAlzheimers")["Diagnosis"].mean() * 100
                )

    # Model Training Page
    elif page == "Model Training":
        st.header("🤖 Model Training & Evaluation")

        model_type = st.selectbox(
            "Select Model",
            [
                "K-Nearest Neighbors",
                "Logistic Regression",
                "Random Forest",
                "Support Vector Machine",
            ],
        )

        if st.button("Train Model"):
            with st.spinner(f"Training {model_type}..."):
                # Prepare data
                df_model = df.copy()

                # Encode categorical variables
                cat_cols = ["BMI_category", "Alcohol_level", "PA_level"]
                le = LabelEncoder()
                for col in cat_cols:
                    if col in df_model.columns:
                        df_model[col] = le.fit_transform(df_model[col].astype(str))

                X = df_model.drop("Diagnosis", axis=1)
                y = df_model["Diagnosis"]

                # Impute and scale
                imputer = SimpleImputer(strategy="mean")
                X = imputer.fit_transform(X)

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=y
                )

                # Train model
                if model_type == "K-Nearest Neighbors":
                    model = KNeighborsClassifier(n_neighbors=5)
                elif model_type == "Logistic Regression":
                    model = LogisticRegression()
                elif model_type == "Random Forest":
                    model = RandomForestClassifier(
                        n_estimators=100, random_state=42, n_jobs=-1
                    )
                else:  # SVM
                    model = SVC(kernel="rbf", C=1, gamma="scale", random_state=42)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Results
                st.success("✅ Model trained successfully!")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Performance Metrics")
                    accuracy = accuracy_score(y_test, y_pred)
                    st.metric("Accuracy", f"{accuracy:.4f}")

                    st.text("Classification Report:")
                    st.text(classification_report(y_test, y_pred))

                with col2:
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)

                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=["No Alzheimer", "Alzheimer"],
                        yticklabels=["No Alzheimer", "Alzheimer"],
                        ax=ax,
                    )
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    ax.set_title(f"Confusion Matrix - {model_type}")
                    st.pyplot(fig)

except FileNotFoundError:
    st.error(
        "❌ Error: 'alzheimers_modified_data.csv' not found. Please upload the dataset."
    )
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.rerun()
except Exception as e:
    st.error(f"❌ An error occurred: {str(e)}")
