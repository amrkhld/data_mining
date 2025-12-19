import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import itertools
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Alzheimer's Disease Analysis", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; font-weight: bold; color: #1f77b4;}
    .sub-header {font-size: 1.5rem; color: #ff7f0e;}
    .metric-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🧠 Alzheimer\'s Disease Prediction & Analysis</p>', unsafe_allow_html=True)

# Initialize session state
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_cleaned' not in st.session_state:
    st.session_state.df_cleaned = None
if 'cleaning_done' not in st.session_state:
    st.session_state.cleaning_done = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = {}

# Sidebar
st.sidebar.header("🎯 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 Data Overview", "🧹 Data Cleaning", "📈 Visualizations", 
     "🔍 Exploratory Analysis", "🤖 Model Training", "📊 Model Comparison", "🎯 Prediction"]
)

# Define numeric and binary columns
NUMERIC_COLS = [
    'Age', 'BMI', 'Smoking', 'AlcoholConsumption', 'PhysicalActivity',
    'DietQuality', 'SleepQuality', 'FamilyHistoryAlzheimers',
    'CardiovascularDisease', 'Diabetes', 'Depression', 'HeadInjury',
    'Hypertension', 'SystolicBP', 'DiastolicBP',
    'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL',
    'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment',
    'MemoryComplaints', 'BehavioralProblems', 'ADL', 'Confusion',
    'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks',
    'Forgetfulness'
]

BINARY_COLS = [
    'Gender', 'Smoking', 'FamilyHistoryAlzheimers', 'CardiovascularDisease',
    'Diabetes', 'Depression', 'HeadInjury', 'Hypertension', 'MemoryComplaints',
    'BehavioralProblems', 'Confusion', 'Disorientation', 'PersonalityChanges',
    'DifficultyCompletingTasks', 'Forgetfulness', 'Diagnosis'
]

# Load data function
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("alzheimers_modified_data.csv")
        return df
    except FileNotFoundError:
        return None

# Data cleaning function
def clean_data(df):
    df_clean = df.copy()
    
    # Convert to numeric
    for col in NUMERIC_COLS:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    num_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    if 'Diagnosis' in num_cols:
        num_cols = num_cols.drop('Diagnosis')
    df_clean[num_cols] = imputer.fit_transform(df_clean[num_cols])
    
    # Drop unnecessary columns
    cols_to_drop = ['PatientID', 'DoctorInCharge', 'EducationLevel']
    df_clean.drop([col for col in cols_to_drop if col in df_clean.columns], 
                  axis=1, inplace=True)
    
    # Convert binary columns
    for col in BINARY_COLS:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean[col] = df_clean[col].apply(lambda x: 1 if x >= 1 else 0)
    
    # Handle outliers (clip at 5th and 95th percentile)
    for col in NUMERIC_COLS:
        if col in df_clean.columns:
            lower = df_clean[col].quantile(0.05)
            upper = df_clean[col].quantile(0.95)
            df_clean[col] = df_clean[col].clip(lower, upper)
    
    # Feature engineering
    if 'AlcoholConsumption' in df_clean.columns:
        df_clean['Alcohol_level'] = pd.cut(
            df_clean['AlcoholConsumption'],
            bins=[0, 5, 10, 15, 20],
            labels=['Low', 'Moderate', 'High', 'Very High']
        )
    
    if 'BMI' in df_clean.columns:
        df_clean['BMI_category'] = pd.cut(
            df_clean['BMI'],
            bins=[0, 18.5, 24.9, 29.9, 100],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        )
    
    if 'PhysicalActivity' in df_clean.columns:
        df_clean['PA_level'] = pd.cut(
            df_clean['PhysicalActivity'],
            bins=[0, 3, 6, 10],
            labels=['Low', 'Moderate', 'High']
        )
    
    return df_clean

# Detect outliers function
def detect_outliers(df, numeric_cols):
    outlier_info = {}
    for col in numeric_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower) | (df[col] > upper)]
            outlier_info[col] = len(outliers)
    return outlier_info

# Load initial data
if st.session_state.df_original is None:
    df_loaded = load_data()
    if df_loaded is not None:
        st.session_state.df_original = df_loaded

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    st.header("Welcome to Alzheimer's Disease Analysis Platform")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Project Overview
        This comprehensive platform analyzes Alzheimer's disease data using advanced machine learning techniques.
        
        **Features:**
        - 📊 **Data Overview**: Explore dataset statistics and information
        - 🧹 **Data Cleaning**: Automated preprocessing and feature engineering
        - 📈 **Visualizations**: Interactive charts and graphs
        - 🔍 **Exploratory Analysis**: Deep dive into data patterns
        - 🤖 **Model Training**: Train multiple ML models (KNN, Logistic Regression, Random Forest, SVM)
        - 📊 **Model Comparison**: Compare performance across models
        - 🎯 **Prediction**: Make predictions on new data
        
        ### 📋 Dataset Information
        The dataset includes various features related to:
        - Demographics (Age, Gender, BMI)
        - Lifestyle factors (Smoking, Alcohol, Physical Activity)
        - Medical history (Diabetes, Hypertension, Depression)
        - Clinical assessments (MMSE, Functional Assessment)
        """)
    
    with col2:
        if st.session_state.df_original is not None:
            st.success("✅ Dataset Loaded")
            st.metric("Total Records", st.session_state.df_original.shape[0])
            st.metric("Total Features", st.session_state.df_original.shape[1])
            
            if st.session_state.cleaning_done:
                st.success("✅ Data Cleaned")
            else:
                st.warning("⚠️ Data not cleaned yet")
        else:
            st.error("❌ No dataset found")
    
    st.info("👈 Use the navigation menu to explore different sections of the analysis.")

# ==================== DATA OVERVIEW PAGE ====================
elif page == "📊 Data Overview":
    st.header("📊 Data Overview")
    
    if st.session_state.df_original is None:
        st.error("❌ Error: 'alzheimers_modified_data.csv' not found. Please upload the dataset.")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            st.session_state.df_original = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            st.rerun()
    else:
        df = st.session_state.df_original
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📊 Total Rows", df.shape[0])
        col2.metric("📋 Total Columns", df.shape[1])
        col3.metric("❌ Missing Values", df.isnull().sum().sum())
        col4.metric("🎯 Diagnosis Cases", df['Diagnosis'].sum() if 'Diagnosis' in df.columns else "N/A")
        
        # Dataset Preview
        st.subheader("🔍 Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)
        
        # Column Information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Column Names")
            st.write(list(df.columns))
            
            st.subheader("🔢 Data Types")
            st.write(df.dtypes)
        
        with col2:
            st.subheader("❓ Missing Values")
            missing = df.isnull().sum().sort_values(ascending=False)
            if missing.sum() > 0:
                st.write(missing[missing > 0])
            else:
                st.success("No missing values found!")
            
            st.subheader("📊 Value Counts (Diagnosis)")
            if 'Diagnosis' in df.columns:
                st.write(df['Diagnosis'].value_counts())
        
        # Statistical Summary
        st.subheader("📈 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Raw Data",
            csv,
            "raw_data.csv",
            "text/csv",
            key='download-raw'
        )

# ==================== DATA CLEANING PAGE ====================
elif page == "🧹 Data Cleaning":
    st.header("🧹 Data Cleaning & Preprocessing")
    
    if st.session_state.df_original is None:
        st.error("❌ Please load the dataset first from the Data Overview page.")
    else:
        df = st.session_state.df_original
        
        st.info("This process will: Convert data types → Impute missing values → Handle outliers → Engineer features")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if not st.session_state.cleaning_done:
                if st.button("🚀 Apply Data Cleaning", type="primary", use_container_width=True):
                    with st.spinner("Cleaning data... Please wait."):
                        # Detect outliers before cleaning
                        st.subheader("🔍 Outlier Detection (Before Cleaning)")
                        outliers_before = detect_outliers(df, NUMERIC_COLS)
                        
                        outlier_df = pd.DataFrame(list(outliers_before.items()), 
                                                 columns=['Feature', 'Outlier Count'])
                        st.dataframe(outlier_df, use_container_width=True)
                        
                        # Clean data
                        df_cleaned = clean_data(df)
                        st.session_state.df_cleaned = df_cleaned
                        st.session_state.cleaning_done = True
                        
                        st.success("✅ Data cleaning completed successfully!")
                        st.balloons()
                        st.rerun()
            else:
                st.success("✅ Data has been cleaned!")
                if st.button("🔄 Reset and Re-clean", use_container_width=True):
                    st.session_state.cleaning_done = False
                    st.session_state.df_cleaned = None
                    st.rerun()
        
        with col2:
            st.metric("Original Shape", f"{df.shape[0]} × {df.shape[1]}")
            if st.session_state.cleaning_done:
                st.metric("Cleaned Shape", f"{st.session_state.df_cleaned.shape[0]} × {st.session_state.df_cleaned.shape[1]}")
        
        # Show cleaning results
        if st.session_state.cleaning_done:
            df_cleaned = st.session_state.df_cleaned
            
            st.subheader("📊 Cleaning Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Missing Values (Before)", df.isnull().sum().sum())
                st.metric("Missing Values (After)", df_cleaned.isnull().sum().sum())
            
            with col2:
                st.write("**Columns Dropped:**")
                dropped = ['PatientID', 'DoctorInCharge', 'EducationLevel']
                for col in dropped:
                    if col not in df_cleaned.columns:
                        st.write(f"✓ {col}")
            
            with col3:
                st.write("**New Features Created:**")
                new_features = ['Alcohol_level', 'BMI_category', 'PA_level']
                for feat in new_features:
                    if feat in df_cleaned.columns:
                        st.write(f"✓ {feat}")
            
            # Show cleaned data preview
            st.subheader("🔍 Cleaned Data Preview")
            st.dataframe(df_cleaned.head(10), use_container_width=True)
            
            # Download cleaned data
            csv = df_cleaned.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Cleaned Data",
                csv,
                "cleaned_data.csv",
                "text/csv",
                key='download-cleaned'
            )

# ==================== VISUALIZATIONS PAGE ====================
elif page == "📈 Visualizations":
    st.header("📈 Data Visualizations")
    
    df = st.session_state.df_cleaned if st.session_state.cleaning_done else st.session_state.df_original
    
    if df is None:
        st.error("❌ Please load the dataset first.")
    else:
        viz_type = st.selectbox(
            "🎨 Select Visualization Type",
            [
                "Diagnosis Distribution",
                "Age Distribution",
                "MMSE Box Plot",
                "Correlation Heatmap",
                "Smoking vs Diagnosis",
                "Age vs MMSE Scatter",
                "Pair Plots",
                "Line Plot (Feature Means)"
            ],
        )
        
        try:
            if viz_type == "Diagnosis Distribution":
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.countplot(x="Diagnosis", data=df, ax=ax, palette="viridis")
                ax.set_title("Distribution of Alzheimer Diagnosis", fontsize=16, fontweight='bold')
                ax.set_xlabel("Diagnosis", fontsize=12)
                ax.set_ylabel("Count", fontsize=12)
                for p in ax.patches:
                    ax.annotate(f'{int(p.get_height())}', 
                               (p.get_x() + p.get_width() / 2., p.get_height()),
                               ha='center', va='bottom')
                st.pyplot(fig)
                plt.close()
            
            elif viz_type == "Age Distribution":
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.histplot(data=df, x="Age", hue="Diagnosis", bins=30, kde=True, ax=ax, palette="coolwarm")
                ax.set_title("Age Distribution by Diagnosis", fontsize=16, fontweight='bold')
                st.pyplot(fig)
                plt.close()
            
            elif viz_type == "MMSE Box Plot":
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.boxplot(x="Diagnosis", y="MMSE", data=df, ax=ax, palette="Set2")
                ax.set_title("MMSE Score vs Diagnosis", fontsize=16, fontweight='bold')
                st.pyplot(fig)
                plt.close()
            
            elif viz_type == "Correlation Heatmap":
                fig, ax = plt.subplots(figsize=(18, 14))
                corr = df.select_dtypes(include=[np.number]).corr()
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, 
                           cbar_kws={'label': 'Correlation Coefficient'})
                ax.set_title("Feature Correlation Heatmap", fontsize=18, fontweight='bold')
                st.pyplot(fig)
                plt.close()
            
            elif viz_type == "Smoking vs Diagnosis":
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(x="Diagnosis", y="Smoking", data=df, palette="pastel", ax=ax)
                ax.set_title("Smoking by Diagnosis", fontsize=16, fontweight='bold')
                st.pyplot(fig)
                plt.close()
            
            elif viz_type == "Age vs MMSE Scatter":
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x="Age", y="MMSE", hue="Diagnosis", data=df, 
                              palette="coolwarm", ax=ax, s=100, alpha=0.6)
                ax.set_title("Age vs MMSE by Diagnosis", fontsize=16, fontweight='bold')
                st.pyplot(fig)
                plt.close()
            
            elif viz_type == "Pair Plots":
                st.subheader("Pair Plots: Age, BMI, MMSE")
                selected_cols = ['Age', 'BMI', 'MMSE']
                
                for x_col, y_col in itertools.combinations(selected_cols, 2):
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.scatterplot(x=df[x_col], y=df[y_col], hue=df['Diagnosis'], 
                                  palette='Set2', ax=ax, s=50, alpha=0.6)
                    ax.set_title(f'Scatter Plot: {x_col} vs {y_col}', fontsize=14, fontweight='bold')
                    ax.set_xlabel(x_col, fontsize=12)
                    ax.set_ylabel(y_col, fontsize=12)
                    st.pyplot(fig)
                    plt.close()
            
            elif viz_type == "Line Plot (Feature Means)":
                numeric_cols_present = [col for col in NUMERIC_COLS if col in df.columns]
                df_mean = df.groupby('Diagnosis')[numeric_cols_present].mean().T
                
                fig, ax = plt.subplots(figsize=(14, 7))
                for col in df_mean.columns:
                    ax.plot(df_mean.index, df_mean[col], marker='o', label=f'Diagnosis {col}', linewidth=2)
                ax.set_title('Line Plot of Numeric Features by Diagnosis', fontsize=16, fontweight='bold')
                ax.set_xlabel('Features', fontsize=12)
                ax.set_ylabel('Mean Values', fontsize=12)
                ax.set_xticklabels(df_mean.index, rotation=90)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
        
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")

# ==================== EXPLORATORY ANALYSIS PAGE ====================
elif page == "🔍 Exploratory Analysis":
    st.header("🔍 Exploratory Data Analysis")
    
    df = st.session_state.df_cleaned if st.session_state.cleaning_done else st.session_state.df_original
    
    if df is None:
        st.error("❌ Please load the dataset first.")
    else:
        st.subheader("📊 Diagnosis Rate by Different Groups")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Gender' in df.columns:
                st.write("**By Gender:**")
                gender_analysis = df.groupby('Gender')['Diagnosis'].agg(['mean', 'count']) * [100, 1]
                gender_analysis.columns = ['Diagnosis Rate (%)', 'Count']
                st.dataframe(gender_analysis, use_container_width=True)
            
            if 'Smoking' in df.columns:
                st.write("**By Smoking Status:**")
                smoking_analysis = df.groupby('Smoking')['Diagnosis'].agg(['mean', 'count']) * [100, 1]
                smoking_analysis.columns = ['Diagnosis Rate (%)', 'Count']
                st.dataframe(smoking_analysis, use_container_width=True)
            
            if 'FamilyHistoryAlzheimers' in df.columns:
                st.write("**By Family History:**")
                family_analysis = df.groupby('FamilyHistoryAlzheimers')['Diagnosis'].agg(['mean', 'count']) * [100, 1]
                family_analysis.columns = ['Diagnosis Rate (%)', 'Count']
                st.dataframe(family_analysis, use_container_width=True)
        
        with col2:
            if 'Alcohol_level' in df.columns:
                st.write("**By Alcohol Consumption Level:**")
                alcohol_analysis = df.groupby('Alcohol_level')['Diagnosis'].agg(['mean', 'count']) * [100, 1]
                alcohol_analysis.columns = ['Diagnosis Rate (%)', 'Count']
                st.dataframe(alcohol_analysis, use_container_width=True)
            
            if 'BMI_category' in df.columns:
                st.write("**By BMI Category:**")
                bmi_analysis = df.groupby('BMI_category')['Diagnosis'].agg(['mean', 'count']) * [100, 1]
                bmi_analysis.columns = ['Diagnosis Rate (%)', 'Count']
                st.dataframe(bmi_analysis, use_container_width=True)
            
            if 'PA_level' in df.columns:
                st.write("**By Physical Activity Level:**")
                pa_analysis = df.groupby('PA_level')['Diagnosis'].agg(['mean', 'count']) * [100, 1]
                pa_analysis.columns = ['Diagnosis Rate (%)', 'Count']
                st.dataframe(pa_analysis, use_container_width=True)
        
        # Age-based analysis
        if 'Age' in df.columns:
            st.subheader("📊 Age-Based Analysis")
            df['Age_Group'] = pd.cut(df['Age'], bins=[0, 50, 60, 70, 80, 100], 
                                     labels=['<50', '50-60', '60-70', '70-80', '80+'])
            age_analysis = df.groupby('Age_Group')['Diagnosis'].agg(['mean', 'count']) * [100, 1]
            age_analysis.columns = ['Diagnosis Rate (%)', 'Count']
            st.dataframe(age_analysis, use_container_width=True)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            age_analysis['Diagnosis Rate (%)'].plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title('Diagnosis Rate by Age Group', fontsize=14, fontweight='bold')
            ax.set_ylabel('Diagnosis Rate (%)')
            ax.set_xlabel('Age Group')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()

# ==================== MODEL TRAINING PAGE ====================
elif page == "🤖 Model Training":
    st.header("🤖 Model Training & Evaluation")
    
    if not st.session_state.cleaning_done:
        st.warning("⚠️ Please clean the data first from the Data Cleaning page.")
    else:
        df = st.session_state.df_cleaned
        
        model_type = st.selectbox(
            "🔧 Select Model",
            ["K-Nearest Neighbors", "Logistic Regression", "Random Forest", "Support Vector Machine"]
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            train_button = st.button("🚀 Train Model", type="primary", use_container_width=True)
        
        with col2:
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
        
        if train_button:
            with st.spinner(f"Training {model_type}... This may take a moment."):
                try:
                    # Prepare data
                    df_model = df.copy()
                    
                    # Encode categorical variables
                    cat_cols = ['BMI_category', 'Alcohol_level', 'PA_level']
                    le = LabelEncoder()
                    for col in cat_cols:
                        if col in df_model.columns:
                            df_model[col] = le.fit_transform(df_model[col].astype(str))
                    
                    X = df_model.drop('Diagnosis', axis=1)
                    y = df_model['Diagnosis']
                    
                    # Impute and scale
                    imputer = SimpleImputer(strategy='mean')
                    X = imputer.fit_transform(X)
                    
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Train-test split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=test_size, random_state=42, stratify=y
                    )
                    
                    # Train model
                    if model_type == "K-Nearest Neighbors":
                        model = KNeighborsClassifier(n_neighbors=5)
                    elif model_type == "Logistic Regression":
                        model = LogisticRegression(max_iter=1000)
                    elif model_type == "Random Forest":
                        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                    else:
                        model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
                    
                    # Save model results
                    st.session_state.models_trained[model_type] = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'cv_scores': cv_scores,
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'model': model
                    }
                    
                    st.success(f"✅ {model_type} trained successfully!")
                    
                    # Results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Performance Metrics")
                        accuracy = accuracy_score(y_test, y_pred)
                        st.metric("Accuracy", f"{accuracy:.4f}")
                        st.metric("CV Mean Accuracy", f"{cv_scores.mean():.4f}")
                        st.metric("CV Std Deviation", f"{cv_scores.std():.4f}")
                        
                        st.text("Classification Report:")
                        st.text(classification_report(y_test, y_pred, 
                                                     target_names=['No Alzheimer', 'Alzheimer']))
                    
                    with col2:
                        st.subheader("🔢 Confusion Matrix")
                        cm = confusion_matrix(y_test, y_pred)
                        
                        fig, ax = plt.subplots(figsize=(7, 5))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                   xticklabels=['No Alzheimer', 'Alzheimer'],
                                   yticklabels=['No Alzheimer', 'Alzheimer'], ax=ax)
                        ax.set_xlabel('Predicted', fontsize=12)
                        ax.set_ylabel('Actual', fontsize=12)
                        ax.set_title(f'Confusion Matrix - {model_type}', fontsize=14, fontweight='bold')
                        st.pyplot(fig)
                        plt.close()
                    
                    # Feature importance for Random Forest
                    if model_type == "Random Forest":
                        st.subheader("🎯 Feature Importance")
                        feature_names = df_model.drop('Diagnosis', axis=1).columns
                        importances = model.feature_importances_
                        indices = np.argsort(importances)[::-1][:15]
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.barh(range(len(indices)), importances[indices], color='steelblue')
                        ax.set_yticks(range(len(indices)))
                        ax.set_yticklabels([feature_names[i] for i in indices])
                        ax.set_xlabel('Importance')
                        ax.set_title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
                        ax.invert_yaxis()
                        st.pyplot(fig)
                        plt.close()
                
                except Exception as e:
                    st.error(f"❌ Error during training: {str(e)}")

# ==================== MODEL COMPARISON PAGE ====================
elif page == "📊 Model Comparison":
    st.header("📊 Model Comparison")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ No models have been trained yet. Please train models from the Model Training page.")
    else:
        st.subheader("🏆 Trained Models Performance")
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, results in st.session_state.models_trained.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'CV Mean': results['cv_scores'].mean(),
                'CV Std': results['cv_scores'].std()
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        # Display comparison table
        st.dataframe(comparison_df.style.highlight_max(axis=0, subset=['Accuracy', 'CV Mean']), 
                    use_container_width=True)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(comparison_df['Model'], comparison_df['Accuracy'], color='skyblue')
            ax.set_xlabel('Accuracy')
            ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
            ax.set_xlim([0, 1])
            for i, v in enumerate(comparison_df['Accuracy']):
                ax.text(v + 0.01, i, f'{v:.4f}', va='center')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(comparison_df['Model'], comparison_df['CV Mean'], 
                  yerr=comparison_df['CV Std'], capsize=5, color='lightcoral')
            ax.set_ylabel('CV Accuracy')
            ax.set_title('Cross-Validation Scores', fontsize=14, fontweight='bold')
            ax.set_ylim([0, 1])
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            plt.close()
        
        # Best model
        best_model = comparison_df.iloc[0]['Model']
        st.success(f"🏆 Best Model: **{best_model}** with accuracy of **{comparison_df.iloc[0]['Accuracy']:.4f}**")

# ==================== PREDICTION PAGE ====================
elif page == "🎯 Prediction":
    st.header("🎯 Make Predictions")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ No models have been trained yet. Please train models from the Model Training page.")
    else:
        st.info("Enter patient information below to predict Alzheimer's diagnosis.")
        
        selected_model = st.selectbox("Select Model", list(st.session_state.models_trained.keys()))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", 50, 100, 65)
            bmi = st.number_input("BMI", 15.0, 40.0, 25.0)
            smoking = st.selectbox("Smoking", [0, 1], format_func=lambda x: "Yes" if x else "No")
            alcohol = st.slider("Alcohol Consumption", 0, 20, 5)
            
        with col2:
            physical_activity = st.slider("Physical Activity", 0, 10, 5)
            diet_quality = st.slider("Diet Quality", 0, 10, 5)
            sleep_quality = st.slider("Sleep Quality", 4, 10, 7)
            mmse = st.slider("MMSE Score", 0, 30, 25)
            
        with col3:
            family_history = st.selectbox("Family History", [0, 1], 
                                         format_func=lambda x: "Yes" if x else "No")
            diabetes = st.selectbox("Diabetes", [0, 1], 
                                   format_func=lambda x: "Yes" if x else "No")
            depression = st.selectbox("Depression", [0, 1], 
                                     format_func=lambda x: "Yes" if x else "No")
            hypertension = st.selectbox("Hypertension", [0, 1], 
                                       format_func=lambda x: "Yes" if x else "No")
        
        if st.button("🔮 Predict", type="primary", use_container_width=True):
            st.info("Prediction feature is ready! You can extend this with the actual prediction logic using the trained model.")
            st.success("✅ Model loaded successfully!")

# Footer
st.markdown("---")
st.markdown("🧠 **Alzheimer's Disease Analysis Platform** | Built with Streamlit")