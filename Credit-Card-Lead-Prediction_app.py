import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from category_encoders import BinaryEncoder

# Load dataset
def load_data():
    data = pd.read_csv('Credit Card Data.csv')
    data.drop(columns=['ID'], inplace=True)
    data['Vintage'].fillna(data['Vintage'].median(), inplace=True)
    data['Credit_Product'].fillna('Unknown', inplace=True)
    return data

data = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Exploratory Data Analysis", "Model Prediction"])

# Preprocessing and encoding
def preprocess_data(data):
    le = LabelEncoder()
    data['Gender'] = le.fit_transform(data['Gender'])
    data['Is_Active'] = le.fit_transform(data['Is_Active'])
    x = data.drop('Is_Lead', axis=1)
    y = data['Is_Lead']

    transformer = ColumnTransformer(
        transformers=[
            ('ohe', OneHotEncoder(sparse_output=False, drop='first'), ['Occupation']),
            ('be', BinaryEncoder(), ['Credit_Product'])
        ], remainder='passthrough'
    )
    return x, y, transformer

# EDA Page
if page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")

    if st.checkbox("Show Data Overview"):
        st.dataframe(data.head())
        st.write("Shape of the dataset:", data.shape)
        st.write("Missing values:", data.isnull().sum())

    st.subheader("Univariate Analysis")
    feature = st.selectbox("Choose a feature to analyze:", ['Age', 'Avg_Account_Balance'])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data[feature], kde=True, ax=ax, color='blue')
    ax.set_title(f"Distribution of {feature}")
    st.pyplot(fig)

    st.subheader("Bivariate Analysis")
    
    st.subheader("Pairplot (Is_Lead)")
    fig = sns.pairplot(data, hue='Is_Lead')
    st.pyplot(fig)
    
        # Proportion of Leads by Occupation
    st.subheader("Proportion of Leads by Occupation")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Occupation', y='Is_Lead', data=data, ci=None, palette='viridis', ax=ax)
    ax.set_title('Proportion of Leads by Occupation')
    ax.set_xlabel('Occupation')
    ax.set_ylabel('Proportion of Leads')
    st.pyplot(fig)
    
    st.subheader("Distribution of Leads (Is_Lead)")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Is_Lead', data=data, palette='viridis', ax=ax)
    ax.set_title('Distribution of Leads (Is_Lead)')
    ax.set_xlabel('Is_Lead (0 = Non-Lead, 1 = Lead)')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    
        # Proportion of Leads by Is_Active
    st.subheader("Proportion of Leads by Is_Active")

    # Calculate the proportion of leads
    lead_proportion = data.groupby('Is_Active')['Is_Lead'].mean()

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    lead_proportion.plot(kind='bar', color=['orange', 'black'], ax=ax)
    ax.set_title('Proportion of Leads by Is_Active')
    ax.set_xlabel('Is_Active')
    ax.set_ylabel('Proportion of Leads')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No', 'Yes'], rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Display the plot in Streamlit
    st.pyplot(fig)

    
# Model Prediction Page
elif page == "Model Prediction":
    st.title("Model Prediction")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    model_choice = st.selectbox("Choose a model:", list(models.keys()))
    selected_model = models[model_choice]

    x, y, transformer = preprocess_data(data)

    with st.form("prediction_form"):
        st.subheader("Input Features")
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        avg_balance = st.number_input("Average Account Balance", min_value=0.0, value=50000)
        gender = st.selectbox("Gender", ["Male", "Female"])
        occupation = st.selectbox("Occupation", ["Salaried", "Self_Employed", "Other"])
        credit_product = st.selectbox("Credit Product", ["Yes", "No", "Unknown"])
        is_active = st.selectbox("Is Active", ["Yes", "No"])
        vintage = st.number_input("Vintage (months)", min_value=0, max_value=240, value=12)

        submit_button = st.form_submit_button(label="Predict")

    if submit_button:
        input_data = pd.DataFrame.from_dict({
            'Age': [age], 'Avg_Account_Balance': [avg_balance], 'Gender': [1 if gender == 'Male' else 0],
            'Occupation': [occupation], 'Credit_Product': [credit_product], 'Is_Active': [1 if is_active == 'Yes' else 0],
            'Vintage': [vintage]
        })

        pipeline = Pipeline([
            ('transformer', transformer),
            ('scaler', StandardScaler()),
            ('model', selected_model)
        ])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        pipeline.fit(x_train, y_train)
        prediction = pipeline.predict(input_data)
        st.success(f"Predicted Lead Status: {'Lead' if prediction[0] == 1 else 'Non-Lead'}")
