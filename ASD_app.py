import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from model_utils import handle_child_model
from model_utils import handle_adult_model
import openai
import numpy as np
import google.generativeai as genai
import requests
import googlemaps



# ---------------------------
# Google Gemini API Key
# ---------------------------
genai.configure(api_key=st.secrets["api_keys"]["genai_key"])

# Google Maps API Key
GOOGLE_MAPS_API_KEY = st.secrets["api_keys"]["gmap_key"]

# Initialize Google Maps Client
gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)


st.markdown("""
    <style>
    /* Import fonts: Poppins Bold for headings, The Season for body text */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=The+Season&display=swap');

    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .btn_one {
        color: white;
        font-size: 30px;
        position: absolute;
        left: 16px;
        line-height: 60px;
        cursor: pointer;
        z-index: 999;
    }
    
    .sidebar_menu {
        background: rgba(255,255,255,0.1);
        box-shadow: 0 0 6px 0 rgba(255,255,255,0.5);
        transition: all 0.3s linear;
    }

    /* ================================
       Custom Sidebar Navigation Styling
       ================================ */
    [data-testid="stSidebar"] .stRadio label {
        display: block; 
        font-size: 20px;              /* Bigger text for options */
        margin: 15px 0;               /* Space between options */
        padding: 10px 0;              /* Minimal vertical padding */
        background: rgba(255,255,255,0.1) !important;      /* No default background */
        width: 100%;                  /* Take full width of the sidebar */
        box-shadow: 0 0 6px 0 rgba(255,255,255,0.5) !important;
        transition: all 0.3s linear !important;
        border: none;                 
        border-radius: 4px;             
        text-align: left;             
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        background-color: white !important;
        color: black !important;
        padding: 10px !important;
        box-shadow: 0 0 4px 0 rgba(255,255,255,0.5) !important;
        font-weight: bold !important;
    }

    .btn_two i {
        color: grey;
        font-size: 25px;
        line-height: 60px;
        position: absolute;
        left: 275px;
        cursor: pointer;
    }

    .sidebar_menu .menu li:hover {
        box-shadow: 0 0 4px 0 rgba(255,255,255,0.5);
    }

    .btn_one i:hover, .btn_two i:hover {
        transform: scale(1.1);
        transition: all 0.3s linear;
    }

    /* Global styling: Use "The Season" for body text */
    html, body, .stApp {
        height: 100% !important;
        margin: 0;
        padding: 0;
        background: linear-gradient(135deg, #FFC1CC, #FFD3FF, #FFFACD) !important;
        background-size: cover;
        background-repeat: no-repeat;
        color: #333;
        font-family: 'The Season', sans-serif;
        scroll-behavior: smooth;
    }

    /* Main content container */
    .main .block-container {
        background: rgba(255, 255, 255, 0.75) !important;
        border: 1px solid #ddd !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1) !important;
        padding: 2rem !important;
        margin: 2rem auto !important;
        max-width: 1200px;
    }

    /* Responsive adjustments for main container */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem !important;
            margin: 1rem auto !important;
            max-width: 95% !important;
        }
    }

    /* Tab container: evenly spaced tabs across the top */
    .css-1pahdxg {
        display: flex !important;
        justify-content: space-evenly !important;
        border-bottom: 2px solid #ddd;
        width: 100% !important;
    }
    .css-1pahdxg .css-1v3fvcr {
        flex: 1 !important;
        margin: 0 5px !important;
        min-width: 120px !important;
        background-color: rgba(255,255,255,0.3) !important;
        padding: 0.8rem 1.5rem !important;
        border: none !important;
        border-radius: 8px 8px 0 0 !important;
        font-weight: 600 !important;
        transition: background-color 0.3s ease, transform 0.3s ease;
    }
    .css-1pahdxg .css-1v3fvcr:hover {
        background-color: rgba(255,255,255,0.5) !important;
        transform: translateY(-2px);
        cursor: pointer;
    }
    .css-1pahdxg .css-1v3fvcr[aria-selected="true"] {
        background-color: #FFFACD !important;
        border-bottom: 2px solid transparent !important;
    }

    /* Headings: Use Poppins Bold */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 700 !important;
        color: #111;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Responsive typography */
    @media (max-width: 768px) {
      h1 { font-size: 1.8rem !important; }
      h2 { font-size: 1.6rem !important; }
      h3 { font-size: 1.4rem !important; }
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #FFC1CC, #FFD3FF, #FFFACD) !important;
        color: #333 !important;
        border: 1px solid #ccc;
        border-radius: 5px;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.6rem 1rem;
        cursor: pointer;
        transition: transform 0.3s ease, box-shadow 0.3s ease, background 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        background: linear-gradient(135deg, #FFFACD, #FFD3FF, #FFC1CC) !important;
        color: #111 !important;
    }

    /* Inputs */
    .stTextInput>div>div>input, 
    .stSelectbox>div>div>div>input, 
    .stNumberInput input {
        background-color: #FFFFFF !important;
        color: #333 !important;
        border-radius: 5px;
        border: 1px solid #ccc !important;
    }

    /* Dataframes */
    .dataframe {
        border: 1px solid #ccc !important;
        border-radius: 5px;
        margin-bottom: 1rem;
        background-color: rgba(255, 255, 255, 0.8) !important;
    }

    /* Expander text boxes (.bordered-text) with 50% opacity */
    .bordered-text {
        border: 1px solid #ccc;
        border-radius: 6px;
        padding: 1rem;
        background-color: rgba(255, 255, 255, 0.5) !important;
        margin-bottom: 1rem;
    }

    

    /* Hide empty label elements (the blank box) */
    [data-testid="stSidebar"] .stRadio label:empty {
        display: none !important;
    }

    /* Style for the radio group container */
    [data-testid="stSidebar"] .stRadio {
        background: transparent;
        border: none;
        padding: 0;
    }

    /* Smooth animations for users with reduced motion preference */
    @media (prefers-reduced-motion: reduce) {
        *,
        *::before,
        *::after {
            transition: none !important;
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
        }
    }
    @media (max-width: 768px) {
        [data-testid="stSidebar"] {
            width: 250px !important;
        }
        
        .btn_one {
            font-size: 25px;
            left: 10px;
        }
        
        .sidebar_menu .menu li {
            padding: 10px 15px;
        }
    }
    </style>
""", unsafe_allow_html=True)



# ---------------------------
# Helper Functions
# ---------------------------
def preprocess_data(df, target_column):
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip().str.lower()
    for col in ['Age_Mons', 'id', 'result']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    if df[target_column].dtype == 'object':
        df[target_column] = df[target_column].map({'yes': 1, 'no': 0})
        df[target_column] = df[target_column].fillna(0)
    label_encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object' and col != target_column:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    return df, label_encoders

def generate_gemini_response(user_input):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content([
            {"role": "user", "parts": [{"text": user_input}]}
        ])
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def get_user_location():
    """Fetches location using IP-based geolocation"""
    try:
        gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
        location = gmaps.geolocate()
        
        if 'location' in location:
            return location['location']['lat'], location['location']['lng']
        return 28.7041, 77.1025  # Default coordinates
    
    except Exception as e:
        st.error(f"Location error: {str(e)}")
        return 28.7041, 77.1025

def get_nearby_hospitals(lat, lon):
    """Fetches nearby hospitals using Google Places API"""
    try:
        gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
        places_result = gmaps.places_nearby(
            location=(lat, lon), 
            radius=5000, 
            type='hospital'
        )
        return places_result.get("results", [])
    except Exception as e:
        st.error(f"Hospital search error: {str(e)}")
        return []

def get_nearby_counselors(lat, lon):
    """Fetch nearby mental health counselors"""
    try:
        gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
        places_result = gmaps.places_nearby(
            location=(lat, lon),
            radius=5000,
            keyword='mental health counselor'
        )
        return places_result.get("results", [])
    except Exception as e:
        st.error(f"Counselor search error: {str(e)}")
        return []
  

# Sidebar Navigation
with st.sidebar:
    st.title("🔍 Navigation")
    option = st.radio(
        label="",  # Hide label
        options=["🏠 Home", "📊 Data Analysis", "⚡ AI Chatbot", "🚑 Help"],
        index=0,  # Set default to the first option so no blank option appears
        label_visibility="collapsed"
    )
# Home Section with Tabs
if option == "🏠 Home":
    st.subheader("Welcome to the Home Page!")
    
    tab1, tab2, tab3 = st.tabs(["🏡 Welcome", "📖 Instructions", "📊 Sample Dataset"])
    
    with tab1:
        st.write("This app helps you with predictions and data analysis.")
        st.image("PinkASDHeader.png", use_container_width=True)
        st.title("Welcome to the ASD Detection System!")
        st.markdown("Explore the sections below to learn about ASD and test detection models for both children and adults.")
        
        with st.expander("What is ASD?"):
            st.markdown("""
            Autism Spectrum Disorder (ASD) is a neurodevelopmental condition characterized by differences in social communication and behavior. Individuals with ASD often experience unique ways of interacting with the world, which can include challenges in interpreting social cues and engaging in typical communication patterns, alongside strengths such as attention to detail and intense focus on personal interests. Because ASD is a spectrum, the presentation and support needs vary widely from one person to another, making personalized approaches essential for development and inclusion.
            """)
            
        with st.expander("ASD in Children"):
            st.markdown("""
            ASD in children often manifests as difficulties in social interaction, communication challenges, and repetitive behaviors. Some children may struggle with eye contact, responding to their names, or understanding emotions, while others may have intense interests in specific topics. Early diagnosis and intervention can be crucial to support their development.
            """)
            
        with st.expander("ASD in Adults"):
            st.markdown("""
            ASD in adults may present differently than in childhood. Adults might develop coping strategies, yet still face challenges in social relationships, sensory sensitivities, and executive functioning. Recognizing adult autism can lead to better workplace accommodations, social support, and personalized interventions.
            """)
            
        with st.expander("How This Web App Helps"):
            st.markdown("""
            This web app assists in the detection of ASD by leveraging data analysis and machine learning. You can upload either a children or adult dataset, train or load a model, and make predictions about ASD. Additionally, an AI-powered chatbot is available to answer your questions about ASD.
            """)
    
    with tab2:
        st.title("Instructions")
        st.write("Follow these steps to use the app effectively.")
        st.markdown("""
        <div class="bordered-text">
        How to Use this Web App:<br><br>
        1. <b>Sample Datasets:</b> Go to the 'Sample Dataset' tab to preview and download sample datasets for both child and adult autism.<br><br>
        2. <b>Child Prediction:</b> In the 'Child Prediction' tab, upload your <code>Autism-Child-Data.csv</code> file or use the sample data. The app will preprocess the data, load or train a RandomForest model, evaluate its performance, and allow you to make predictions.<br><br>
        3. <b>Adult Prediction:</b> In the 'Adult Prediction' tab, upload your <code>Adult_autism_screening.csv</code> file or use the sample data. Similar processing and prediction steps will be applied to detect ASD in adults.<br><br>
        4. <b>AI Chatbot:</b> Ask any questions related to ASD in the 'AI Chatbot' tab and receive AI-powered responses.<br><br>
        <i>Note: This tool is for educational purposes only and should not replace professional medical advice.</i>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.title("Sample ASD Datasets")
        st.markdown("Download sample datasets to test the ASD Detection System.")
        sample_tabs = st.tabs(["Child Dataset", "Adult Dataset"])
        
        with sample_tabs[0]:
            st.subheader("Child Autism Dataset")
            try:
                df_child = pd.read_csv("Autism-Child-Data.csv")
                st.write(df_child.head())
                csv_data_child = df_child.to_csv(index=False)
                st.download_button(
                    label="Download Child Autism CSV",
                    data=csv_data_child,
                    file_name="Autism-Child-Data.csv",
                    mime="text/csv"
                )
            except FileNotFoundError:
                st.error("Autism-Child-Data.csv not found. Please place it in the same directory as this app.")
                
        with sample_tabs[1]:
            st.subheader("Adult Autism Dataset")
            try:
                df_adult = pd.read_csv("Adult_autism_screening.csv")
                st.write(df_adult.head())
                csv_data_adult = df_adult.to_csv(index=False)
                st.download_button(
                    label="Download Adult Autism CSV",
                    data=csv_data_adult,
                    file_name="Adult_autism_screening.csv",
                    mime="text/csv"
                )
            except FileNotFoundError:
                st.error("Adult_autism_screening.csv not found. Please place it in the same directory as this app.")

# Data Analysis Section with Child & Adult Prediction Tabs
elif option == "📊 Data Analysis":
    st.subheader("Analyze Your Data Here!")
    
    tab1, tab2 = st.tabs(["🧒 Child Prediction", "🧑 Adult Prediction"])
    
    with tab1:
        st.title("ASD Prediction for Children")
        st.write("### Perform predictions based on child data.")
        uploaded_file = st.file_uploader("Upload your Autism-Child-Data.csv file", type=["csv"], key="child")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.subheader("Dataset Preview")
            st.write(df.head())
            
            target_column = "Class/ASD"
            st.write(f"Detected Target Column: *{target_column}*")
            
            df_processed, encoders = preprocess_data(df.copy(), target_column)
            X = df_processed.drop(columns=[target_column])
            y = df_processed[target_column].astype(int)
            
            model, retrain_flag = handle_child_model(X, y)
            
            # Model Evaluation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            accuracy = model.score(X_test, y_test)
            st.subheader(f"Model Accuracy on Test Split: {accuracy:.2%}")
            cv_scores = cross_val_score(model, X, y, cv=5)
            st.markdown("*Cross-Validation Scores:*")
            fig_cv, ax_cv = plt.subplots(figsize=(6, 4))
            ax_cv.bar(range(len(cv_scores)), cv_scores, color='#bbb')
            ax_cv.set_xlabel("Fold")
            ax_cv.set_ylabel("CV Score")
            ax_cv.set_ylim([0, 1])
            ax_cv.set_title("CV Scores")
            st.pyplot(fig_cv)
            st.markdown(f"*Mean CV Accuracy:* {cv_scores.mean():.2%}")

            # TABS for Confusion Matrix & Classification Report
            tab1, tab2 = st.tabs(["Confusion Matrix", "Classification Report"])
            with tab1:
                st.markdown("*Confusion Matrix (Heatmap):*")
                preds = model.predict(X_test)
                cm = confusion_matrix(y_test, preds)
                fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                st.pyplot(fig_cm)
            with tab2:
                st.markdown("*Classification Report:*")
                report_dict = classification_report(y_test, preds, output_dict=True)
                df_report = pd.DataFrame(report_dict).transpose()
                st.dataframe(df_report.style.background_gradient(cmap='Blues', axis=1))
            
            # Prediction Inputs
            st.subheader("Make a Prediction")
            user_input = {}
            for col in X.columns:
                if col in encoders:
                    original_categories = list(encoders[col].classes_)
                    user_input[col] = st.selectbox(f"Select {col}", original_categories)
                else:
                    min_val = float(X[col].min())
                    max_val = float(X[col].max())
                    median_val = float(X[col].median())
                    user_input[col] = st.number_input(f"Enter value for {col}", min_val, max_val, median_val)
            
            if st.button("Predict", key="predict_child"):
                st.write("Current user inputs:", user_input)
                input_df = pd.DataFrame([user_input])
                for col, le in encoders.items():
                    if col in input_df.columns:
                        user_value = input_df[col].iloc[0]
                        valid_classes = set(le.classes_)
                        if user_value not in valid_classes:
                            st.error(f"Unseen category '{user_value}' for column '{col}'. Please pick a valid option.")
                            st.stop()
                        else:
                            input_df[col] = le.transform(input_df[col])
                prediction = model.predict(input_df)[0]
                result_text = "ASD Positive" if prediction == 1 else "ASD Negative"
                st.success(f"Prediction: *{result_text}*")
            
            # Data Analysis
            st.subheader("Data Analysis")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(x=df[target_column], ax=ax)
            ax.set_title("ASD Classification Distribution")
            st.pyplot(fig)
            with st.expander("Show Correlation Heatmap"):
                corr = df_processed.corr()
                fig_corr, ax_corr = plt.subplots(figsize=(8, 4))
                sns.heatmap(corr, ax=ax_corr, cmap='coolwarm', annot=False)
                ax_corr.set_title("Correlation Heatmap")
                st.pyplot(fig_corr)
    
    with tab2:
        st.title("ASD Prediction for Adults")
        st.write("### Perform predictions based on adult data.")
        uploaded_file = st.file_uploader("Upload your Adult_autism_screening.csv file", type=["csv"], key="adult")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.subheader("Dataset Preview")
            st.write(df.head())
            
            target_column = "Class/ASD"
            st.write(f"Detected Target Column: *{target_column}*")
            
            df_processed, encoders = preprocess_data(df.copy(), target_column)
            X = df_processed.drop(columns=[target_column])
            y = df_processed[target_column].astype(int)
            
            model, retrain_flag = handle_adult_model(X, y)
            
            # Model Evaluation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            accuracy = model.score(X_test, y_test)
            st.subheader(f"Model Accuracy on Test Split: {accuracy:.2%}")
            cv_scores = cross_val_score(model, X, y, cv=5)
            st.markdown("*Cross-Validation Scores:*")
            fig_cv, ax_cv = plt.subplots(figsize=(6, 4))
            ax_cv.bar(range(len(cv_scores)), cv_scores, color='#bbb')
            ax_cv.set_xlabel("Fold")
            ax_cv.set_ylabel("CV Score")
            ax_cv.set_ylim([0, 1])
            ax_cv.set_title("CV Scores")
            st.pyplot(fig_cv)
            st.markdown(f"*Mean CV Accuracy:* {cv_scores.mean():.2%}")

            # TABS for Confusion Matrix & Classification Report
            tab1, tab2 = st.tabs(["Confusion Matrix", "Classification Report"])
            with tab1:
                st.markdown("*Confusion Matrix (Heatmap):*")
                preds = model.predict(X_test)
                cm = confusion_matrix(y_test, preds)
                fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                st.pyplot(fig_cm)
            with tab2:
                st.markdown("*Classification Report:*")
                report_dict = classification_report(y_test, preds, output_dict=True)
                df_report = pd.DataFrame(report_dict).transpose()
                st.dataframe(df_report.style.background_gradient(cmap='Blues', axis=1))
            
            # Prediction Inputs
            st.subheader("Make a Prediction")
            user_input = {}
            for col in X.columns:
                if col in encoders:
                    original_categories = list(encoders[col].classes_)
                    user_input[col] = st.selectbox(f"Select {col}", original_categories)
                else:
                    min_val = float(X[col].min())
                    max_val = float(X[col].max())
                    median_val = float(X[col].median())
                    user_input[col] = st.number_input(f"Enter value for {col}", min_val, max_val, median_val)
            
            if st.button("Predict", key="predict_adult"):
                st.write("Current user inputs:", user_input)
                input_df = pd.DataFrame([user_input])
                for col, le in encoders.items():
                    if col in input_df.columns:
                        user_value = input_df[col].iloc[0]
                        valid_classes = set(le.classes_)
                        if user_value not in valid_classes:
                            st.error(f"Unseen category '{user_value}' for column '{col}'. Please pick a valid option.")
                            st.stop()
                        else:
                            input_df[col] = le.transform(input_df[col])
                prediction = model.predict(input_df)[0]
                result_text = "ASD Positive" if prediction == 1 else "ASD Negative"
                st.success(f"Prediction: *{result_text}*")
            
            # Data Analysis
            st.subheader("Data Analysis")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(x=df[target_column], ax=ax)
            ax.set_title("ASD Classification Distribution")
            st.pyplot(fig)
            with st.expander("Show Correlation Heatmap"):
                corr = df_processed.corr()
                fig_corr, ax_corr = plt.subplots(figsize=(8, 4))
                sns.heatmap(corr, ax=ax_corr, cmap='coolwarm', annot=False)
                ax_corr.set_title("Correlation Heatmap")
                st.pyplot(fig_corr)
#Chatbot Section
elif option == "⚡ AI Chatbot":
    st.title("AI-Powered Chatbot")
    st.markdown("Ask any questions related to Autism Spectrum Disorder (ASD) and get responses powered by AI.")
    
    user_query = st.text_input("Your Question:")
    
    if st.button("Send"):
        if user_query:
            with st.spinner("Generating response..."):
                response = generate_gemini_response(user_query)  # Use Gemini API function
            st.markdown("*Chatbot:*")
            st.write(response)
        else:
            st.error("Please enter a question.")
    st.markdown("*Disclaimer:* This chatbot is for informational purposes only and should not replace professional medical advice.")

elif option == "🚑 Help":
    st.subheader("Emergency Contacts & Medical Facilities")
    
    # Get location using IP-based geolocation
    user_lat, user_lon = get_user_location()
    st.warning("Using approximate location based on IP address")

    tab1, tab2 = st.tabs(["🏥 Medical Facilities", "📞 Counseling & Mental Health Support"])
    
    with tab1:
        st.write("### Nearby Hospitals & Ambulance Services")
        st.markdown("ℹ️ **Click on the hospital names to get directions on Google Maps.**")
        
        if user_lat and user_lon:
            with st.spinner("Searching nearby hospitals..."):
                hospitals = get_nearby_hospitals(user_lat, user_lon)
            
            if hospitals:
                st.map(pd.DataFrame(
                    [[h["geometry"]["location"]["lat"], h["geometry"]["location"]["lng"]] 
                    for h in hospitals
                ], columns=["lat", "lon"]))

                st.write("#### 🏥 Nearby Hospitals (within 5km)")
                for h in hospitals:
                    name = h.get("name", "Unnamed Hospital")
                    address = h.get("vicinity", "Address Not Available")
                    rating = h.get("rating", "N/A")
                    lat, lon = h["geometry"]["location"]["lat"], h["geometry"]["location"]["lng"]
                    
                    maps_url = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
                    st.markdown(
                        f"""
                        <div style="margin-bottom: 1rem;">
                            <a href="{maps_url}" target="_blank" style="text-decoration: none;">
                                <h4 style="margin: 0; color: #2b5876;">📍 {name}</h4>
                            </a>
                            <p style="margin: 0.2rem 0; color: #666;">{address}</p>
                            <p style="margin: 0; color: #888;">Google Rating: {rating}/5</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.warning("No hospitals found within 5km radius")
        else:
            st.error("Location services unavailable")

    with tab2:
        st.write("### Mental Health & Counseling Support")
        st.markdown("ℹ️ Families can seek help from IAC (India Autism Center) by: ")
        st.markdown("✉️ info@indiaautismcenter.org ")
        st.markdown("📞 +91 90511 10656 ")
        st.markdown("[Visit India Autism Center Website](https://www.indiaautismcenter.org/)")
        
        if user_lat and user_lon:
            with st.spinner("Searching mental health professionals..."):
                counselors = get_nearby_counselors(user_lat, user_lon)
            
            if counselors:
                st.write("#### 🧠 Nearby Mental Health Counselors (within 5km)")
                for c in counselors:
                    name = c.get("name", "Unnamed Center")
                    address = c.get("vicinity", "Address Not Available")
                    phone = c.get("formatted_phone_number", "Not Available")
                    website = c.get("website", "")
                    
                    st.markdown(
                        f"""
                        <div style="margin-bottom: 1rem;">
                            <h4 style="margin: 0; color: #2b5876;">🧠 {name}</h4>
                            <p style="margin: 0.2rem 0; color: #666;">{address}</p>
                            <p style="margin: 0; color: #888;">📞 {phone}</p>
                            {f'<a href="{website}" target="_blank" style="color: #3b8ed6;">Website</a>' if website else ""}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.warning("No mental health services found within 5km radius")
        else:
            st.error("Location services unavailable")
