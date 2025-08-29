import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import os
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="ML-Based Field Prediction for Rwanda Polytechnic",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stTitle {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-result {
        background-color: #ecf0f1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    .error-message {
        color: #e74c3c;
        font-size: 0.9em;
        margin-top: 0.2rem;
    }
    .success-message {
        color: #27ae60;
        font-weight: bold;
    }
    .batch-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load models and encoders
@st.cache_resource
def load_models():
    try:
        # Get the directory where the script is located
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct absolute paths
        model_path = os.path.join(base_dir, "model", "decision_tree_model.pkl")
        encoder_path = os.path.join(base_dir, "model", "one_hot_encoder.pkl")
        field_encoder_path = os.path.join(base_dir, "model", "label_encoder.pkl")
        
        model = load(model_path)
        encoder = load(encoder_path)
        field_encoder = load(field_encoder_path)
        return model, encoder, field_encoder
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.stop()

# Define the Excel file path
EXCEL_FILE_PATH = "predictions_log.xlsx"

def preprocess_user_input(user_data, encoder):
    """Preprocess user input to match the model's input format."""
    user_df = pd.DataFrame([user_data])
    
    # One-hot encode High_School_Stream
    stream_encoded = encoder.transform(user_df[["High_School_Stream"]])
    stream_encoded_df = pd.DataFrame(stream_encoded, columns=encoder.get_feature_names_out(["High_School_Stream"]))
    
    # Combine encoded features with subject scores
    user_processed = pd.concat([stream_encoded_df, user_df.iloc[:, 1:]], axis=1)
    
    return user_processed

def preprocess_batch_data(batch_df, encoder):
    """Preprocess batch data to match the model's input format."""
    # One-hot encode High_School_Stream
    stream_encoded = encoder.transform(batch_df[["High_School_Stream"]])
    stream_encoded_df = pd.DataFrame(stream_encoded, columns=encoder.get_feature_names_out(["High_School_Stream"]))
    
    # Get subject columns (exclude High_School_Stream)
    subject_columns = [col for col in batch_df.columns if col != "High_School_Stream"]
    
    # Combine encoded features with subject scores
    batch_processed = pd.concat([stream_encoded_df, batch_df[subject_columns]], axis=1)
    
    return batch_processed

def save_to_excel(user_data, predicted_field):
    """Save the input data and predicted field to an Excel file."""
    data = {
        "High_School_Stream": [user_data["High_School_Stream"]],
        "Physics": [user_data["Physics"]],
        "Math": [user_data["Math"]],
        "Chemistry": [user_data["Chemistry"]],
        "Biology": [user_data["Biology"]],
        "Economics": [user_data["Economics"]],
        "Geography": [user_data["Geography"]],
        "History": [user_data["History"]],
        "Literature": [user_data["Literature"]],
        "Predicted_Field": [predicted_field]
    }
    df = pd.DataFrame(data)
    
    if os.path.exists(EXCEL_FILE_PATH):
        existing_df = pd.read_excel(EXCEL_FILE_PATH)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        updated_df = df
    
    updated_df.to_excel(EXCEL_FILE_PATH, index=False)

def save_batch_to_excel(batch_df):
    """Save batch predictions to Excel file."""
    if os.path.exists(EXCEL_FILE_PATH):
        existing_df = pd.read_excel(EXCEL_FILE_PATH)
        updated_df = pd.concat([existing_df, batch_df], ignore_index=True)
    else:
        updated_df = batch_df
    
    updated_df.to_excel(EXCEL_FILE_PATH, index=False)

def validate_scores(stream, scores):
    """Validate scores based on the selected stream."""
    stream_subjects = {
        "PCM": ["Physics", "Chemistry", "Math"],
        "PCB": ["Physics", "Chemistry", "Biology"],
        "MEG": ["Math", "Economics", "Geography"],
        "MPG": ["Math", "Physics", "Geography"],
        "HGL": ["History", "Geography", "Literature"]
    }
    
    applicable_subjects = stream_subjects.get(stream, [])
    errors = {}
    
    for subject, score in scores.items():
        if subject in applicable_subjects:
            if not (0 <= score <= 100):
                errors[subject] = f"{subject} score must be between 0 and 100."
        else:
            if score != 0:
                errors[subject] = f"{subject} must be 0 as it's not part of {stream} stream."
    
    return errors

def validate_batch_file(df, stream):
    """Validate uploaded batch file format and content."""
    stream_subjects = {
        "PCM": ["Physics", "Chemistry", "Math"],
        "PCB": ["Physics", "Chemistry", "Biology"],
        "MEG": ["Math", "Economics", "Geography"],
        "MPG": ["Math", "Physics", "Geography"],
        "HGL": ["History", "Geography", "Literature"]
    }
    
    applicable_subjects = stream_subjects.get(stream, [])
    errors = []
    
    # Check if all applicable subjects are present
    missing_subjects = [subj for subj in applicable_subjects if subj not in df.columns]
    if missing_subjects:
        errors.append(f"Missing required columns for {stream} stream: {', '.join(missing_subjects)}")
    
    # Check for invalid scores
    for subject in applicable_subjects:
        if subject in df.columns:
            invalid_scores = df[(df[subject] < 0) | (df[subject] > 100)]
            if not invalid_scores.empty:
                errors.append(f"Invalid scores found in {subject} column (must be between 0-100)")
    
    return errors

def prepare_batch_data(df, stream):
    """Prepare batch data by adding missing subjects with 0 scores and High_School_Stream."""
    all_subjects = ["Physics", "Math", "Chemistry", "Biology", "Economics", "Geography", "History", "Literature"]
    stream_subjects = {
        "PCM": ["Physics", "Chemistry", "Math"],
        "PCB": ["Physics", "Chemistry", "Biology"],
        "MEG": ["Math", "Economics", "Geography"],
        "MPG": ["Math", "Physics", "Geography"],
        "HGL": ["History", "Geography", "Literature"]
    }
    
    applicable_subjects = stream_subjects.get(stream, [])
    
    # Add High_School_Stream column
    df["High_School_Stream"] = stream
    
    # Add missing subjects with 0 scores
    for subject in all_subjects:
        if subject not in df.columns:
            df[subject] = 0
    
    # Reorder columns to match expected format
    column_order = ["High_School_Stream"] + all_subjects
    df = df[column_order]
    
    return df

def generate_sample_file(stream):
    """Generate a sample CSV file for the selected stream."""
    stream_subjects = {
        "PCM": ["Physics", "Chemistry", "Math"],
        "PCB": ["Physics", "Chemistry", "Biology"],
        "MEG": ["Math", "Economics", "Geography"],
        "MPG": ["Math", "Physics", "Geography"],
        "HGL": ["History", "Geography", "Literature"]
    }
    
    applicable_subjects = stream_subjects.get(stream, [])
    
    # Create sample data
    sample_data = {}
    for subject in applicable_subjects:
        sample_data[subject] = [85, 78, 92, 67, 89]
    
    sample_df = pd.DataFrame(sample_data)
    
    # Convert to CSV
    csv_buffer = BytesIO()
    sample_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    return csv_buffer.getvalue()

def main():
    # Load models
    model, encoder, field_encoder = load_models()
    
    # Title
    st.title("🎓 ML-Based Field Recommendation")
    st.subheader("Rwanda Polytechnic - Academic Field Prediction Research")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📝 Single Prediction", "📊 Batch Prediction", "📈 Prediction History"])
    
    with tab1:
        st.markdown("### Enter Student Information")
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            # High School Stream selection
            stream = st.selectbox(
                "High School Stream:",
                ["", "PCM", "PCB", "MEG", "MPG", "HGL"],
                format_func=lambda x: "Select Stream" if x == "" else {
                    "PCM": "PCM (Physics, Chemistry, Math)",
                    "PCB": "PCB (Physics, Chemistry, Biology)", 
                    "MEG": "MEG (Math, Economics, Geography)",
                    "MPG": "MPG (Math, Physics, Geography)",
                    "HGL": "HGL (History, Geography, Literature)"
                }.get(x, x),
                key="single_stream"
            )
            
            if stream:
                # Define applicable subjects for the selected stream
                stream_subjects = {
                    "PCM": ["Physics", "Chemistry", "Math"],
                    "PCB": ["Physics", "Chemistry", "Biology"],
                    "MEG": ["Math", "Economics", "Geography"],
                    "MPG": ["Math", "Physics", "Geography"],
                    "HGL": ["History", "Geography", "Literature"]
                }
                
                applicable_subjects = stream_subjects.get(stream, [])
                
                # Display applicable subjects info
                st.info(f"**Applicable subjects for {stream}:** {', '.join(applicable_subjects)}")
        
        if stream:
            # Subject scores input - only show applicable subjects
            col1, col2 = st.columns(2)
            
            all_subjects = ["Physics", "Math", "Chemistry", "Biology", "Economics", "Geography", "History", "Literature"]
            scores = {}
            
            # Initialize all scores to 0
            for subject in all_subjects:
                scores[subject] = 0
            
            # Only show input fields for applicable subjects
            applicable_count = len(applicable_subjects)
            mid_point = (applicable_count + 1) // 2
            
            left_applicable = applicable_subjects[:mid_point]
            right_applicable = applicable_subjects[mid_point:]
            
            with col1:
                for subject in left_applicable:
                    scores[subject] = st.number_input(
                        f"{subject} Score:",
                        min_value=0,
                        max_value=100,
                        value=0,
                        key=f"single_score_{subject}",
                        help=f"Enter score for {subject} (0-100)"
                    )
            
            with col2:
                for subject in right_applicable:
                    scores[subject] = st.number_input(
                        f"{subject} Score:",
                        min_value=0,
                        max_value=100,
                        value=0,
                        key=f"single_score_{subject}",
                        help=f"Enter score for {subject} (0-100)"
                    )
            
            # Show info about hidden subjects
            non_applicable = [s for s in all_subjects if s not in applicable_subjects]
            if non_applicable:
                st.info(f"**Note:** {', '.join(non_applicable)} scores are automatically set to 0 as they are not part of the {stream} stream.")
            
            # Prediction button
            if st.button("🔮 Predict Field", type="primary", use_container_width=True, key="single_predict"):
                # Validate scores
                errors = validate_scores(stream, scores)
                
                if errors:
                    for subject, error in errors.items():
                        st.error(error)
                else:
                    # Prepare user data
                    user_data = {"High_School_Stream": stream}
                    user_data.update(scores)
                    
                    try:
                        # Preprocess and make prediction
                        user_processed = preprocess_user_input(user_data, encoder)
                        y_pred_encoded = model.predict(user_processed)
                        predicted_field = field_encoder.inverse_transform(y_pred_encoded)[0]
                        
                        # Save to Excel
                        save_to_excel(user_data, predicted_field)
                        
                        # Display result
                        st.markdown(f"""
                        <div class="prediction-result">
                            <h3>🎯 Prediction Result</h3>
                            <p class="success-message">Recommended Field: {predicted_field}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.success("Prediction saved successfully!")
                        
                    except Exception as e:
                        st.error(f"An error occurred during prediction: {str(e)}")
    
    with tab2:
        st.markdown("### 📊 Batch Prediction")
        st.markdown("Upload a CSV file with multiple students' scores for batch prediction.")
        
        # Stream selection for batch prediction
        batch_stream = st.selectbox(
            "High School Stream:",
            ["", "PCM", "PCB", "MEG", "MPG", "HGL"],
            format_func=lambda x: "Select Stream" if x == "" else {
                "PCM": "PCM (Physics, Chemistry, Math)",
                "PCB": "PCB (Physics, Chemistry, Biology)", 
                "MEG": "MEG (Math, Economics, Geography)",
                "MPG": "MPG (Math, Physics, Geography)",
                "HGL": "HGL (History, Geography, Literature)"
            }.get(x, x),
            key="batch_stream"
        )
        
        if batch_stream:
            stream_subjects = {
                "PCM": ["Physics", "Chemistry", "Math"],
                "PCB": ["Physics", "Chemistry", "Biology"],
                "MEG": ["Math", "Economics", "Geography"],
                "MPG": ["Math", "Physics", "Geography"],
                "HGL": ["History", "Geography", "Literature"]
            }
            
            applicable_subjects = stream_subjects.get(batch_stream, [])
            
            # Display information about the expected file format
            st.markdown(f"""
            <div class="batch-info">
                <h4>📋 File Format Requirements for {batch_stream} Stream</h4>
                <p><strong>Required columns:</strong> {', '.join(applicable_subjects)}</p>
                <p><strong>Score range:</strong> 0-100 for each subject</p>
                <p><strong>Note:</strong> Other subjects ({', '.join([s for s in ["Physics", "Math", "Chemistry", "Biology", "Economics", "Geography", "History", "Literature"] if s not in applicable_subjects])}) will be automatically set to 0</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Generate and provide sample file
            col1, col2 = st.columns(2)
            
            with col1:
                # Sample file download
                sample_csv = generate_sample_file(batch_stream)
                st.download_button(
                    label=f"📥 Download Sample CSV for {batch_stream}",
                    data=sample_csv,
                    file_name=f"sample_{batch_stream.lower()}_batch.csv",
                    mime="text/csv"
                )
            
            with col2:
                # File upload
                uploaded_file = st.file_uploader(
                    "Choose CSV file",
                    type=['csv'],
                    help="Upload a CSV file with student scores"
                )
            
            if uploaded_file is not None:
                try:
                    # Read the uploaded file
                    batch_df = pd.read_csv(uploaded_file)
                    
                    st.markdown("#### 📄 Uploaded File Preview")
                    st.dataframe(batch_df.head(), use_container_width=True)
                    
                    # Validate the file
                    validation_errors = validate_batch_file(batch_df, batch_stream)
                    
                    if validation_errors:
                        for error in validation_errors:
                            st.error(error)
                    else:
                        st.success(f"✅ File validation passed! Found {len(batch_df)} records.")
                        
                        # Prepare data for prediction
                        prepared_df = prepare_batch_data(batch_df.copy(), batch_stream)
                        
                        st.markdown("#### 🔧 Prepared Data (with missing subjects set to 0)")
                        st.dataframe(prepared_df.head(), use_container_width=True)
                        
                        # Batch prediction button
                        if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
                            with st.spinner("Processing batch predictions..."):
                                try:
                                    # Make predictions
                                    batch_processed = preprocess_batch_data(prepared_df, encoder)
                                    y_pred_encoded = model.predict(batch_processed)
                                    predicted_fields = field_encoder.inverse_transform(y_pred_encoded)
                                    
                                    # Add predictions to the dataframe
                                    prepared_df['Predicted_Field'] = predicted_fields
                                    
                                    # Save to Excel
                                    save_batch_to_excel(prepared_df)
                                    
                                    # Display results
                                    st.markdown("#### 🎯 Batch Prediction Results")
                                    st.dataframe(prepared_df, use_container_width=True, height=400)
                                    
                                    # Summary statistics
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Total Predictions", len(prepared_df))
                                    with col2:
                                        unique_fields = prepared_df['Predicted_Field'].nunique()
                                        st.metric("Unique Fields", unique_fields)
                                    with col3:
                                        most_common = prepared_df['Predicted_Field'].mode()[0]
                                        st.metric("Most Common Field", most_common)
                                    
                                    # Field distribution
                                    st.markdown("#### 📊 Field Distribution")
                                    field_counts = prepared_df['Predicted_Field'].value_counts()
                                    st.bar_chart(field_counts)
                                    
                                    # Download results
                                    result_csv = prepared_df.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Results as CSV",
                                        data=result_csv,
                                        file_name=f"batch_predictions_{batch_stream.lower()}.csv",
                                        mime="text/csv"
                                    )
                                    
                                    st.success(f"✅ Batch prediction completed! {len(prepared_df)} predictions saved successfully.")
                                    
                                except Exception as e:
                                    st.error(f"An error occurred during batch prediction: {str(e)}")
                
                except Exception as e:
                    st.error(f"Error reading the uploaded file: {str(e)}")
                    st.info("Please ensure your file is a valid CSV with the correct format.")
    
    with tab3:
        st.markdown("### 📈 Prediction History")
        
        if os.path.exists(EXCEL_FILE_PATH):
            try:
                df = pd.read_excel(EXCEL_FILE_PATH)
                
                if not df.empty:
                    # Sidebar filters
                    st.sidebar.markdown("### 🔍 Filters")
                    
                    # Field filter
                    unique_fields = ['All'] + sorted(df['Predicted_Field'].unique().tolist())
                    selected_field = st.sidebar.selectbox("Filter by Predicted Field:", unique_fields)
                    
                    # Stream filter
                    unique_streams = ['All'] + sorted(df['High_School_Stream'].unique().tolist())
                    selected_stream = st.sidebar.selectbox("Filter by High School Stream:", unique_streams)
                    
                    # Apply filters
                    filtered_df = df.copy()
                    if selected_field != 'All':
                        filtered_df = filtered_df[filtered_df['Predicted_Field'] == selected_field]
                    if selected_stream != 'All':
                        filtered_df = filtered_df[filtered_df['High_School_Stream'] == selected_stream]
                    
                    # Display statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Predictions", len(df))
                    with col2:
                        st.metric("Filtered Results", len(filtered_df))
                    with col3:
                        most_common_field = df['Predicted_Field'].mode()[0] if not df.empty else "N/A"
                        st.metric("Most Common Field", most_common_field)
                    
                    # Records per page
                    records_per_page = st.selectbox(
                        "Records per page:",
                        [5, 10, 20, 50, "All"],
                        index=0
                    )
                    
                    # Pagination
                    if records_per_page != "All":
                        total_records = len(filtered_df)
                        total_pages = max(1, (total_records + records_per_page - 1) // records_per_page)
                        
                        # Page selection
                        if total_pages > 1:
                            page = st.selectbox(
                                f"Page (1 to {total_pages}):",
                                range(1, total_pages + 1),
                                index=0
                            )
                            
                            start_idx = (page - 1) * records_per_page
                            end_idx = start_idx + records_per_page
                            display_df = filtered_df.iloc[start_idx:end_idx].copy()
                        else:
                            display_df = filtered_df.copy()
                    else:
                        display_df = filtered_df.copy()
                    
                    # Add row numbers
                    if not display_df.empty:
                        display_df.index = range(1, len(display_df) + 1)
                        
                        # Display table
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            height=400
                        )
                        
                        # Download button
                        csv = display_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Current View as CSV",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                        
                        # Statistics visualization
                        if len(df) > 1:
                            st.markdown("### 📊 Quick Analytics")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Field distribution
                                field_counts = df['Predicted_Field'].value_counts()
                                st.bar_chart(field_counts)
                                st.caption("Distribution of Predicted Fields")
                            
                            with col2:
                                # Stream distribution
                                stream_counts = df['High_School_Stream'].value_counts()
                                st.bar_chart(stream_counts)
                                st.caption("Distribution of High School Streams")
                    
                    else:
                        st.info("No records match the current filters.")
                else:
                    st.info("No predictions have been made yet.")
                    
            except Exception as e:
                st.error(f"Error loading prediction history: {str(e)}")
        else:
            st.info("No prediction history available. Make your first prediction!")

if __name__ == "__main__":
    main()
