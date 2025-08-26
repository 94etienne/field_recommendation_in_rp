import streamlit as st
import pandas as pd
from joblib import load
import os

# Set page configuration
st.set_page_config(
    page_title="Incoming Student Field Prediction",
    page_icon="🎓",
    layout="wide"
)

# Load the saved model and encoders
@st.cache_resource
def load_model():
    try:
        model = load("../model/decision_tree_model.pkl")
        encoder = load("../model/one_hot_encoder.pkl")
        field_encoder = load("../model/label_encoder.pkl")
        return model, encoder, field_encoder
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, encoder, field_encoder = load_model()

# Define the path for the Excel file
EXCEL_FILE_PATH = "predictions_log.xlsx"

def preprocess_user_input(user_data, encoder):
    """
    Preprocess user input to match the model's input format.
    """
    # Convert user input into a DataFrame
    user_df = pd.DataFrame([user_data])

    # One-hot encode High_School_Stream
    stream_encoded = encoder.transform(user_df[["High_School_Stream"]])
    stream_encoded_df = pd.DataFrame(stream_encoded, columns=encoder.get_feature_names_out(["High_School_Stream"]))

    # Combine encoded features with subject scores
    user_processed = pd.concat([stream_encoded_df, user_df.iloc[:, 1:]], axis=1)

    return user_processed, user_df

def save_to_excel(user_data, predicted_field):
    """
    Save the input data and predicted field to an Excel file.
    """
    # Create a DataFrame with the input data and prediction
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

    # Check if the Excel file already exists
    if os.path.exists(EXCEL_FILE_PATH):
        # Load the existing Excel file
        existing_df = pd.read_excel(EXCEL_FILE_PATH)
        # Append the new data
        updated_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        # Create a new Excel file
        updated_df = df

    # Save the updated DataFrame to Excel
    updated_df.to_excel(EXCEL_FILE_PATH, index=False)

def get_predictions(page=1, per_page=5, predicted_field=None):
    """
    Get paginated predictions from the Excel file.
    """
    if os.path.exists(EXCEL_FILE_PATH):
        # Load the Excel file
        df = pd.read_excel(EXCEL_FILE_PATH)

        # Apply the predicted field filter if provided
        if predicted_field:
            df = df[df["Predicted_Field"] == predicted_field]

        # Calculate total number of records and pages
        total_records = len(df)
        total_pages = (total_records + per_page - 1) // per_page

        # Paginate the data
        start = (page - 1) * per_page
        end = start + per_page
        paginated_data = df.iloc[start:end]

        return paginated_data, total_records, total_pages
    else:
        return pd.DataFrame(), 0, 0

def main():
    st.title("Incoming Student Field Recommendation in Rwanda Polytechnic")
    
    # Initialize session state
    if 'current_stream' not in st.session_state:
        st.session_state.current_stream = ""
    if 'show_history' not in st.session_state:
        st.session_state.show_history = False
    if 'page' not in st.session_state:
        st.session_state.page = 1
    if 'per_page' not in st.session_state:
        st.session_state.per_page = 5
    if 'filter_field' not in st.session_state:
        st.session_state.filter_field = None

    # Define subjects for each stream
    stream_subjects = {
        "PCM": ["Physics", "Chemistry", "Math"],
        "PCB": ["Physics", "Chemistry", "Biology"],
        "MEG": ["Math", "Economics", "Geography"],
        "MPG": ["Math", "Physics", "Geography"],
        "HGL": ["History", "Geography", "Literature"]
    }

    # Stream descriptions for better user understanding
    stream_descriptions = {
        "PCM": "Physics, Chemistry, Mathematics",
        "PCB": "Physics, Chemistry, Biology",
        "MEG": "Mathematics, Economics, Geography",
        "MPG": "Mathematics, Physics, Geography",
        "HGL": "History, Geography, Literature"
    }

    # Create form for prediction
    with st.form("prediction_form"):
        st.subheader("Student Information")
        
        # Stream selection
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_stream = st.selectbox(
                "High School Stream:",
                ["", "PCM", "PCB", "MEG", "MPG", "HGL"],
                format_func=lambda x: "Select Stream" if x == "" else f"{x} ({stream_descriptions[x]})",
                key="stream_select"
            )
        
        with col2:
            if selected_stream:
                st.info(f"**Selected Stream:** {selected_stream} - {stream_descriptions[selected_stream]}")
        
        # Display relevant subjects based on selected stream
        if selected_stream:
            st.subheader("Enter Subject Scores (0-100)")
            applicable_subjects = stream_subjects.get(selected_stream, [])
            
            # Create columns for better layout of subjects
            num_subjects = len(applicable_subjects)
            if num_subjects <= 2:
                cols = st.columns(num_subjects)
            else:
                cols = st.columns(2)
            
            # Display input fields only for applicable subjects
            for i, subject in enumerate(applicable_subjects):
                if num_subjects <= 2:
                    col_index = i
                else:
                    col_index = i % 2
                
                with cols[col_index]:
                    st.number_input(
                        f"{subject} score:",
                        min_value=0,
                        max_value=100,
                        value=0,
                        key=f"{subject.lower()}_score",
                        help=f"Enter your {subject} score (0-100)"
                    )
            
            # Show message about non-applicable subjects being set to 0
            all_subjects = ["Physics", "Math", "Chemistry", "Biology", "Economics", "Geography", "History", "Literature"]
            non_applicable = [subj for subj in all_subjects if subj not in applicable_subjects]
            
            if non_applicable:
                st.info(f"💡 **Note:** The following subjects are automatically set to 0 as they are not part of the {selected_stream} stream: {', '.join(non_applicable)}")
        
        submitted = st.form_submit_button("🎯 Predict Field of Study", type="primary")
        
        if submitted:
            if not selected_stream:
                st.error("Please select a High School Stream.")
            else:
                # Initialize user data with all subjects set to 0
                user_data = {
                    "High_School_Stream": selected_stream,
                    "Physics": 0,
                    "Math": 0,
                    "Chemistry": 0,
                    "Biology": 0,
                    "Economics": 0,
                    "Geography": 0,
                    "History": 0,
                    "Literature": 0
                }
                
                # Update with actual values only for applicable subjects
                applicable_subjects = stream_subjects.get(selected_stream, [])
                for subject in applicable_subjects:
                    user_data[subject] = st.session_state.get(f"{subject.lower()}_score", 0)
                
                # Validate marks for applicable subjects
                valid = True
                validation_errors = []
                
                for subject in applicable_subjects:
                    score = user_data[subject]
                    if score == 0:
                        validation_errors.append(f"{subject} score cannot be 0")
                        valid = False
                    elif score < 0 or score > 100:
                        validation_errors.append(f"{subject} score must be between 0 and 100")
                        valid = False
                
                if validation_errors:
                    for error in validation_errors:
                        st.error(error)
                
                if valid and model is not None:
                    try:
                        # Preprocess and predict
                        user_processed, user_df = preprocess_user_input(user_data, encoder)
                        y_pred_encoded = model.predict(user_processed)
                        y_pred = field_encoder.inverse_transform(y_pred_encoded)
                        
                        # Save to Excel
                        save_to_excel(user_data, y_pred[0])
                        
                        # Display result with success styling
                        st.balloons()
                        st.success("### 🎓 Field Recommendation")
                        
                        # Create a highlighted result box
                        st.markdown(f"""
                        <div style="
                            background-color: #e8f5e8;
                            padding: 20px;
                            border-radius: 10px;
                            border-left: 5px solid #28a745;
                            margin: 10px 0;
                        ">
                            <h3 style="color: #28a745; margin-top: 0;">
                                🎯 Recommended Field of Study: {y_pred[0]}
                            </h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show the submitted data in an expandable section
                        with st.expander("📋 View Submitted Data"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Stream:**", selected_stream)
                                for i, subject in enumerate(applicable_subjects):
                                    if i < len(applicable_subjects)//2 + len(applicable_subjects)%2:
                                        st.write(f"**{subject}:**", user_data[subject])
                            with col2:
                                for i, subject in enumerate(applicable_subjects):
                                    if i >= len(applicable_subjects)//2 + len(applicable_subjects)%2:
                                        st.write(f"**{subject}:**", user_data[subject])
                        
                    except Exception as e:
                        st.error(f"Error making prediction: {e}")
                elif model is None:
                    st.error("Model not loaded. Please check if the model files exist.")
    
    # Prediction History Section
    st.markdown("---")
    st.header("📊 Prediction History")
    
    # Toggle button for showing/hiding history
    if st.button("📖 Show Prediction History" if not st.session_state.show_history else "📕 Hide Prediction History"):
        st.session_state.show_history = not st.session_state.show_history
        st.rerun()
    
    if st.session_state.show_history:
        # Get unique fields for filter
        if os.path.exists(EXCEL_FILE_PATH):
            try:
                df_all = pd.read_excel(EXCEL_FILE_PATH)
                unique_fields = df_all["Predicted_Field"].unique().tolist()
            except:
                unique_fields = []
                st.warning("Could not read prediction history file.")
        else:
            unique_fields = []
            st.info("No prediction history available yet.")
        
        if unique_fields:
            # Filter and pagination controls
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                filter_field = st.selectbox(
                    "Filter by Predicted Field:",
                    [None] + sorted(unique_fields),
                    format_func=lambda x: "All Fields" if x is None else x,
                    key="filter_select"
                )
                st.session_state.filter_field = filter_field
            
            with col2:
                per_page_options = [5, 10, 20, 50]
                per_page = st.selectbox(
                    "Records per page:",
                    per_page_options,
                    index=per_page_options.index(st.session_state.per_page) if st.session_state.per_page in per_page_options else 0,
                    key="per_page_select"
                )
                st.session_state.per_page = per_page
            
            with col3:
                st.write("")  # Spacer
                if st.button("🔄 Refresh Data"):
                    st.rerun()
            
            # Get paginated data
            data, total_records, total_pages = get_predictions(
                st.session_state.page, 
                per_page, 
                st.session_state.filter_field
            )
            
            # Display data
            if not data.empty:
                st.write(f"**Total records:** {total_records}")
                
                # Add row numbers
                data_display = data.copy()
                start_num = (st.session_state.page - 1) * per_page + 1
                data_display.insert(0, "No", range(start_num, start_num + len(data)))
                
                # Format the display
                styled_data = data_display.style.format({
                    'Physics': '{:.0f}',
                    'Math': '{:.0f}',
                    'Chemistry': '{:.0f}',
                    'Biology': '{:.0f}',
                    'Economics': '{:.0f}',
                    'Geography': '{:.0f}',
                    'History': '{:.0f}',
                    'Literature': '{:.0f}'
                })
                
                st.dataframe(styled_data, use_container_width=True, height=400)
                
                # Pagination controls
                if total_pages > 1:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col1:
                        if st.button("⬅️ Previous") and st.session_state.page > 1:
                            st.session_state.page -= 1
                            st.rerun()
                    
                    with col2:
                        st.write(f"**Page {st.session_state.page} of {total_pages}**")
                    
                    with col3:
                        if st.button("Next ➡️") and st.session_state.page < total_pages:
                            st.session_state.page += 1
                            st.rerun()
            else:
                st.info("No records found with the selected filter.")
        
        # Download button for the entire history
        if os.path.exists(EXCEL_FILE_PATH):
            with open(EXCEL_FILE_PATH, "rb") as file:
                st.download_button(
                    label="📥 Download Full Prediction History (Excel)",
                    data=file,
                    file_name="prediction_history.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main()