import streamlit as st
import pickle
import pandas as pd
import os
import sklearn

st.set_page_config(page_title="IPL Win Predictor", layout="centered")

# App Title
st.title('🏏 IPL Win Predictor')
st.caption(f"Using scikit-learn version: `{sklearn.__version__}`")

# Team and city options
teams = sorted([
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
])

cities = sorted([
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
])

# Load the trained model safely
model_path = 'pipe.pkl'

if not os.path.exists(model_path):
    st.error("❌ 'pipe.pkl' file not found! Please ensure it's in the same folder as this app.")
    st.stop()

try:
    with open(model_path, 'rb') as f:
        pipe = pickle.load(f)
except AttributeError as e:
    st.error("⚠️ Model loading failed due to a version mismatch with scikit-learn.")
    st.code(str(e))
    st.markdown("""
    ### 🔧 Solution:
    - Reinstall scikit-learn to match the model's version (likely `1.6.0`):
      ```bash
      pip install scikit-learn==1.6.0
      ```
    - Or re-train and re-save the model in your current environment.
    """)
    st.stop()
except Exception as e:
    st.error(f"Unexpected error while loading model: {e}")
    st.stop()

# UI for team selection
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the batting team', teams)
with col2:
    bowling_team = st.selectbox('Select the bowling team', teams)

# City and target input
selected_city = st.selectbox('Select host city', cities)
target = st.number_input('🏁 Target Score', min_value=1)

# Match situation input
col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('🏏 Current Score', min_value=0)
with col4:
    overs = st.number_input('⏱ Overs Completed', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets_out = st.number_input('💥 Wickets Out', min_value=0, max_value=10)

# Predict button
if st.button('🔮 Predict Probability'):
    if overs == 0:
        st.warning("⚠️ Overs cannot be 0.")
        st.stop()

    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_remaining = 10 - wickets_out
    crr = score / overs
    rrr = (runs_left * 6 / balls_left) if balls_left > 0 else 0

    # Create input DataFrame
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_remaining],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    try:
        result = pipe.predict_proba(input_df)
        loss_prob = result[0][0]
        win_prob = result[0][1]

        # Display result
        st.markdown("### 📊 Prediction Result")
        st.success(f"✅ {batting_team} Win Probability: **{round(win_prob * 100, 2)}%**")
        st.info(f"📉 {bowling_team} Win Probability: **{round(loss_prob * 100, 2)}%**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
