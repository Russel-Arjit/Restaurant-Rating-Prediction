# ============================================================
# RESTAURANT RATING PREDICTOR
# ============================================================

import streamlit as st
import pandas as pd
import joblib


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Restaurant Rating Predictor",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():
    trained_model = joblib.load("model/restaurant_rating_model.pkl")
    saved_features = joblib.load("model/model_features.pkl")
    return trained_model, saved_features


try:
    model, model_features = load_model()
    model_loaded = True

except Exception as e:
    model_loaded = False
    st.error(f"Error loading model: {e}")
    st.stop()


# ============================================================
# 3. CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* ---------------------------------------------------------
   MAIN APP
--------------------------------------------------------- */

.stApp {
    background:
        radial-gradient(circle at top right,
        rgba(37, 99, 235, 0.08),
        transparent 30%),
        linear-gradient(135deg, #050914 0%, #08111f 100%);

    color: #f8fafc;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}


/* ---------------------------------------------------------
   SIDEBAR
--------------------------------------------------------- */

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #070c17 0%,
            #091426 100%
        );

    border-right: 1px solid rgba(124, 58, 237, 0.25);
}

[data-testid="stSidebar"] * {
    color: #f8fafc;
}

.sidebar-brand {
    text-align: center;
    padding: 20px 5px 30px 5px;
}

.sidebar-icon {
    font-size: 55px;
}

.sidebar-title {
    font-size: 23px;
    font-weight: 800;
    margin-top: 8px;
}

.sidebar-purple {
    color: #8b5cf6;
}

.sidebar-card {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 15px;
    padding: 18px;
    margin-top: 20px;
    margin-bottom: 20px;
}


/* ---------------------------------------------------------
   HERO
--------------------------------------------------------- */

.hero-card {
    padding: 42px;
    border-radius: 24px;

    background:
        radial-gradient(
            circle at 85% 20%,
            rgba(14, 165, 233, 0.25),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            rgba(76, 29, 149, 0.85),
            rgba(15, 23, 42, 0.95)
        );

    border: 1px solid rgba(99, 102, 241, 0.60);

    box-shadow:
        0px 20px 60px rgba(0, 0, 0, 0.35);

    margin-bottom: 22px;
}

.hero-title {
    font-size: 48px;
    font-weight: 800;
    line-height: 1.12;
    color: #ffffff;
    margin-bottom: 16px;
}

.hero-subtitle {
    font-size: 18px;
    line-height: 1.7;
    color: #dbeafe;
    max-width: 650px;
}


/* ---------------------------------------------------------
   MODEL STATUS
--------------------------------------------------------- */

.model-status {
    padding: 17px 20px;

    background: rgba(16, 185, 129, 0.10);

    border:
        1px solid rgba(52, 211, 153, 0.35);

    border-radius: 14px;

    color: #6ee7b7;

    font-weight: 600;

    margin-bottom: 28px;
}


/* ---------------------------------------------------------
   HEADINGS
--------------------------------------------------------- */

.section-title {
    font-size: 27px;
    font-weight: 750;
    color: #ffffff;

    margin-top: 12px;
    margin-bottom: 18px;
}

.result-title {
    font-size: 28px;
    font-weight: 800;
    color: #a78bfa;

    margin-bottom: 10px;
}


/* ---------------------------------------------------------
   STREAMLIT INPUTS
--------------------------------------------------------- */

label {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}

[data-baseweb="select"] > div {
    background-color: #111827;
    border-radius: 10px;
}

[data-testid="stNumberInput"] input {
    background-color: #111827;
}


/* ---------------------------------------------------------
   BUTTON
--------------------------------------------------------- */

.stButton > button {
    width: 100%;

    border: none;

    border-radius: 12px;

    padding: 0.85rem;

    font-size: 18px;
    font-weight: 750;

    color: white;

    background:
        linear-gradient(
            90deg,
            #6d28d9 0%,
            #2563eb 100%
        );

    transition: all 0.25s ease;
}

.stButton > button:hover {

    color: white;

    border: none;

    transform: translateY(-2px);

    box-shadow:
        0 10px 30px
        rgba(79, 70, 229, 0.40);
}


/* ---------------------------------------------------------
   RESULT CARD
--------------------------------------------------------- */

.rating-card {

    background:
        linear-gradient(
            135deg,
            rgba(15, 23, 42, 0.95),
            rgba(30, 27, 75, 0.55)
        );

    border:
        1px solid rgba(139, 92, 246, 0.35);

    border-radius: 20px;

    padding: 30px;

    margin-top: 20px;

    text-align: center;
}

.rating-number {

    font-size: 60px;

    font-weight: 850;

    color: #4ade80;

    margin-bottom: 0px;
}

.rating-outof {

    color: #94a3b8;

    font-size: 20px;
}

.rating-stars {

    font-size: 36px;

    margin-top: 10px;

    margin-bottom: 15px;
}


/* ---------------------------------------------------------
   FOOTER
--------------------------------------------------------- */

.footer {

    text-align: center;

    color: #64748b;

    font-size: 14px;

    padding-top: 35px;

    padding-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 4. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
<div class="sidebar-brand">
<div class="sidebar-icon">🍽️</div>
<div class="sidebar-title">RESTAURANT</div>
<div class="sidebar-title sidebar-purple">PREDICTOR</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("### 🏠 Predict Rating")

    st.divider()

    st.markdown("### 📈 About Model")

    st.write(
        "This application uses a Machine Learning regression "
        "model to estimate the expected aggregate rating of "
        "a restaurant."
    )

    st.markdown("### 📊 Model Performance")

    st.metric(
        label="Tuned Random Forest R²",
        value="0.593"
    )

    st.metric(
        label="RMSE",
        value="0.355"
    )

    st.metric(
        label="MAE",
        value="0.256"
    )

    st.divider()

    st.caption("Machine Learning Portfolio Project")
    st.caption(
        "Built with Python • Scikit-learn • Streamlit"
    )


# ============================================================
# 5. HERO SECTION
# ============================================================

st.markdown("""
<div class="hero-card">
<div class="hero-title">
🍽️ Restaurant<br>Rating Predictor
</div>

<div class="hero-subtitle">
Predict a restaurant's expected rating using a Machine
Learning model trained on restaurant characteristics,
location information and customer engagement data.
</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 6. MODEL STATUS
# ============================================================

if model_loaded:

    st.markdown("""
<div class="model-status">
✓ &nbsp; Machine Learning model loaded successfully!
</div>
""", unsafe_allow_html=True)


# ============================================================
# 7. RESTAURANT INPUT FORM
# ============================================================

st.markdown(
    '<div class="section-title">📋 Enter Restaurant Details</div>',
    unsafe_allow_html=True
)


col1, col2 = st.columns(2, gap="large")


# ---------------- LEFT COLUMN ----------------

with col1:

    city = st.selectbox(
        "📍 City",
        [
            "New Delhi",
            "Gurgaon",
            "Noida",
            "Faridabad"
        ]
    )

    average_cost = st.number_input(
        "💰 Average Cost for Two (₹)",
        min_value=0,
        max_value=100000,
        value=800,
        step=100
    )

    price_range = st.selectbox(
        "🏷️ Price Range",
        [1, 2, 3, 4],
        index=1
    )

    votes = st.number_input(
        "👍 Number of Votes",
        min_value=0,
        max_value=50000,
        value=500,
        step=10
    )


# ---------------- RIGHT COLUMN ----------------

with col2:

    table_booking = st.selectbox(
        "📅 Table Booking",
        ["Yes", "No"]
    )

    online_delivery = st.selectbox(
        "🛵 Online Delivery",
        ["Yes", "No"]
    )

    delivering_now = st.selectbox(
        "📦 Currently Delivering",
        ["No", "Yes"]
    )

    switch_order = st.selectbox(
        "🛒 Switch to Order Menu",
        ["No", "Yes"]
    )


st.write("")
st.write("")


# ============================================================
# 8. CITY LOCATION INFORMATION
# ============================================================

city_details = {

    "New Delhi": {
        "country": 1,
        "longitude": 77.2090,
        "latitude": 28.6139
    },

    "Gurgaon": {
        "country": 1,
        "longitude": 77.0266,
        "latitude": 28.4595
    },

    "Noida": {
        "country": 1,
        "longitude": 77.3910,
        "latitude": 28.5355
    },

    "Faridabad": {
        "country": 1,
        "longitude": 77.3178,
        "latitude": 28.4089
    }
}


location = city_details[city]


# ============================================================
# 9. CREATE MODEL INPUT
# ============================================================

input_data = pd.DataFrame(
    0,
    index=[0],
    columns=model_features,
    dtype=float
)


# ---------------- NUMERICAL FEATURES ----------------

if "Country Code" in input_data.columns:
    input_data.loc[0, "Country Code"] = location["country"]

if "Longitude" in input_data.columns:
    input_data.loc[0, "Longitude"] = location["longitude"]

if "Latitude" in input_data.columns:
    input_data.loc[0, "Latitude"] = location["latitude"]

if "Average Cost for two" in input_data.columns:
    input_data.loc[0, "Average Cost for two"] = average_cost

if "Price range" in input_data.columns:
    input_data.loc[0, "Price range"] = price_range

if "Votes" in input_data.columns:
    input_data.loc[0, "Votes"] = votes


# ---------------- BINARY FEATURES ----------------

if "Has Table booking" in input_data.columns:
    input_data.loc[0, "Has Table booking"] = (
        1 if table_booking == "Yes" else 0
    )

if "Has Online delivery" in input_data.columns:
    input_data.loc[0, "Has Online delivery"] = (
        1 if online_delivery == "Yes" else 0
    )

if "Is delivering now" in input_data.columns:
    input_data.loc[0, "Is delivering now"] = (
        1 if delivering_now == "Yes" else 0
    )

if "Switch to order menu" in input_data.columns:
    input_data.loc[0, "Switch to order menu"] = (
        1 if switch_order == "Yes" else 0
    )


# ---------------- CITY ONE-HOT ENCODING ----------------

city_column = f"City_{city}"

if city_column in input_data.columns:
    input_data.loc[0, city_column] = 1


# ============================================================
# 10. PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "✨ Predict Restaurant Rating",
    use_container_width=True
)


# ============================================================
# 11. PREDICTION RESULT
# ============================================================

if predict_button:

    try:

        prediction = model.predict(input_data)[0]

        # Rating must remain between 0 and 5
        prediction = max(
            0.0,
            min(5.0, float(prediction))
        )


        # ----------------------------------------
        # Rating Category
        # ----------------------------------------

        if prediction >= 4.5:

            category = "Excellent 🌟"
            message = (
                "This restaurant is predicted to have "
                "an excellent customer rating."
            )

        elif prediction >= 4.0:

            category = "Very Good 😍"
            message = (
                "This restaurant is predicted to receive "
                "very positive customer feedback."
            )

        elif prediction >= 3.5:

            category = "Good 🙂"
            message = (
                "This restaurant is likely to have "
                "a good overall customer rating."
            )

        elif prediction >= 2.5:

            category = "Average 😐"
            message = (
                "This restaurant is predicted to have "
                "an average customer rating."
            )

        else:

            category = "Poor 📉"
            message = (
                "This restaurant may receive a relatively "
                "low customer rating."
            )


        # ----------------------------------------
        # Stars
        # ----------------------------------------

        rounded_stars = int(round(prediction))

        stars = (
            "⭐" * rounded_stars
            + "☆" * (5 - rounded_stars)
        )


        # ----------------------------------------
        # Result Heading
        # ----------------------------------------

        st.markdown(
            '<div class="result-title">🎯 Prediction Result</div>',
            unsafe_allow_html=True
        )


        # ----------------------------------------
        # Result Card
        # ----------------------------------------

        st.markdown(
            f"""
<div class="rating-card">

<div style="
font-size:18px;
color:#cbd5e1;
margin-bottom:10px;
">
Predicted Restaurant Rating
</div>

<div class="rating-number">
{prediction:.2f}
</div>

<div class="rating-outof">
/ 5.0
</div>

<div class="rating-stars">
{stars}
</div>

<div style="
font-size:26px;
font-weight:750;
color:#ffffff;
margin-bottom:10px;
">
{category}
</div>

<div style="
color:#94a3b8;
font-size:16px;
">
{message}
</div>

</div>
""",
            unsafe_allow_html=True
        )

        # ============================================================
        # PREDICTION FACTORS
        # ============================================================

        st.markdown("### 🧠 What Factors Influence This Prediction?")

        st.caption(
            "The model considers several restaurant characteristics "
            "when estimating the rating."
        )

        factor_col1, factor_col2 = st.columns(2)

        with factor_col1:

            st.info(
                f"👍 **Customer Engagement**\n\n"
                f"This restaurant has **{votes:,} votes**. "
                f"Votes were the most important feature in the trained model."
            )

            st.info(
                f"💰 **Pricing Information**\n\n"
                f"Average cost for two: **₹{average_cost:,}**\n\n"
                f"Price range: **{price_range}/4**"
            )

        with factor_col2:

            st.info(
                f"📍 **Location Information**\n\n"
                f"The selected city is **{city}**. "
                f"Location-related features such as longitude and latitude "
                f"are used by the model."
            )

            delivery_text = (
                "available"
                if online_delivery == "Yes"
                else "not available"
            )

            booking_text = (
                "available"
                if table_booking == "Yes"
                else "not available"
            )

            st.info(
                f"🛵 **Restaurant Services**\n\n"
                f"Online delivery: **{delivery_text}**\n\n"
                f"Table booking: **{booking_text}**"
            )

        st.caption(
            "ℹ️ These are important model inputs, not proof that any "
            "individual factor caused the predicted rating."
        )

        # ----------------------------------------
        # Prediction Details
        # ----------------------------------------

        st.write("")

        with st.expander("🔎 View Prediction Details"):

            details = pd.DataFrame({

                "Restaurant Detail": [
                    "City",
                    "Average Cost for Two",
                    "Price Range",
                    "Votes",
                    "Table Booking",
                    "Online Delivery",
                    "Currently Delivering"
                ],

                "Selected Value": [
                    city,
                    f"₹{average_cost}",
                    price_range,
                    votes,
                    table_booking,
                    online_delivery,
                    delivering_now
                ]

            })

            st.dataframe(
                details,
                use_container_width=True,
                hide_index=True
            )


        st.info(
            "ℹ️ This prediction is based on patterns learned "
            "from historical restaurant data. Actual ratings "
            "may vary in real-world scenarios."
        )


    except Exception as e:

        st.error(
            "Prediction could not be generated."
        )

        st.exception(e)


# ============================================================
# 12. ABOUT / PROJECT INFORMATION
# ============================================================

st.write("")
st.divider()

with st.expander("🤖 About the Machine Learning Model"):

    st.markdown("""
### Tuned Random Forest Regressor

The application uses a **Random Forest regression model**
to predict restaurant aggregate ratings.

The project includes:

- Data cleaning and preprocessing
- Exploratory Data Analysis
- Feature engineering
- One-hot encoding
- Linear Regression
- Decision Tree Regression
- Random Forest Regression
- Model comparison
- Cross-validation
- Hyperparameter tuning
- Feature importance analysis
- Model persistence
- Interactive prediction interface

The tuned Random Forest produced the strongest performance
among the evaluated models.
""")


# ============================================================
# 13. FOOTER
# ============================================================

st.markdown("""
<div class="footer">

Restaurant Rating Prediction • Machine Learning Portfolio Project

<br>

Python • Pandas • Scikit-learn • Streamlit

</div>
""", unsafe_allow_html=True)