import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AutoML Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🩺 AutoML Diabetes Prediction System")

st.write(
    "Automatically compare 3 machine learning models "
    "and select the model with the highest test accuracy."
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ AutoML Settings")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload CSV Dataset",
    type=["csv"]
)


# =========================================================
# STOP IF NO FILE
# =========================================================

if uploaded_file is None:

    st.info(
        "👈 Please upload your CSV dataset from the sidebar."
    )

    st.stop()


# =========================================================
# READ CSV
# =========================================================

try:

    data = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(
        "❌ Could not read the uploaded CSV file."
    )

    st.error(str(e))

    st.stop()


# =========================================================
# CHECK DATA
# =========================================================

if data is None:

    st.error(
        "❌ Dataset could not be loaded."
    )

    st.stop()


if data.empty:

    st.error(
        "❌ The uploaded CSV file is empty."
    )

    st.stop()


# =========================================================
# DATASET INFORMATION
# =========================================================

st.header("📊 Dataset Information")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Rows",
        data.shape[0]
    )

with col2:

    st.metric(
        "Columns",
        data.shape[1]
    )

with col3:

    st.metric(
        "Missing Values",
        int(data.isnull().sum().sum())
    )

with col4:

    st.metric(
        "Features",
        data.shape[1] - 1
    )


with st.expander("👀 View Dataset"):

    st.dataframe(
        data,
        use_container_width=True
    )


# =========================================================
# TARGET COLUMN
# =========================================================

st.divider()

st.header("🎯 Select Target Column")

target_column = st.selectbox(
    "Choose the column to predict:",
    data.columns
)

st.success(
    f"Selected Target Column: **{target_column}**"
)


# =========================================================
# X AND Y
# =========================================================

X = data.drop(
    columns=[target_column]
).copy()

y = data[target_column].copy()


# Remove rows with missing target

valid_rows = y.notna()

X = X.loc[valid_rows].copy()

y = y.loc[valid_rows].copy()


# =========================================================
# TARGET CHECK
# =========================================================

if y.nunique() < 2:

    st.error(
        "❌ Target column must contain at least 2 classes."
    )

    st.stop()


# =========================================================
# TARGET ENCODING
# =========================================================

target_classes = sorted(
    y.astype(str).unique()
)

target_mapping = {
    value: index
    for index, value in enumerate(target_classes)
}

y_encoded = (
    y.astype(str)
    .map(target_mapping)
    .values
)


# =========================================================
# FEATURE ENCODING
# =========================================================

X_encoded = pd.get_dummies(
    X,
    drop_first=True
)

X_encoded = X_encoded.apply(
    pd.to_numeric,
    errors="coerce"
)

X_encoded = X_encoded.dropna(
    axis=1,
    how="all"
)


if X_encoded.shape[1] == 0:

    st.error(
        "❌ No usable feature columns found."
    )

    st.stop()


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

st.divider()

st.header("✂️ Train / Test Split")

test_percentage = st.slider(
    "Testing data percentage",
    10,
    40,
    20,
    5
)


try:

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y_encoded,
        test_size=test_percentage / 100,
        random_state=42,
        stratify=y_encoded
    )

except ValueError:

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y_encoded,
        test_size=test_percentage / 100,
        random_state=42
    )


col1, col2 = st.columns(2)

with col1:

    st.metric(
        "📚 Training Data",
        len(X_train)
    )

with col2:

    st.metric(
        "🧪 Testing Data",
        len(X_test)
    )


# =========================================================
# MODELS
# =========================================================

models = {

    "Logistic Regression": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]),

    "Decision Tree": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            DecisionTreeClassifier(
                random_state=42
            )
        )
    ]),

    "Random Forest": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )
    ])
}


# =========================================================
# TRAIN MODELS
# =========================================================

st.divider()

st.header("🤖 Model Training")

results = {}

trained_models = {}


for model_name, model in models.items():

    with st.spinner(
        f"Training {model_name}..."
    ):

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        results[model_name] = accuracy

        trained_models[model_name] = model


st.success(
    "✅ All 3 models trained successfully!"
)


# =========================================================
# MODEL ACCURACY
# =========================================================

st.divider()

st.header("📈 Model Accuracy Comparison")

accuracy_table = pd.DataFrame({

    "Model": list(results.keys()),

    "Test Accuracy": [
        f"{value * 100:.2f}%"
        for value in results.values()
    ]

})

st.dataframe(
    accuracy_table,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# ACCURACY CHART
# =========================================================

chart_data = pd.DataFrame({

    "Accuracy (%)": [
        value * 100
        for value in results.values()
    ]

}, index=results.keys())

st.bar_chart(
    chart_data
)


# =========================================================
# BEST MODEL
# =========================================================

best_model_name = max(
    results,
    key=results.get
)

best_model = trained_models[
    best_model_name
]

best_accuracy = results[
    best_model_name
]


st.divider()

st.header("🏆 Best Model")

st.success(
    f"🏆 Best Model: **{best_model_name}**"
)

st.metric(
    "Best Test Accuracy",
    f"{best_accuracy * 100:.2f}%"
)


# =========================================================
# WHY BEST MODEL
# =========================================================

with st.expander(
    "🧠 Why was this model selected?"
):

    st.write("""
    All three models were trained using the training data.

    Each model was then tested using unseen test data.

    The test accuracy of every model was calculated.

    The model with the highest test accuracy was automatically
    selected as the best model.
    """)


# =========================================================
# PATIENT INPUT
# =========================================================

st.sidebar.divider()

st.sidebar.header(
    "🩺 Patient Information"
)

st.sidebar.write(
    f"Selected Model: **{best_model_name}**"
)


input_values = {}


for column in X_encoded.columns:

    if column in X.columns:

        if pd.api.types.is_numeric_dtype(
            X[column]
        ):

            default_value = X[column].median()

        else:

            default_value = 0

    else:

        default_value = 0


    if pd.isna(default_value):

        default_value = 0


    input_values[column] = st.sidebar.number_input(
        column,
        value=float(default_value)
    )


# =========================================================
# INPUT DATAFRAME
# =========================================================

input_data = pd.DataFrame(
    [input_values]
)

input_data = input_data.reindex(
    columns=X_encoded.columns,
    fill_value=0
)


# =========================================================
# PREDICTION
# =========================================================

st.divider()

st.header("🔮 Patient Prediction")

st.write(
    "Enter patient information in the sidebar "
    "and click the prediction button."
)


if st.button(
    "🔮 Predict Diabetes",
    type="primary",
    use_container_width=True
):

    prediction = best_model.predict(
        input_data
    )[0]


    probabilities = best_model.predict_proba(
        input_data
    )[0]


    predicted_class = target_classes[
        prediction
    ]


    prediction_probability = (
        probabilities[prediction] * 100
    )


    # =====================================================
    # RESULT
    # =====================================================

    st.divider()

    st.header("🩺 Prediction Result")


    result_text = str(
        predicted_class
    ).lower().strip()


    if result_text in [
        "1",
        "yes",
        "true",
        "diabetes",
        "diabetic",
        "positive"
    ]:

        st.error(
            "⚠️ Patient is predicted to have DIABETES"
        )

    else:

        st.success(
            "✅ Patient is predicted to NOT HAVE DIABETES"
        )


    # =====================================================
    # PROBABILITY
    # =====================================================

    st.subheader(
        "🎯 Prediction Probability"
    )

    st.metric(
        "Predicted Class Probability",
        f"{prediction_probability:.2f}%"
    )

    st.progress(
        min(
            int(prediction_probability),
            100
        )
    )


    # =====================================================
    # MODEL USED
    # =====================================================

    st.info(
        f"🤖 Model Used: **{best_model_name}**"
    )


    # =====================================================
    # ALL PROBABILITIES
    # =====================================================

    st.subheader(
        "📊 Probability for Each Outcome"
    )

    probability_table = pd.DataFrame({

        "Outcome": target_classes,

        "Probability": [
            f"{value * 100:.2f}%"
            for value in probabilities
        ]

    })


    st.dataframe(
        probability_table,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # EXPLANATION
    # =====================================================

    st.divider()

    st.subheader(
        "🧠 How was this prediction made?"
    )

    st.write("""
    The selected model learned patterns from the training
    dataset.

    The patient's input values are given to the trained model.

    The model calculates the probability of each possible
    outcome.

    The outcome with the highest probability becomes the
    final prediction.
    """)


    # =====================================================
    # PATIENT VALUES
    # =====================================================

    with st.expander(
        "🔍 View Patient Values"
    ):

        st.dataframe(
            input_data,
            use_container_width=True
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🩺 AutoML Diabetes Prediction System | "
    "Machine Learning Project"
)