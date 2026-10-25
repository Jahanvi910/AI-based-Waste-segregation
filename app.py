
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="EcoSort AI | Smart Waste Segregation",
    page_icon="♻️",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
HISTORY_FILE = ROOT / "scan_history.json"
FEEDBACK_FILE = ROOT / "user_feedback.json"
POINTS_FILE = ROOT / "eco_points.json"

# ---------------- WEBSITE DESIGN ----------------
st.markdown(
    """
    <style>
    /* Main background and default text */
    .stApp {
        background-color: #F5FAF5 !important;
        color: #24352B !important;
        font-family: "Times New Roman", Times, serif !important;
    }

    /* Times New Roman across Streamlit components */
    html, body, [class*="st-"], [data-testid="stApp"],
    [data-testid="stSidebar"], [data-testid="stHeader"],
    button, input, textarea, select, label,
    p, span, li, div, h1, h2, h3, h4, h5, h6 {
        font-family: "Times New Roman", Times, serif !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #176B3A !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #E3F2E8 !important;
    }

    [data-testid="stSidebar"] * {
        color: #24352B;
    }

    /* General buttons */
    div.stButton > button {
        border-radius: 9px;
        font-family: "Times New Roman", Times, serif !important;
    }

    div.stButton > button[kind="primary"] {
        background-color: #218C52 !important;
        border-color: #218C52 !important;
        color: #FFFFFF !important;
        border-radius: 10px;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"] * {
        color: #FFFFFF !important;
    }

    /* File uploader: force a white background */
    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] > div,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzone"] {
        background-color: #FFFFFF !important;
        color: #24352B !important;
        border-color: #6CA783 !important;
        border-radius: 12px;
        font-family: "Times New Roman", Times, serif !important;
    }

    /* Upload instructions and text */
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] div,
    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderDropzoneInstructions"] * {
        color: #24352B !important;
        font-family: "Times New Roman", Times, serif !important;
    }

    /* Browse files button */
    [data-testid="stFileUploader"] button {
        background-color: #E3F2E8 !important;
        color: #174D32 !important;
        border: 1px solid #6CA783 !important;
        border-radius: 8px !important;
        font-family: "Times New Roman", Times, serif !important;
    }

    [data-testid="stFileUploader"] button * {
        color: #174D32 !important;
        fill: #174D32 !important;
    }

    /* Upload dropzone border */
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
        background: #FFFFFF !important;
        border: 1px dashed #6CA783 !important;
    }

    /* Form inputs */
    input, textarea {
        background-color: #FFFFFF !important;
        color: #24352B !important;
        font-family: "Times New Roman", Times, serif !important;
    }

    /* Cards */
    .eco-card {
        padding: 1rem;
        border-radius: 14px;
        background-color: #FFFFFF;
        border: 1px solid #D9EADC;
        margin-bottom: 0.7rem;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 12px;
        border-radius: 12px;
        border: 1px solid #D9EADC;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("♻️ EcoSort AI")
st.caption("AI-Based Smart Waste Segregation • Scan • Learn • Take Action")


# ---------------- SAFE DATA STORAGE ----------------
def read_data(path):
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_data(path, data):
    try:
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        return True
    except OSError as exc:
        st.error(f"Could not save {path.name}: {exc}")
        return False


history = read_data(HISTORY_FILE)
feedback_data = read_data(FEEDBACK_FILE)
eco_points = read_data(POINTS_FILE)


# ---------------- LOAD AI MODELS ----------------
@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    category_model_path = ROOT / "waste_model.keras"
    binary_model_path = ROOT / "binary_waste_model.keras"

    missing = [
        path.name
        for path in (category_model_path, binary_model_path)
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing model file(s): "
            + ", ".join(missing)
            + ". Place both .keras files beside app.py."
        )

    category_model = tf.keras.models.load_model(category_model_path)
    binary_model = tf.keras.models.load_model(binary_model_path)

    return category_model, binary_model


def load_names(filename, fallback):
    path = ROOT / filename

    if not path.exists():
        return fallback

    try:
        with path.open("r", encoding="utf-8") as file:
            names = json.load(file)

        if isinstance(names, list) and names:
            return names

        return fallback

    except (json.JSONDecodeError, OSError):
        return fallback


category_names = load_names(
    "class_names.json",
    ["cardboard", "glass", "metal", "paper", "plastic", "trash"],
)

binary_names = load_names(
    "binary_class_names.json",
    ["O", "R"],
)

try:
    category_model, binary_model = load_models()
    models_ready = True

except Exception as exc:
    models_ready = False
    category_model = None
    binary_model = None

    st.error("The AI models could not be loaded.")
    st.code(f"{type(exc).__name__}: {exc}")

    st.info(
        "Check that waste_model.keras and binary_waste_model.keras "
        "are in the same folder as app.py."
    )


# ---------------- DISPOSAL INFORMATION ----------------
disposal_tips = {
    "cardboard": [
        "Keep cardboard clean and dry.",
        "Flatten boxes before recycling.",
        "Remove food scraps and excessive tape when practical.",
    ],
    "glass": [
        "Empty and rinse containers if local rules recommend it.",
        "Handle broken glass carefully and follow local collection rules.",
        "Do not mix broken glass into recycling unless accepted locally.",
    ],
    "metal": [
        "Empty containers before recycling.",
        "Rinse cans if practical and follow local sorting rules.",
    ],
    "paper": [
        "Keep paper clean and dry.",
        "Avoid mixing it with food-soiled paper.",
        "Check whether coated or laminated paper is accepted locally.",
    ],
    "plastic": [
        "Empty the container before disposal.",
        "Check local recycling rules; not every plastic type is accepted.",
        "Reuse sturdy containers when safe and practical.",
    ],
    "trash": [
        "Check whether the item can be reused, repaired, or recycled.",
        "Use the appropriate local waste collection route.",
        "Keep hazardous or electronic waste out of ordinary bins.",
    ],
}

binary_meanings = {
    "O": "Organic (model prediction)",
    "R": "Recyclable (model prediction)",
    "ORGANIC": "Organic (model prediction)",
    "RECYCLABLE": "Recyclable (model prediction)",
}


# ---------------- NAVIGATION ----------------
st.sidebar.markdown("## ♻️ EcoSort AI")
st.sidebar.caption("Smart Waste Segregation")

page = st.sidebar.radio(
    "Navigate",
    [
        "Waste Scanner",
        "Eco Dashboard",
        "Disposal Guide",
        "Eco Challenges",
        "Feedback",
        "About Project",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Small actions can make a greener future 🌱")


# ---------------- WASTE SCANNER ----------------
if page == "Waste Scanner":
    st.subheader("🔍 Scan Your Waste")
    st.write(
        "Upload a clear image of one waste item for AI predictions."
    )

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        uploaded_file = st.file_uploader(
            "Upload a waste image",
            type=["jpg", "jpeg", "png", "webp"],
            key="waste_upload",
        )

        image = None

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(
                    image,
                    caption="Uploaded waste image",
                    width=350,
                )
            except Exception:
                st.error(
                    "Could not open this image. Try another JPG or PNG."
                )

        else:
            st.info(
                "Tip: use one item per photo, good lighting "
                "and a plain background."
            )

        predict_clicked = st.button(
            "♻️ Analyse Waste",
            type="primary",
            disabled=(image is None or not models_ready),
            use_container_width=True,
        )

    with right:
        st.markdown("### What You Will Get")
        st.markdown(
            """
            - **Material category:** cardboard, glass, metal, paper,
              plastic or trash.
            - **Broad waste type:** Organic or Recyclable.
            - **Confidence scores** from both models.
            - **Smart disposal tips** for common waste materials.
            """
        )

        st.info(
            "The models were trained independently, so their "
            "predictions may sometimes disagree."
        )

    if predict_clicked and image is not None:
        try:
            with st.spinner("Analysing image with both AI models..."):
                # Models perform their own internal rescaling.
                category_input = np.asarray(
                    image.resize((224, 224)),
                    dtype=np.float32,
                )[None, ...]

                binary_input = np.asarray(
                    image.resize((160, 160)),
                    dtype=np.float32,
                )[None, ...]

                category_scores = category_model.predict(
                    category_input,
                    verbose=0,
                )[0]

                binary_scores = binary_model.predict(
                    binary_input,
                    verbose=0,
                )[0]

                category_index = int(np.argmax(category_scores))
                binary_index = int(np.argmax(binary_scores))

                category = str(category_names[category_index]).lower()
                binary_code = str(binary_names[binary_index]).upper()

                binary_label = binary_meanings.get(
                    binary_code,
                    binary_code,
                )

                result = {
                    "time": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "category": category,
                    "category_confidence": round(
                        float(category_scores[category_index]) * 100, 2
                    ),
                    "binary_label": binary_code,
                    "binary_confidence": round(
                        float(binary_scores[binary_index]) * 100, 2
                    ),
                    "category_scores": {
                        str(name): round(float(score) * 100, 2)
                        for name, score in zip(
                            category_names, category_scores
                        )
                    },
                    "binary_scores": {
                        str(name): round(float(score) * 100, 2)
                        for name, score in zip(
                            binary_names, binary_scores
                        )
                    },
                }

                st.session_state["last_result"] = result

                history.append(result)
                save_data(HISTORY_FILE, history)

            st.success("Analysis complete!")

        except Exception as exc:
            st.error("The image could not be processed.")
            st.code(f"{type(exc).__name__}: {exc}")

    # Show the latest prediction
    result = st.session_state.get("last_result")

    if result:
        st.divider()
        st.subheader("🌱 AI Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Waste Category",
                result["category"].title(),
                f'{result["category_confidence"]:.2f}% confidence',
            )

        with col2:
            label = binary_meanings.get(
                result["binary_label"],
                result["binary_label"],
            )

            st.metric(
                "Organic / Recyclable",
                label,
                f'{result["binary_confidence"]:.2f}% confidence',
            )

        if min(
            result["category_confidence"],
            result["binary_confidence"],
        ) < 60:
            st.warning(
                "One or both predictions have low confidence. "
                "Try a clearer image and verify the result."
            )

        st.subheader("Six-Category Model Scores")

        for name, score in result["category_scores"].items():
            st.write(f"**{name.title()} — {score:.2f}%**")
            st.progress(max(0.0, min(1.0, score / 100)))

        st.subheader("Organic / Recyclable Model Scores")

        for name, score in result["binary_scores"].items():
            display_name = binary_meanings.get(name.upper(), name)
            st.write(f"**{display_name} — {score:.2f}%**")
            st.progress(max(0.0, min(1.0, score / 100)))

        st.subheader("♻️ Smart Disposal Assistant")

        for tip in disposal_tips.get(
            result["category"],
            ["Follow your local waste disposal guidelines."],
        ):
            st.write(f"• {tip}")

        st.caption(
            "The models are independently trained and may disagree. "
            "Organic is not exactly the same as biodegradable, and "
            "recyclable does not mean every local service accepts the item."
        )


# ---------------- ECO DASHBOARD ----------------
elif page == "Eco Dashboard":
    st.subheader("🌍 Eco Impact Dashboard")

    total_scans = len(history)

    organic_scans = sum(
        1
        for item in history
        if str(item.get("binary_label", "")).strip().upper() == "O"
    )

    recyclable_scans = sum(
        1
        for item in history
        if str(item.get("binary_label", "")).strip().upper() == "R"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Scans", total_scans)
    col2.metric("Organic Predictions", organic_scans)
    col3.metric("Recyclable Predictions", recyclable_scans)

    st.caption(
        "These are AI predictions, not verified recycling quantities "
        "or measured environmental savings."
    )

    st.subheader("Waste Category Distribution")

    category_counts = {
        name: sum(
            1
            for item in history
            if str(item.get("category", "")).lower() == name.lower()
        )
        for name in category_names
    }

    st.bar_chart(category_counts)

    st.subheader("Organic vs Recyclable")

    st.bar_chart(
        {
            "Organic": organic_scans,
            "Recyclable": recyclable_scans,
        }
    )

    points_total = sum(
        int(item.get("points", 0))
        for item in eco_points
        if str(item.get("points", 0)).isdigit()
    )

    st.metric("Eco Points Earned", points_total)

    if history:
        st.subheader("Recent Scan History")

        for item in reversed(history[-10:]):
            st.write(
                f'{item.get("time", "Unknown time")} — '
                f'{str(item.get("category", "Unknown")).title()} | '
                f'{item.get("binary_label", "Unknown")} | '
                f'{item.get("category_confidence", 0):.2f}% confidence'
            )

        st.download_button(
            "Download Scan History",
            data=json.dumps(history, indent=4),
            file_name="scan_history.json",
            mime="application/json",
        )

        if st.button("Clear Scan History"):
            if save_data(HISTORY_FILE, []):
                history.clear()
                st.session_state.pop("last_result", None)
                st.success("Scan history cleared.")
                st.rerun()

    else:
        st.info("Scan an image to start building your dashboard.")


# ---------------- DISPOSAL GUIDE ----------------
elif page == "Disposal Guide":
    st.subheader("♻️ Waste Disposal Guide")

    st.write(
        "Choose a waste category to see practical disposal suggestions. "
        "Always follow your municipality's rules."
    )

    chosen = st.selectbox(
        "Select waste category",
        ["cardboard", "glass", "metal", "paper", "plastic", "trash"],
    )

    st.markdown(f"### {chosen.title()}")

    for tip in disposal_tips[chosen]:
        st.write(f"• {tip}")

    st.info(
        "Batteries, electronics, chemicals, medicines and other hazardous "
        "waste may need a separate authorised collection point."
    )


# ---------------- ECO CHALLENGES ----------------
elif page == "Eco Challenges":
    st.subheader("🌱 Eco Challenges")

    st.write(
        "Complete an activity and record it to earn project eco points."
    )

    challenges = {
        "Used a reusable bottle or bag": 10,
        "Separated dry and wet waste": 15,
        "Reused or repaired an item": 15,
        "Disposed of e-waste at an authorised point": 25,
        "Helped clean a shared area": 20,
    }

    selected_challenge = st.selectbox(
        "Challenge completed",
        list(challenges.keys()),
    )

    st.write(
        f"Points for this activity: "
        f"**{challenges[selected_challenge]}**"
    )

    if st.button("Record Completed Challenge", type="primary"):
        eco_points.append(
            {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "challenge": selected_challenge,
                "points": challenges[selected_challenge],
            }
        )

        if save_data(POINTS_FILE, eco_points):
            st.success(
                f"Challenge recorded! "
                f"+{challenges[selected_challenge]} points"
            )
            st.rerun()

    total_points = sum(
        int(item.get("points", 0))
        for item in eco_points
        if str(item.get("points", 0)).isdigit()
    )

    st.metric("Your Eco Points", total_points)

    if eco_points:
        st.subheader("Completed Challenges")

        for item in reversed(eco_points[-10:]):
            st.write(
                f'• {item.get("challenge", "Challenge")} — '
                f'+{item.get("points", 0)} points '
                f'({item.get("time", "")})'
            )

    st.caption(
        "Points are self-reported for motivation; "
        "they are not verified impact data."
    )


# ---------------- FEEDBACK ----------------
elif page == "Feedback":
    st.subheader("📝 Help Improve the Predictions")

    result = st.session_state.get("last_result")

    if result:
        st.write(
            f'Latest category: **{result["category"].title()}**'
        )

        st.write(
            f'Latest waste type: **{result["binary_label"]}**'
        )

        with st.form("feedback_form"):
            category_correct = st.radio(
                "Is the six-category prediction correct?",
                ["Yes", "No"],
            )

            correct_category = st.selectbox(
                "Correct category, if wrong",
                ["Not specified"] + [
                    str(name) for name in category_names
                ],
            )

            binary_correct = st.radio(
                "Is the Organic/Recyclable prediction correct?",
                ["Yes", "No"],
            )

            correct_binary = st.selectbox(
                "Correct waste type, if wrong",
                ["Not specified", "O", "R"],
            )

            comment = st.text_area(
                "Additional feedback (optional)"
            )

            submitted = st.form_submit_button("Submit Feedback")

            if submitted:
                feedback_data.append(
                    {
                        "time": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "predicted_category": result["category"],
                        "category_correct": category_correct,
                        "correct_category": (
                            correct_category
                            if category_correct == "No"
                            else result["category"]
                        ),
                        "predicted_binary": result["binary_label"],
                        "binary_correct": binary_correct,
                        "correct_binary": (
                            correct_binary
                            if binary_correct == "No"
                            else result["binary_label"]
                        ),
                        "comment": comment,
                    }
                )

                if save_data(FEEDBACK_FILE, feedback_data):
                    st.success("Feedback saved successfully!")

    else:
        st.info(
            "First scan an image on the Waste Scanner page, "
            "then submit feedback here."
        )


# ---------------- ABOUT PROJECT ----------------
elif page == "About Project":
    st.subheader("About EcoSort AI")

    st.markdown(
        """
        **EcoSort AI** is a student project that uses image classification
        to support waste sorting. It combines two independently trained models.

        - **Six-category model:** cardboard, glass, metal, paper, plastic
          and trash.
        - **Binary model:** Organic (O) or Recyclable (R), based on its
          training labels.
        - **Disposal Guide:** practical tips for common waste categories.
        - **Eco Dashboard:** scan history and prediction summaries.
        - **Eco Challenges:** self-reported actions with motivational points.
        - **Feedback:** lets users record whether predictions seem correct.

        **Technology:** Python, Streamlit, TensorFlow/Keras, NumPy and Pillow.

        **Limitations:** results depend on image quality and training data.
        The six-category model has no dedicated food-scrap class, so it may
        not identify fruit peels correctly. Organic does not always mean
        biodegradable, and recyclable does not mean every local recycling
        service accepts the item. Always verify uncertain results and local rules.
        """
    )

    st.info(
        "This tool is an educational prototype. It does not replace local "
        "waste management guidance or professional handling instructions."
    )
