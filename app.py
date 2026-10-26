
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
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');
 
:root { --green:#176B3A; --green2:#218C52; --soft:#E3F2E8; --border:#D9EADC; --text:#24352B; }
 
/* Font only on text elements, never on icon spans (fixes duplicate "upload"/"add" text) */
html, body, .stApp, [data-testid="stMarkdownContainer"],
label, input, textarea, button p { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { font-family: 'Poppins', sans-serif !important; color: var(--green); }
[data-testid="stIconMaterial"], .material-symbols-rounded {
    font-family: "Material Symbols Rounded" !important;
}
 
.stAppDeployButton, [data-testid="stToolbar"] { display: none !important; }
[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding-top: 2rem; max-width: 1200px; }
[data-testid="stSidebar"] { background: var(--soft); border-right: 1px solid var(--border); }
 
.hero {
    background: linear-gradient(135deg, #176B3A 0%, #2FA866 100%);
    padding: 2rem 2.2rem; border-radius: 18px; margin-bottom: 1.5rem;
    box-shadow: 0 6px 18px rgba(23,107,58,0.25);
}
.hero h1 { color: #fff !important; margin: 0; font-size: 2.4rem; }
.hero p  { color: #E8F7EE; margin: 0.4rem 0 0; font-size: 1.05rem; }
 
[data-testid="stFileUploaderDropzone"] {
    background: #fff; border: 2px dashed #6CA783; border-radius: 14px;
}
[data-testid="stFileUploader"] button {
    background: var(--soft); border: 1px solid #6CA783; border-radius: 8px;
}
 
div.stButton > button[kind="primary"],
div.stFormSubmitButton > button[kind="primary"] {
    background: var(--green2); border: none; color: #fff;
    border-radius: 12px; font-weight: 600; padding: 0.6rem 1rem;
}
div.stButton > button[kind="primary"]:hover { background: var(--green); }
 
.result-card {
    background: #fff; border: 1px solid var(--border); border-radius: 16px;
    padding: 1.2rem 1.4rem; box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
.rc-title { color: #5B7A66; font-size: 0.9rem; }
.rc-value { color: var(--green); font-family: 'Poppins', sans-serif;
            font-size: 1.9rem; font-weight: 600; margin: 0.2rem 0 0.6rem; }
.badge { background: #DDF3E5; color: #176B3A; padding: 3px 12px;
         border-radius: 999px; font-size: 0.85rem; font-weight: 600; }
.badge.low { background: #FDECC8; color: #8A5A00; }
 
.bar-row { margin-bottom: 0.9rem; }
.bar-label { display: flex; justify-content: space-between; font-weight: 500; margin-bottom: 4px; }
.bar-track { background: #DCEBE0; border-radius: 999px; height: 12px; overflow: hidden; }
.bar-fill  { background: linear-gradient(90deg, #218C52, #5CCB8A); height: 100%; border-radius: 999px; }
 
[data-testid="stMetric"] {
    background: #fff; padding: 14px; border-radius: 14px; border: 1px solid var(--border);
}
</style>
""",
    unsafe_allow_html=True,
)
 
st.markdown(
    """
<div class="hero">
  <h1>♻️ EcoSort AI</h1>
  <p>AI-Based Smart Waste Segregation • Scan • Learn • Take Action</p>
</div>
""",
    unsafe_allow_html=True,
)
 
 
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
    paths = (ROOT / "waste_model.keras", ROOT / "binary_waste_model.keras")
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model file(s): " + ", ".join(missing)
            + ". Place both .keras files beside app.py."
        )
    return tf.keras.models.load_model(paths[0]), tf.keras.models.load_model(paths[1])
 
 
def load_names(filename, fallback):
    path = ROOT / filename
    if not path.exists():
        return fallback
    try:
        with path.open("r", encoding="utf-8") as file:
            names = json.load(file)
        return names if isinstance(names, list) and names else fallback
    except (json.JSONDecodeError, OSError):
        return fallback
 
 
category_names = load_names(
    "class_names.json", ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
)
binary_names = load_names("binary_class_names.json", ["O", "R"])
 
try:
    category_model, binary_model = load_models()
    models_ready = True
except Exception as exc:
    models_ready = False
    category_model = binary_model = None
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
    "O": "Organic",
    "R": "Recyclable",
    "ORGANIC": "Organic",
    "RECYCLABLE": "Recyclable",
}
 
 
# ---------------- UI HELPERS ----------------
def score_bar(label, score):
    st.markdown(
        f'<div class="bar-row"><div class="bar-label"><span>{label}</span>'
        f"<b>{score:.2f}%</b></div><div class=\"bar-track\">"
        f'<div class="bar-fill" style="width:{max(0, min(100, score))}%"></div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )
 
 
def result_card(title, value, conf):
    cls = "badge low" if conf < 60 else "badge"
    st.markdown(
        f'<div class="result-card"><div class="rc-title">{title}</div>'
        f'<div class="rc-value">{value}</div>'
        f'<span class="{cls}">{conf:.2f}% confidence</span></div>',
        unsafe_allow_html=True,
    )
 
 
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
    st.write("Upload a clear image of one waste item for AI predictions.")
 
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
                st.image(image, caption="Uploaded waste image", width=350)
            except Exception:
                st.error("Could not open this image. Try another JPG or PNG.")
        else:
            st.info(
                "Tip: use one item per photo, good lighting and a plain background."
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
- **Material category:** cardboard, glass, metal, paper, plastic or trash.
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
                    image.resize((224, 224)), dtype=np.float32
                )[None, ...]
                binary_input = np.asarray(
                    image.resize((160, 160)), dtype=np.float32
                )[None, ...]
 
                category_scores = category_model.predict(category_input, verbose=0)[0]
                binary_scores = binary_model.predict(binary_input, verbose=0)[0]
 
                category_index = int(np.argmax(category_scores))
                binary_index = int(np.argmax(binary_scores))
 
                result = {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "category": str(category_names[category_index]).lower(),
                    "category_confidence": round(
                        float(category_scores[category_index]) * 100, 2
                    ),
                    "binary_label": str(binary_names[binary_index]).upper(),
                    "binary_confidence": round(
                        float(binary_scores[binary_index]) * 100, 2
                    ),
                    "category_scores": {
                        str(n): round(float(s) * 100, 2)
                        for n, s in zip(category_names, category_scores)
                    },
                    "binary_scores": {
                        str(n): round(float(s) * 100, 2)
                        for n, s in zip(binary_names, binary_scores)
                    },
                }
 
                st.session_state["last_result"] = result
                history.append(result)
                save_data(HISTORY_FILE, history)
 
            st.success("Analysis complete!")
 
        except Exception as exc:
            st.error("The image could not be processed.")
            st.code(f"{type(exc).__name__}: {exc}")
 
    result = st.session_state.get("last_result")
 
    if result:
        st.divider()
        st.subheader("🌱 AI Prediction Results")
 
        col1, col2 = st.columns(2)
        with col1:
            result_card(
                "Waste category",
                result["category"].title(),
                result["category_confidence"],
            )
        with col2:
            result_card(
                "Organic / Recyclable",
                binary_meanings.get(result["binary_label"], result["binary_label"]),
                result["binary_confidence"],
            )
 
        if min(result["category_confidence"], result["binary_confidence"]) < 60:
            st.warning(
                "One or both predictions have low confidence. "
                "Try a clearer image and verify the result."
            )
 
        st.write("")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.subheader("Six-Category Model Scores")
            for name, score in result["category_scores"].items():
                score_bar(name.title(), score)
        with c2:
            st.subheader("Organic / Recyclable Scores")
            for name, score in result["binary_scores"].items():
                score_bar(binary_meanings.get(name.upper(), name), score)
 
        st.subheader("♻️ Smart Disposal Assistant")
        for tip in disposal_tips.get(
            result["category"], ["Follow your local waste disposal guidelines."]
        ):
            st.success(tip, icon="✅")
 
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
        1 for i in history if str(i.get("binary_label", "")).strip().upper() == "O"
    )
    recyclable_scans = sum(
        1 for i in history if str(i.get("binary_label", "")).strip().upper() == "R"
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
    st.bar_chart(
        {
            name: sum(
                1 for i in history if str(i.get("category", "")).lower() == name.lower()
            )
            for name in category_names
        },
        color="#218C52",
    )
 
    st.subheader("Organic vs Recyclable")
    st.bar_chart(
        {"Organic": organic_scans, "Recyclable": recyclable_scans},
        color="#218C52",
    )
 
    points_total = sum(
        int(i.get("points", 0)) for i in eco_points if str(i.get("points", 0)).isdigit()
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
        st.success(tip, icon="✅")
 
    st.info(
        "Batteries, electronics, chemicals, medicines and other hazardous "
        "waste may need a separate authorised collection point."
    )
 
 
# ---------------- ECO CHALLENGES ----------------
elif page == "Eco Challenges":
    st.subheader("🌱 Eco Challenges")
    st.write("Complete an activity and record it to earn project eco points.")
 
    challenges = {
        "Used a reusable bottle or bag": 10,
        "Separated dry and wet waste": 15,
        "Reused or repaired an item": 15,
        "Disposed of e-waste at an authorised point": 25,
        "Helped clean a shared area": 20,
    }
 
    selected_challenge = st.selectbox("Challenge completed", list(challenges.keys()))
    st.write(f"Points for this activity: **{challenges[selected_challenge]}**")
 
    if st.button("Record Completed Challenge", type="primary"):
        eco_points.append(
            {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "challenge": selected_challenge,
                "points": challenges[selected_challenge],
            }
        )
        if save_data(POINTS_FILE, eco_points):
            st.success(f"Challenge recorded! +{challenges[selected_challenge]} points")
            st.rerun()
 
    total_points = sum(
        int(i.get("points", 0)) for i in eco_points if str(i.get("points", 0)).isdigit()
    )
    st.metric("Your Eco Points", total_points)
 
    if eco_points:
        st.subheader("Completed Challenges")
        for item in reversed(eco_points[-10:]):
            st.write(
                f'• {item.get("challenge", "Challenge")} — '
                f'+{item.get("points", 0)} points ({item.get("time", "")})'
            )
 
    st.caption(
        "Points are self-reported for motivation; they are not verified impact data."
    )
 
 
# ---------------- FEEDBACK ----------------
elif page == "Feedback":
    st.subheader("📝 Help Improve the Predictions")
 
    result = st.session_state.get("last_result")
 
    if result:
        st.write(f'Latest category: **{result["category"].title()}**')
        st.write(
            "Latest waste type: "
            f'**{binary_meanings.get(result["binary_label"], result["binary_label"])}**'
        )
 
        with st.form("feedback_form"):
            category_correct = st.radio(
                "Is the six-category prediction correct?", ["Yes", "No"]
            )
            correct_category = st.selectbox(
                "Correct category, if wrong",
                ["Not specified"] + [str(n) for n in category_names],
            )
            binary_correct = st.radio(
                "Is the Organic/Recyclable prediction correct?", ["Yes", "No"]
            )
            correct_binary = st.selectbox(
                "Correct waste type, if wrong", ["Not specified", "O", "R"]
            )
            comment = st.text_area("Additional feedback (optional)")
 
            submitted = st.form_submit_button("Submit Feedback", type="primary")
 
            if submitted:
                feedback_data.append(
                    {
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
 
- **Six-category model:** cardboard, glass, metal, paper, plastic and trash.
- **Binary model:** Organic (O) or Recyclable (R), based on its training labels.
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