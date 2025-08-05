import os
import base64
import streamlit as st
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage
from dotenv import load_dotenv
import tempfile
import markdown2


# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Set Streamlit page configuration
st.set_page_config(page_title="Medical Imaging Diagnosis Agent", layout="centered")

# Set background image (local)
def set_background_from_local(path="background.jpg"):
    with open(path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
    """, unsafe_allow_html=True)

set_background_from_local("background.jpg")  # <-- Use your actual file name

# App UI
st.markdown("""
<h1 style="
    font-size: 3.2rem;
    font-weight: 900;
    text-align: center;
    color: #00d9ff;
    margin-bottom: 1.5rem;">
⚕️ <span style="color:#ffffff;">Medical Image Diagnosis Assistant</span>
</h1>
""", unsafe_allow_html=True)



st.markdown("""
<div style="
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border-radius: 12px;
    padding: 20px;
    margin: 20px auto;
    width: 80%;
    text-align: center;
    color: white;
    font-size: 18px;
    font-weight: 500;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
">
Upload a medical image to receive an AI-generated analysis report.
</div>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* Label text (e.g., "Translate Report To") */
.stSelectbox label {
    color: #ffffff !important;
    font-weight: 600;
    font-size: 16px;
}

/* Selectbox container */
div[data-baseweb="select"] > div {
    background-color: #1e1e1e !important;
    color: #ffffff !important;
    font-weight: 600;
    font-size: 16px;
    border-radius: 10px;
    border: 1px solid #ff4b4b !important;  /* optional: red border */
}

/* Selected value inside select box */
div[data-baseweb="select"] span {
    color: #ffffff !important;
    font-weight: 600;
    font-size: 16px;
}

/* Dropdown menu options */
ul[role="listbox"] li {
    background-color: #1e1e1e !important;
    color: #ffffff !important;
}

/* Highlight hovered option */
ul[role="listbox"] li:hover {
    background-color: #333333 !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)


# Language selector
language = st.selectbox("🌐 Translate Report To", [
    "English", "Hindi", "Telugu", "Tamil", "Kannada", "Malayalam", "Spanish", "French", "German"
])





st.markdown("""
<style>
/* Make uploaded file preview text highly visible */
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] div {
    color: #ffffff !important;
    font-weight: 800;
    font-size: 1.1rem !important;
    text-shadow:
        0 0 6px rgba(0, 0, 0, 1),
        0 0 12px rgba(0, 0, 0, 0.9);
}
</style>
""", unsafe_allow_html=True)




uploaded_file = st.file_uploader(" 🖼️Upload a medical image", type=["jpg", "jpeg", "png"])


# Prompt used for analysis
query = """
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. Analyze the patient's medical image and structure your response as follows:

### 1. Image Type & Region
- Specify imaging modality (X-ray/MRI/CT/Ultrasound/etc.)
- Identify the patient's anatomical region and positioning
- Comment on image quality and technical adequacy

### 2. Key Findings
- List primary observations systematically
- Note any abnormalities in the patient's imaging with precise descriptions
- Include measurements and densities where relevant
- Describe location, size, shape, and characteristics
- Rate severity: Normal/Mild/Moderate/Severe

### 3. Diagnostic Assessment
- Provide primary diagnosis with confidence level
- List differential diagnoses in order of likelihood
- Support each diagnosis with observed evidence from the patient's imaging
- Note any critical or urgent findings

### 4. Patient-Friendly Explanation
- Explain the findings in simple, clear language that the patient can understand
- Avoid medical jargon or provide clear definitions
- Include visual analogies if helpful
- Address common patient concerns related to these findings


Format your response using clear markdown headers and bullet points. Be concise yet thorough.
Then translate the entire report into the following language:
"""



st.markdown("""
<style>
/* Analyze Button Styling */
div.stButton > button {
    background-color: #00bfff;  /* Bright cyan */
    color: #ffffff;             /* White text */
    font-weight: bold;
    font-size: 1rem;
    padding: 0.6rem 1.2rem;
    border: none;
    border-radius: 10px;
    transition: 0.3s ease-in-out;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.25);
}
div.stButton > button:hover {
    background-color: #009ac9; /* Slightly darker on hover */
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
}

/* Spinner Styling (Analyzing image...) */
.css-1cpxqw2, .stSpinner > div {
    color: #ffffff !important;         /* White for dark BG */
    font-size: 1.1rem;
    font-weight: 600;
    text-shadow: 0px 0px 4px rgba(0, 0, 0, 0.6); /* visible on light BG */
}
</style>
""", unsafe_allow_html=True)

# Analyze button
if st.button("🔍 Analyze"):
    if not api_key:
        st.error("❌ Google API key not found. Please check your .env file.")
    elif not uploaded_file:
        st.error("❌ Please upload a medical image.")
    else:
        image = PILImage.open(uploaded_file).convert("RGB")
        image = image.resize((500, int(500 / image.width * image.height)))

        # Save to temporary path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            image.save(tmp.name)
            temp_path = tmp.name

        with st.spinner("Analyzing image..."):
            full_prompt = query.strip() + f"\n\n{language}"
            agent = Agent(
                model=Gemini(id="gemini-2.0-flash", api_key=api_key),
                tools=[DuckDuckGoTools()],
                markdown=True
            )
            agno_img = AgnoImage(filepath=temp_path)
            result = agent.run(full_prompt, images=[agno_img])

        # Convert Markdown to HTML
        clean_html = markdown2.markdown(result.content)

        # Apply CSS and styled HTML
        st.markdown("""
            <style>
            .custom-report {
                background-color: rgba(255, 255, 255, 0.85);
                padding: 24px;
                border-radius: 12px;
                color: #111 !important;
                font-family: 'Segoe UI', sans-serif;
            }

            .custom-report *, 
            .custom-report h1, 
            .custom-report h2, 
            .custom-report h3, 
            .custom-report h4, 
            .custom-report strong {
                color: #111 !important;
                text-shadow: none !important;
            }
            </style>
        """, unsafe_allow_html=True)

        # Now wrap and display converted content
        st.markdown(f"<div class='custom-report'>{clean_html}</div>", unsafe_allow_html=True)
