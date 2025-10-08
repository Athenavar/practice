import streamlit as st
import pandas as pd
import json
import google.generativeai as genai

# ---------------- Page Setup ----------------
st.set_page_config(page_title="Universal NER with Gemini", layout="wide")
st.title("Universal Domain-Aware NER (Google Gemini)")
st.markdown(
    "Select an input method, provide your text or file, and click **Extract Entities**."
)

# ---------------- Gemini API Key ----------------
api_key = st.text_input("Enter your Gemini API Key:", type="password")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.warning("Please enter your Gemini API key to proceed.")
    st.stop()

# ---------------- Input Method Selection ----------------
input_method = st.radio(
    "Choose input method:",
    ("Paste Text", "Upload File")
)

txt_input = ""

if input_method == "Paste Text":
    txt_input = st.text_area("Paste your text here", height=200)
elif input_method == "Upload File":
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file is not None:
        txt_input = uploaded_file.read().decode('utf-8')

# ---------------- Extract Button ----------------
if st.button("Extract Entities"):

    if not txt_input.strip():
        st.warning("Please provide input text or upload a file first.")
    else:
        st.subheader("Extracting Entities...")
        try:
            prompt = f"""
            Extract all named entities from the following text and categorize them by domain
            (Person, Organization, Location, Product, Law, Disease, etc.) in JSON format.
            Include the text, type, start_char, and end_char.
            Text: {txt_input}
            """

            response = genai.chat.completions.create(
                model="gemini-1.5-chat",
                messages=[
                    {"role": "system", "content": "You are an expert NER assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            text_response = response.choices[0].message.content

            # Parse JSON
            try:
                entities = json.loads(text_response)
            except Exception:
                st.warning("Gemini returned non-JSON response. Showing raw text:")
                st.code(text_response)
                entities = []

            if entities:
                df = pd.DataFrame(entities)
                st.subheader("Entities Table")
                st.dataframe(df)

                # Highlight entities in text
                st.subheader("Entities Highlighted in Text")
                highlighted_text = txt_input
                for ent in sorted(entities, key=lambda x: x.get("start_char", 0), reverse=True):
                    start = ent.get("start_char")
                    end = ent.get("end_char")
                    label = ent.get("type", "ENTITY")
                    if start is not None and end is not None:
                        highlighted_text = (
                            highlighted_text[:start] +
                            f"<mark title='{label}'>{highlighted_text[start:end]}</mark>" +
                            highlighted_text[end:]
                        )
                st.markdown(highlighted_text, unsafe_allow_html=True)

                # Download options
                st.subheader("Download")
                st.download_button(
                    "Download JSON",
                    data=df.to_json(orient='records').encode('utf-8'),
                    file_name='entities.json',
                    mime='application/json'
                )
                st.download_button(
                    "Download CSV",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name='entities.csv',
                    mime='text/csv'
                )
            else:
                st.info("No entities detected.")

        except Exception as e:
            st.error(f"Error calling Gemini API: {e}")
