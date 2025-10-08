import streamlit as st
import pandas as pd
import json
import google.generativeai as genai

st.set_page_config(page_title="Universal NER with Gemini", layout="wide")
st.title("Universal Domain-Aware NER (Google Gemini) — Long Text Support")
st.markdown("Paste text below or upload a .txt file, and extract entities across any domain.")

# --- Ask user for Gemini API key ---
api_key = st.text_input("Enter your Gemini API Key:", type="password")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.warning("Please enter your Gemini API key to proceed.")
    st.stop()  # Stop app until key is provided

# --- Input Section ---
txt_input = st.text_area("Paste text here (any length)", height=200)
uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"])
if uploaded_file is not None:
    txt_input = uploaded_file.read().decode('utf-8')

if st.button("Load example"):
    txt_input = (
        "Apple is looking at buying U.K. startup for $1 billion.\n"
        "Barack Obama was born in Hawaii. He was elected President in 2008.\n"
        "Google LLC is headquartered in Mountain View, California.\n"
        "COVID-19 vaccines are produced by Pfizer and Moderna."
    )

# --- NER Processing ---
if txt_input:
    st.subheader("Extracting Entities...")
    all_entities = []

    # --- Optional: chunking for long text can be added here ---
    try:
        prompt = f"""
        Extract all named entities from the following text and categorize them by domain 
        (Person, Organization, Location, Product, Law, Disease, etc.) in JSON format.
        Include the text, type, start_char, and end_char.
        Text: {txt_input}
        """
        response = genai.generate(model="gemini-1.5-flash", prompt=prompt)
        text_response = response.text.strip()

        try:
            all_entities = json.loads(text_response)
        except Exception:
            st.warning("Gemini returned non-JSON response. Showing raw text:")
            st.code(text_response)

        if all_entities:
            df = pd.DataFrame(all_entities)
            st.subheader("Entities Table")
            st.dataframe(df)

            # Highlight entities in text
            st.subheader("Entities Highlighted in Text")
            highlighted_text = txt_input
            for ent in sorted(all_entities, key=lambda x: x.get("start_char", 0), reverse=True):
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

else:
    st.info("Paste text above, upload a file, or load the example to get started.")
