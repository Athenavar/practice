import streamlit as st
import pandas as pd
import json
import google.generativeai as genai

st.set_page_config(page_title="Universal NER with Gemini (Chunked)", layout="wide")
st.title("Universal Domain-Aware NER (Google Gemini) — Long Text Support")
st.markdown("Paste text below or upload a `.txt` file, and extract entities across any domain.")

# Configure API key
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

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
        "COVID-19 vaccines are produced by Pfizer and Moderna.\n"
        "Tesla released the new Model S Plaid in 2023.\n"
        "The GDPR is a regulation in EU law on data protection and privacy."
    )

# --- Function to chunk text ---
def chunk_text(text, chunk_size=500):
    """Split text into chunks of approximately chunk_size words."""
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield ' '.join(words[i:i+chunk_size]), i  # return chunk and starting word index

# --- NER Processing ---
if txt_input:
    st.subheader("Extracting Entities (Chunked)...")
    all_entities = []

    try:
        for chunk_text_str, word_start_idx in chunk_text(txt_input, chunk_size=500):
            prompt = f"""
            Extract all named entities from the following text and categorize them by domain 
            (Person, Organization, Location, Product, Law, Disease, etc.) in JSON format.
            Include the text, type, start_char, and end_char relative to the chunk.
            Text: {chunk_text_str}
            """
            response = genai.generate(model="gemini-1.5-flash", prompt=prompt)
            chunk_response = response.text.strip()

            # Parse JSON safely
            try:
                chunk_entities = json.loads(chunk_response)
                # Adjust start_char/end_char to original text positions
                for ent in chunk_entities:
                    if 'start_char' in ent and 'end_char' in ent:
                        ent['start_char'] += txt_input.find(chunk_text_str)
                        ent['end_char'] += txt_input.find(chunk_text_str)
                all_entities.extend(chunk_entities)
            except Exception:
                st.warning(f"Chunk {word_start_idx} returned non-JSON response, skipping.")
                continue

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

            # Download
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

st.markdown("---")
st.markdown("Built with ❤️ — Google Gemini + Streamlit. Supports any domain NER and long texts.")
