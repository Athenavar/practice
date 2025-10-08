import streamlit as st
import pandas as pd
import json
import openai

# ---------------- Page Setup ----------------
st.set_page_config(page_title="Universal NER with OpenAI", layout="wide")
st.title("Universal Domain-Aware NER (OpenAI NER)")
st.markdown(
    "Paste text or upload a .txt file, then click **Extract Entities**."
)

# ---------------- OpenAI API Key ----------------
api_key = st.text_input("Enter your OpenAI API Key:", type="password")
if not api_key:
    st.warning("Please enter your OpenAI API key to proceed.")
    st.stop()
openai.api_key = api_key

# ---------------- Input Method ----------------
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
            Extract all named entities from the following text and categorize them by type
            (Person, Organization, Location, Product, Law, Disease, etc.) in JSON format.
            Include the text, type, start_char, and end_char.
            Text: {txt_input}
            """

            # ---------------- OpenAI ChatCompletion ----------------
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )

            text_response = response.choices[0].message.content

            # ---------------- Parse JSON ----------------
            try:
                entities = json.loads(text_response)
            except Exception:
                st.warning("OpenAI returned non-JSON response. Showing raw text:")
                st.code(text_response)
                entities = []

            # ---------------- Display Results ----------------
            if entities:
                df = pd.DataFrame(entities)
                st.subheader("Entities Table")
                st.dataframe(df)

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

                # ---------------- Download Options ----------------
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
            st.error(f"Error calling OpenAI API: {e}")
