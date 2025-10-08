import google.generativeai as genai
import sys
import pkg_resources

st.write("Python version:", sys.version)
st.write("Streamlit version:", pkg_resources.get_distribution("streamlit").version)
st.write("Pandas version:", pkg_resources.get_distribution("pandas").version)
st.write("Google GenerativeAI version:", pkg_resources.get_distribution("google-generativeai").version)
