import streamlit as st
from views import show_sidebar, show_current_view
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Gemma Fine-Tuning Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "current_step" not in st.session_state:
    st.session_state.current_step = 1

# Initialize step completion tracking
for i in range(1, 5):
    if f"step_{i}_completed" not in st.session_state:
        st.session_state[f"step_{i}_completed"] = False


def main():
    with st.sidebar:
        show_sidebar()
    show_current_view()


if __name__ == "__main__":
    main()
