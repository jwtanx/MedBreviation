# To start: streamlit run main.py
import os
import gdown
import base64
from io import BytesIO
import streamlit as st

def setup():
    st.set_page_config(page_title="MedBreviation", layout="wide", page_icon="logo2.png", initial_sidebar_state="auto")
    st.title("MedBreviation")
    if not os.path.exists("tmp"):
        # Make a new folder for the temp
        os.mkdir("tmp")
    if not os.path.exists("dataset"):
        # Make a new folder for the temp
        os.mkdir("dataset")
        print("Downloading important files")
        gdown.download(id="1-3f1qu69xhswpTbm7AeZe_cUxvkyJTrr", output="./dataset/clinical_abbr_dataset_sm.csv")


def start():
    textbox = st.empty().text_area(label="You can paste in the text here", value="", placeholder="Please take note that only the capitalized words will be processed...")

    if textbox:
        st.button("RUN", disabled=False, on_click=extract)
    else:
        st.button("RUN", disabled=True)

    st.empty().caption('<h2 align=center>OR</h2>', unsafe_allow_html=True)


    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "png", "jpg", "jpeg"])
    display_file = st.empty()
    filename = ""

    if not uploaded_file:
        display_file.info("Please drag in your file here")
        return

    content = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()
    tmp_filepath = os.path.join("./tmp", filename)

    if isinstance(uploaded_file, BytesIO):

        with open(tmp_filepath, "wb") as f:
            f.write(content)  # write this content elsewhere

        if tmp_filepath.endswith(".pdf"):

            # Showing the extracted text from pdf
            extracted = extract_text_from_doc(tmp_filepath)
            st.empty().caption("<h2>Text extracted from PDF</h2>", unsafe_allow_html=True)
            text_display = f'<div style="overflow:scroll;height:300px;overflow-x:hidden;">{extracted}</div></br>'
            st.markdown(text_display, unsafe_allow_html=True)

            # TODO: Run the function for abbr locater here

            # Showing the original pdf
            base64_pdf = base64.b64encode(content).decode("utf-8")
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width=100% height=1000 type="application/pdf" >'
            st.markdown(pdf_display, unsafe_allow_html=True)

        else:
            display_file.image(uploaded_file)

    uploaded_file.close()


if __name__ == "__main__":
    setup()
    from utils import extract_text_from_doc, extract
    start()

