# To start: streamlit run main.py
import os
import nltk
import gdown
import base64
from io import BytesIO
import streamlit as st

def setup():
    """ Initial setup: creating folder and downloading dataset """
    st.set_page_config(page_title="MedBreviation", layout="wide", page_icon="logo.png", initial_sidebar_state="auto")
    col1, col2 = st.columns([0.6, 10])
    with col1:
        st.image("logo.png")
    with col2:
        st.title("MedBreviation")
    
    st.subheader("📝 Helps you to capture and analyze the abbreviations in medical notes")
    nltk.download("punkt")
    
    if not os.path.exists("tmp"):
        # Make a new folder for the temp
        os.mkdir("tmp")
    if not os.path.exists("dataset"):
        # Make a new folder for the temp
        os.mkdir("dataset")
        print("Downloading important files")
        gdown.download(id="1-3f1qu69xhswpTbm7AeZe_cUxvkyJTrr", output="./dataset/clinical_abbr_dataset_sm.csv")


def get_abbr_fullform_helper(content):
    abbr_fullform, abbr_sent = get_abbr_fullform(content)


def start():
    """ Main webpage """
    textbox = st.text_area(label="You can paste in the text here", placeholder="Please take note that only the capitalized words will be processed...")

    if textbox:
        st.button("RUN", disabled=False, on_click=get_abbr_fullform_helper, args=[textbox])
    else:
        st.button("RUN", disabled=True)

    st.caption('<h2 align=center>OR</h2>', unsafe_allow_html=True)

    if not textbox:
        uploaded_file = st.file_uploader("Upload a document", type=["pdf", "png", "jpg", "jpeg"])
        display_file = st.empty()
        filename = ""

        if not uploaded_file:
            display_file.info("Waiting for file to be uploaded")
            return

        bytes = uploaded_file.getvalue()
        filename = uploaded_file.name.lower()
        tmp_filepath = os.path.join("./tmp", filename)

        if isinstance(uploaded_file, BytesIO):

            with open(tmp_filepath, "wb") as f:
                f.write(bytes)  # write this content elsewhere

            if tmp_filepath.endswith(".pdf"):

                # Showing the extracted text from pdf
                extracted = extract_text_from_doc(tmp_filepath)
                st.caption("<h2>Text extracted from PDF</h2>", unsafe_allow_html=True)
                text_display = f'<div style="overflow:auto;height:150px;overflow-x:hidden;">{extracted}</div></br>'
                st.markdown(text_display, unsafe_allow_html=True)

                # TODO: Run the function for abbr locater here
                abbr_fullform, abbr_sent = get_abbr_fullform(extracted)

                # Showing the original pdf
                base64_pdf = base64.b64encode(bytes).decode("utf-8")
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width=100% height=1000 type="application/pdf" >'
                st.markdown(pdf_display, unsafe_allow_html=True)

            else:
                extracted = extract_text_from_doc(tmp_filepath)
                st.caption("<h2>Text extracted from IMAGE</h2>", unsafe_allow_html=True)
                text_display = f'<div style="overflow:auto;height:150px;overflow-x:hidden;">{extracted}</div></br>'
                st.markdown(text_display, unsafe_allow_html=True)

                base64_img = base64.b64encode(bytes).decode("utf-8")
                img_display = f'<img src="data:image;base64,{base64_img}" alt="Uploaded" width=100%></br>'
                st.markdown(img_display, unsafe_allow_html=True)

        uploaded_file.close()


if __name__ == "__main__":
    setup()
    from utils import extract_text_from_doc, get_abbr_fullform
    start()