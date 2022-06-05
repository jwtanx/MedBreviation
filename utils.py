import re
import cv2
import json
import nltk
nltk.download("punkt")
import pickle
import pytesseract
import numpy as np
import pandas as pd
from nltk import tokenize
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# MODEL2 = SentenceTransformer("bert-base-nli-mean-tokens")
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================================== #

# Reading the list of abbreviation available
with open("./abbr.json", "r") as f:
    db_abbr_ls = json.load(f)

# Getting the vectorized pandas dataframe from the pickle
with open("./dataset/clinical_abbr_full.pkl", "rb") as f:
    df = pickle.load(f)

def extract_text_from_doc(filename):
    """Extracting the text from the images / pdf uploaded by the user
    
    Parameters
    ----------
    filename : str
        The path of the temporary file uploaded by the users.
        Can be in the form of pdf or any kind of image extensions

    Returns
    -------
    str
        The texts extracted from the document

    """
    content = ""

    # Checking if the file is pdf or image
    if filename.endswith(".pdf"):
        content = " ".join(extract_text(filename).split())
    else:
        img = cv2.imread(filename, 0)
        extracted = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)["text"]
        extracted = " ".join([w for w in extracted if w != ""])
        content = re.sub(r"\s+", " ", extracted).strip()

    return content


def _locate_abbr(content):
    """ Recognize the list of abbreviation and its sentence 
    
    Parameters
    ----------
    content : str
        The content extract
    
    Returns
    -------
    dict
        The dictionary of the list of the abbreviation with their respective sentence
        Key: Abbreviation
        Val: The sentence in which the abbreviation is in

    """
    sent_ls = tokenize.sent_tokenize(content)

    abbr_sent = {}
    for sent in sent_ls:
        # TODO: Chcek English word "in", "or" not to be taken in as abbr
        cleaned_sent = re.sub(r"[\,|\.|\?|\!]", "", sent)
        cur_ls = [w for w in cleaned_sent.split() if w in db_abbr_ls and w.lower() != w]
        for abbr in cur_ls:
            abbr_sent[abbr] = sent

    return abbr_sent


def get_abbr_fullform(content):
    """ Getting the full form of the abbreviation 
    
    Parameters
    ----------
    content : str
        The content extract

    Returns
    -------
    pandas.core.frame.DataFrame
        Includes: Abbreviation, Fullform & Sentence

    """
    # Locating the list of the abbreviation with their respective sentence
    abbr_sent = _locate_abbr(content)

    # Initialize the list of the fullform
    abbr_fullform = ["N/A" for _ in range(len(abbr_sent))] # N/A is impossible but for future work

    for i, (abbr, sent) in enumerate(abbr_sent.items()):
        filtered_df = df[df.ABBR == abbr]
        vector_ls = filtered_df[filtered_df.columns[2:]].values.tolist()
        abbr_ls = filtered_df.LABEL.values.tolist()

        if vector_ls != []:
            cur_embed = MODEL.encode(sent, show_progress_bar=False)

            similarities = cosine_similarity(
                [cur_embed],
                vector_ls
            ).flatten()

            abbr_fullform[i] = abbr_ls[np.argmax(similarities)+1]

    table = pd.DataFrame()
    table["Abbreviation"] = abbr_sent.keys()
    table["Fullform"] = abbr_fullform
    table["Sentence"] = [re.sub(abbr, f"‼️{abbr}‼️", sent) for abbr, sent in abbr_sent.items()]

    return table

