import cv2
import re
import pytesseract
import numpy as np
import pandas as pd
from nltk import tokenize
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
# MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# MODEL2 = SentenceTransformer("bert-base-nli-mean-tokens")
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================================== #

df = pd.read_csv("dataset/clinical_abbr_dataset_sm.csv")

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
        cur_ls = re.findall(r"[\s]{1,}[\W+]?[A-Z]{2,}[\W+]?[^\.][\s]?", sent)
        for abbr in cur_ls:
            abbr_sent[abbr.strip()] = sent

    return abbr_sent


def get_abbr_fullform(content):
    """ Getting the full form of the abbreviation 
    
    Parameters
    ----------
    content : str
        The content extract

    Returns
    -------
    list
        The list of the full form
    dict
        The dictionary of the list of the abbreviation with their respective sentence
        Key: Abbreviation
        Val: The sentence in which the abbreviation is in

    """
    
    abbr_sent = _locate_abbr(content)

    abbr_fullform = ["N/A" for _ in range(len(abbr_sent))]

    for i, (abbr, sent) in enumerate(abbr_sent.items()):
        filtered_df = df[df.ABBR == abbr]
        text_ls = filtered_df.TEXT.tolist()
        abbr_ls = filtered_df.LABEL.tolist()

        if text_ls == []:
            continue

        embeddings = MODEL.encode(text_ls) # TODO: Remove this and use the processed dataset
        cur_embed = MODEL.encode(sent)

        similarities = cosine_similarity(
            [cur_embed],
            embeddings
        ).flatten()

        abbr_fullform[i] = abbr_ls[np.argmax(similarities)+1]

    return abbr_fullform, abbr_sent

