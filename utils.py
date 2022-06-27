import re
import os
import cv2
import json
import nltk
import platform
import pickle
import pytesseract
import numpy as np
import pandas as pd
from nltk import tokenize
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

@st.cache(allow_output_mutation=True)
def load_model():
    nltk.download("punkt")
    nltk.download('omw-1.4')
    try:
        if platform.system() == "Windows":
            return SentenceTransformer(f"C:/Users/{os.getlogin()}/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2")
        else:
            return SentenceTransformer(f"~/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2")
    except Exception as e:
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# MODEL2 = SentenceTransformer("bert-base-nli-mean-tokens")
MODEL = load_model()

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


def _preprocess(content):
    """ Preprocessing the extracted content

    Parameters
    ----------
    content : str
        The textual content will need to be preprocessed before passing it into the
    
    Returns
    -------
    str
        Preprocessed text for the model to vectorize the whole text into a vector

    """
    # Removing the numbers
    processing = re.sub(r"[\d+|\d+.\d+]", "", content)

    # Medical + English stopword removal
    extra_stop_words = "disease, diseases, disorder, symptom, symptoms, drug, drugs, problems, problem,prob, probs, med, meds,\
    , pill, pills, medicine, medicines, medication, medications, treatment, treatments, caps, capsules, capsule,\
    , tablet, tablets, tabs, doctor, dr, dr., doc, physician, physicians, test, tests, testing, specialist, specialists,\
    , side-effect, side-effects, patient, patients, pharmaceutical, pharmaceuticals, pharma, diagnosis, diagnose, diagnosed, exam,\
    , challenge, device, condition, conditions, suffer, suffering ,suffered, feel, feeling, prescription, prescribe,\
    , prescribed, over-the-counter, a, about, above, after, again, against, all, am, an, and, any, are, aren't, as, at, be, because, been, before,\
    , being, below, between, both, but, by, can, can't, cannot, could, couldn't, did, didn't, do, does, doesn't,\
    , doing, don't, down, during, each, few, for, from, further, had, hadn't, has, hasn't, have, haven't, having, he,\
    , he'd, he'll, he's, her, here, here's, hers, herself, him, himself, his, how, how's, i, i'd, i'll, i'm, i've, if, in, into,\
    , is, isn't, it, it's, its, itself, let's, me, more, most, mustn't, my, myself, no, nor, not, of, off, on, once, only, or,\
    , other, ought, our, ours , ourselves, out, over, own, same, shan't, she, she'd, she'll, she's, should, shouldn't,\
    , so, some, such, than, that, that's, the, their, theirs, them, themselves, then, there, there's, these, they,\
    , they'd, they'll, they're, they've, this, those, through, to, too, under, until, up, very, was, wasn't, we, we'd,\
    , we'll, we're, we've, were, weren't, what, what's, when, when's, where, where's, which, while, who, who's,\
    , whom, why, why's, with, won't, would, wouldn't, you, you'd, you'll, you're, you've, your, yours, yourself,\
    , yourselves, n't, 're, 've, 'd, 's, 'll, 'm".replace(',','').split()
    nltk_stopwords = set(nltk.corpus.stopwords.words('english'))
    nltk_stopwords = nltk_stopwords | set(extra_stop_words)
    word_list = processing.split()
    word_list = [word for word in word_list if word.lower() not in nltk_stopwords]

    # Lemmatization
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_word_list = [lemmatizer.lemmatize(word) if word not in db_abbr_ls else word for word in word_list ]
    
    # Remove extra whitespaces
    processed = re.sub(r"[\s]{2,}", " ", " ".join(lemmatized_word_list)).strip()

    return processed


def _locate_abbr(content):
    """ Recognize the list of abbreviation and its sentence

    TODO: RNN to check if the next sentence is related to the prior sentence?
    Then group those sentence together to be vectorized as one vector for better
    context capturing?
    
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
        cleaned_sent = re.sub(r"[\.|\?|\!|\,]", "", sent)
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
    list
        The lsit fo the abbreviation found in the current content

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

            # Text preprocessing
            preprocessed = _preprocess(sent)
            cur_embed = MODEL.encode(preprocessed, show_progress_bar=False)

            similarities = cosine_similarity(
                [cur_embed],
                vector_ls
            ).flatten()

            abbr_fullform[i] = abbr_ls[np.argmax(similarities)+1]

    table = pd.DataFrame()
    table["Abbreviation"] = abbr_sent.keys()
    table["Fullform"] = abbr_fullform

    phrase_ls = []

    for abbr, sent in abbr_sent.items():
        replaced = f"‼️{abbr}‼️"
        text_around = re.sub(abbr, replaced, sent)

        word_idx = text_around.index(f"‼️{abbr}‼️")
        
        if word_idx != -1:
            if word_idx-30 < 0:
                phrase_ls.append(f"{text_around[:word_idx+len(replaced)+30]}…")
            else:
                phrase_ls.append(f"…{text_around[word_idx-30:word_idx+len(replaced)+30]}…")

    table["Sentence"] = phrase_ls

    return table, list(abbr_sent.keys())

