import cv2
import re
import pytesseract
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
MODEL1 = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# MODEL2 = SentenceTransformer("bert-base-nli-mean-tokens")

# =========================================================================== #

def extract_text_from_doc(filename):
    """Extracting the text from the images / pdf uploaded by the user
    
    Parameters
    ----------
    filename : str
        The path of the temporary file uploaded by the users.
        Can be in the form of pdf or any kind of image extensions
    """
    content = ""

    # Checking if the file is pdf or image
    if filename.lower().endswith(".pdf"):
        content = " ".join(extract_text(filename).split())

    else:
        img = cv2.imread(filename, 0)
        extracted = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)["text"]
        content = re.sub(r"\s+", " ", extracted).strip()

    return content

def extract():
    # TODO: RUN THE MODEL HERE
    pass