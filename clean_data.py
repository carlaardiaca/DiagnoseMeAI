import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Descargar stopwords si no están descargadas
nltk.download('stopwords')

# Definir stopwords y contracciones
stop_words = (stopwords.words('english'))

contraccions = {"i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is", "it's": "it is", "we're": "we are", "they're": "they are", "i've": "i have",
                "you've": "you have", "we've": "we have", "they've": "they have", "i'm": "i am", "i'd": "i would", "you'd": "you would", "he'd": "he would",
                "she'd": "she would", "we'd": "we would", "they'd": "they would", "i'll": "i will", "you'll": "you will", "he'll": "he will", "she'll": "she will",
                "we'll": "we will", "they'll": "they will", "don't": "do not", "can't": "can not", "cannot": "can not"
                }

# Inicializar el stemmer                
stemmer = PorterStemmer()

# Funciones de preprocesamiento
def steeming(text):
    """
    Apply stemming to a given text, reducing words to their root form.
    """
    return stemmer.stem(text)
    
def convert_to_lowercase(text):
    """
    Convert a given text to lowercase to ensure uniformity.
    """
    if isinstance(text, str):
      return text.lower()
    else:
       return text.str.lower()

def remove_contractions(text):
    """
    Replace contractions in the text with their expanded forms to aid in tokenization.
    """
    for paraula in contraccions.keys():
        text = text.replace(paraula, contraccions[paraula])
    return text
    
def remove_punctuation(text):
    """
    Remove punctuation from the text as it usually does not contribute to the meaning of words.
    """
    result = text.translate(str.maketrans('', '', string.punctuation))
    return result

def remove_stopwords(text):
    """
    Filter out stopwords from the text. These words are common and do not carry important meaning.
    """
    if not isinstance(text, str):
            return text
    paraules = text.split()
    paraules_bones = []
    for par in paraules:
        if par not in stop_words:
            paraules_bones.append(par)
    return ' '.join(paraules_bones)
  
  
def clean_text(text):
    """
    Perform a series of preprocessing steps on the text: stemming, converting to lowercase,
    removing contractions, punctuation, and stopwords.
    """
    if not isinstance(text, str):
        return text
    text = steeming(text)
    text = convert_to_lowercase(text)
    text = remove_contractions(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    return text