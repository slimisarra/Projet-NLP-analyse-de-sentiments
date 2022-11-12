import pickle
import nltk
import random
import contractions
import spacy
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from textblob import TextBlob
import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

DATASET_FILE = "./dataset_cleaned.csv"
dataset_df = pd.read_csv(DATASET_FILE)
Negative_data = dataset_df[(dataset_df['stars'] == 1) | (dataset_df['stars'] == 2)]
data = Negative_data.text_cleaned
with (open('./Vectorizer.pkl', "rb")) as f:
    while True:
        try:
           Vect= pickle.load(f)
        except EOFError:
            break


with (open('./modelNmf.pkl', "rb")) as f:
    while True:
        try:
            Model=pickle.load(f)
        except EOFError:
            break

tokenizer = RegexpTokenizer(r'\w+')

def tokenize_text(text):
    text_processed = " ".join(tokenizer.tokenize(text))
    return text_processed

#nlp = en_core_web_sm.load(disable=['parser', 'tagger', 'ner'])

lemmatizer = WordNetLemmatizer()


def lemmatize_text(text):
    tokens_tagged = nltk.pos_tag(nltk.word_tokenize(text))
    lemmatized_text_list = list()

    for word, tag in tokens_tagged:
        if tag.startswith('J'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word,
                                                             'a'))  # Lemmatise adjectives. Not doing anything since we remove all adjective
        elif tag.startswith('V'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word, 'v'))  # Lemmatise verbs
        elif tag.startswith('N'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word, 'n'))  # Lemmatise nouns
        elif tag.startswith('R'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word, 'r'))  # Lemmatise adverbs
        else:
            lemmatized_text_list.append(
                lemmatizer.lemmatize(word))  # If no tags has been found, perform a non specific lemmatisation

    return " ".join(lemmatized_text_list)

def normalize_text(text):
    return " ".join([word.lower() for word in text.split()])

def contraction_text(text):
    return contractions.fix(text)

negative_words = ['not', 'no', 'never', 'nor', 'hardly', 'barely']
negative_prefix = "NOT_"


def get_negative_token(text):
    tokens = text.split()
    negative_idx = [i + 1 for i in range(len(tokens) - 1) if tokens[i] in negative_words]
    for idx in negative_idx:
        if idx < len(tokens):
            tokens[idx] = negative_prefix + tokens[idx]

    tokens = [token for i, token in enumerate(tokens) if i + 1 not in negative_idx]

    return " ".join(tokens)


from spacy.lang.en.stop_words import STOP_WORDS


def remove_stopwords(text):
    english_stopwords = stopwords.words("english") + list(STOP_WORDS) + ["tell", "restaurant"]

    return " ".join([word for word in text.split() if word not in english_stopwords])


def preprocess_text(text):
    # Tokenize review
    text = tokenize_text(text)

    # Lemmatize review
    text = lemmatize_text(text)

    # Normalize review
    text = normalize_text(text)

    # Remove contractions
    text = contraction_text(text)

    # Get negative tokens
    text = get_negative_token(text)

    # Remove stopwords
    text = remove_stopwords(text)

    return text

output=[]
out=[]

topics=mon_dictionnaire = {0: "le lieu/la sécurité", 1: "qualité des plats" , 2: "qualiré des pizzas" , 3: "problémes des commandes en ligne / livraison" , 4: "qualité des plats et de service" , 5: "qualité de service / les serveurs" , 6: "qualité des burgers" , 7: "Temps d'attente" , 8: "la qualité de poulet" , 9: "le bar / qualité des boissons" , 10: "Répétition de mauvaises expériences" , 11: "le service / l'équipe de travail" , 12: "la qualité des sandwichs" , 13: "la qualité des sushi" ,14: "la qualité de service"}
def Predict(Vect,Model,text,n):
  Preprocessed_text=[preprocess_text(text)]
  blob = TextBlob(text)
  polarite=blob.polarity
  if polarite>0:
    print("le commentaire est positif:" , polarite )
    return None, polarite
  else :
    print("le commentaire est négatif:" , polarite)
    doc_topic = Vect.transform(Preprocessed_text)
    result= Model.transform(doc_topic)
    highest_indices = np.argsort(-1*result)[:n]
    for i in range(n) :
     output.append(highest_indices[0][i])

    list=[]
    for i in output :
      list.append(topics[i])

    topics_polarity=[]
    topics_polarity.append(list)
    topics_polarity.append(polarite)
    return topics_polarity




def main():

    st.sidebar.header("Review analyser")
    type = st.sidebar.radio("Vous aimez écrire votre avis ou générer un avis aléatoire de dataset ?", ('Dataset', 'Avis'))

    image = Image.open('./frai-03-00042-g002.jpg')
    st.image(image, caption='Topic Modeling')

    if type == "Dataset":
        dataset = random.choices(random.choices(data, k=1))
        st.sidebar.info(dataset)
        string = ' '.join([str(item) for item in dataset])
        text = st.sidebar.text_area(label="Vous avez choisi un avis de la dataset 👇", value=string,
                                    disabled=False, label_visibility="visible")

    if type == 'Avis':
        text = st.sidebar.text_area(label="Veuillez saisir ici votre avis 👇", value="Entrez ...", disabled=False, label_visibility="visible")

    n=st.slider("choisir le nombre de topics",1,15)


    #Button
    if st.button('Submit'):
        result = Predict(Vect,Model,text,n)
        if result[1] > 0:
            st.error("POLARITE : " + f'{result[1]}' + "  (AVIS POSITIF)")
        if result[1] < 0:
            st.error("POLARITE : " + f'{result[1]}' + "  (AVIS NEGATIF)")

        st.error("TOPICS : "+f'{result[0]}', icon="🚨")



if __name__ == '__main__':
    main()
