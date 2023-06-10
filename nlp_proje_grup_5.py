import pandas as pd
import re
import snowballstemmer

df = pd.read_csv("data/nlp_proje_metin_siniflandirma.csv",index_col=0)

df.head()
df.tail()

# emojilerin kaldırılması
def remove_emoji(value):
    bfr=re.compile("[\U00010000-\U0010ffff]",flags=re.UNICODE)
    bfr=bfr.sub(r'',value)
    return bfr

#linklerin kaldırılması 
def remove_link(value):
    return re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',value)

# numerik karakterlerin kaldırılması
def remove_numeric(value):
    bfr= [item for item in value if not item.isdigit()]
    bfr = "".join(bfr)
    return bfr 

# hashtaglerin kaldırılması
def remove_hashtag(value):
    return re.sub(r'#[^\s]+','',value)

#noktalama işaretlerinin kaldırılması
def remove_noktalama(value):
    return re.sub(r'[^\w\s]','',value) 

#tek karakterli ifadelerin kaldırılması
def remove_single_character(value):
    return re.sub(r'(?:^| )\w(?:$| )','',value)

# kullanıcı adlarının kaldırılması
def remove_username(value):
    return re.sub('@[^\s]+','',value)

# Boşlukların kaldırılması
def remove_space(value):
    return[item for item in value if item.strip()]

#kök indirgeme ve stop words işlemleri
def stem_word(value):
    stemmer= snowballstemmer.stemmer("turkish")
    value = value.lower()
    value=stemmer.stemWords(value.split())
    stop_words= ['bence','a','acaba','altı','altmış','ama','ancak','arada','artık','asla','aslında','ayrıca','az',
                 'bana','bazen','bazı','bile','biraz','bu','bunu','bunun','çoğu','çoğunu','çok','çünkü',
                 'da','daha','de','ise','defa','diye','gibi','en','kim','mı','mi','mu','mü','bir','iki',
                 'üç','dört','beş','altı','yedi','sekiz','dokuz','on','niçin','niye','şey','siz','şu',
                'her','hiç','ve','veya','ya','yani','ne','neden']
    value= [item for item in value if not item in stop_words]
    value=' ' .join(value)
    return value

# ön işlem fonksiyonlarının sırayla çağırılması
def pre_processing(value):
    return [remove_numeric(remove_emoji
                          (remove_single_character
                           (remove_noktalama
                            (remove_link
                             (remove_hashtag
                              (remove_username
                               (stem_word(word)))))))) for word in value.split()]

# Boşlukların kaldırılması
def remove_space(value):
    return[item for item in value if item.strip()]

#Tanımlamalar
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import pandas as pd

df= pd.read_csv("data/nlp_proje_metin_siniflandirma.csv",index_col=0)
df.head()

df["Text"] = df["Text"].astype(str)

import on_islem

df["Text_2"]=df["Text"].apply(on_islem.pre_processing)
df["Text_2"]=df["Text_2"].apply(on_islem.remove_space)
df.head()

#boş liste kontrolü
df[df["Text_2"].str[0].isnull()]

df_index = df[df["Text_2"].str[0].isnull()].index
df = df.drop(df_index)
df = df.reset_index()
del df["index"]
df[df["Text_2"].str[0].isnull()]

df["Text_2"]

df["Text_3"] = [' '.join(w for w in item) for item in df["Text_2"]]
df

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Text_3"].tolist())

print(X.toarray())
set(X.toarray()[0])

df["Text_4"] = X.toarray().tolist()
df

len(X.toarray()[0])

svm= Pipeline([('vect',TfidfVectorizer()),('svm',LinearSVC())])

msg_train,msg_test,label_train,label_test,= train_test_split(df["Text_3"].tolist(),df["Label"].tolist(),test_size=0.2,random_state= 42)

svm.fit(msg_train,label_train)
y_pred_class = svm.predict(msg_test)

df.head()

len(msg_train)
len(msg_test)
len(label_train)

#KNN Algoritması uygulama
from sklearn.neighbors import KNeighborsClassifier

msg_train,msg_test,label_train,label_test,= train_test_split(df["Text_3"].tolist(),df["Label"].tolist(),test_size=0.2,random_state= 42)
knn = Pipeline([('vect',TfidfVectorizer()),('knn',KNeighborsClassifier())]) 
knn.fit(msg_train,label_train)
y_pred_class = knn.predict(msg_test)

print("knn accuracy score:",accuracy_score(label_test,y_pred_class))
print("knn f1 score:",f1_score(label_test,y_pred_class,average="weighted"))

len(msg_test)

#600 taneden kaç tanesini bilmiş
600*0.48 

cm=confusion_matrix(label_test,y_pred_class,labels=svm.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=svm.classes_)
disp.plot() 


# BURAYA KADAR ÖN İŞLEM ADIMLARI VE TF-IDF İLE KNN ALGORİTMASI YAPILDI (METİN SINIFLANDIRMA ADIM 1 VE ADIM 2)



# Adım3 Word2vec 

from gensim.models import Word2Vec
import numpy as np

# Word2Vec modelini eğitme
word2vec_model = Word2Vec(df["Text_2"], min_count=1)


# Metinlerin Word2Vec gömme vektörlerini hesaplama
def get_sentence_embedding(sentence):
    sentence_vectors = []
    for word in sentence:
        if word in word2vec_model.wv:
            sentence_vectors.append(word2vec_model.wv[word])
    if len(sentence_vectors) == 0:
        return np.zeros(100)  # Eğer tüm kelimeler Word2Vec modelinde yoksa sıfır vektör döndürülür
    return np.mean(sentence_vectors, axis=0)

df["Text_5"] = df["Text_2"].apply(get_sentence_embedding)



# Word2Vec gömme vektörleriyle KNN algoritmasını kullanma
msg_train, msg_test, label_train, label_test = train_test_split(df["Text_5"].tolist(), df["Label"].tolist(), test_size=0.2, random_state=42)
knn_word2vec = KNeighborsClassifier()
knn_word2vec.fit(msg_train, label_train)
y_pred_class_word2vec = knn_word2vec.predict(msg_test)

print("KNN ile Word2Vec accuracy score:", accuracy_score(label_test, y_pred_class_word2vec))
print("KNN ile Word2Vec F1 score:", f1_score(label_test, y_pred_class_word2vec, average="weighted"))


# Confusion Matrix
cm_word2vec = confusion_matrix(label_test, y_pred_class_word2vec, labels=knn_word2vec.classes_)
disp_word2vec = ConfusionMatrixDisplay(confusion_matrix=cm_word2vec, display_labels=knn_word2vec.classes_)
disp_word2vec.plot()

