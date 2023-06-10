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
