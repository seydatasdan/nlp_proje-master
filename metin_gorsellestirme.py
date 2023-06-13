import pandas as pd
import re
import snowballstemmer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


df = pd.read_csv("data/nlp_proje_metin_gorsellestirme.csv",index_col=0)
df

df.head()

# pip install snowballstemmer


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


def remove_hashtag(value):
    if not isinstance(value, (str, bytes)):
        raise TypeError("value parametresi bir dize veya bayt benzeri nesne olmalıdır.")
    return re.sub(r'#[^\s]+', '', value)


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


#kök indirgeme ve stop words işlemleri
import snowballstemmer
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
    return [(remove_emoji(remove_link
                          (remove_numeric
                           (remove_hashtag
                            (remove_noktalama
                             (remove_single_character
                              (remove_username
                               (stem_word(word))))))))) for word in value.split()]




# Adım 2 Word2Vec modeli ile eğitme işlemleri

from gensim.models import Word2Vec

sentences = df["Text"].dropna().apply(pre_processing).tolist()  #NaN değerleri kaldırmak için dropna kullanıldı
word2vec_model = Word2Vec(sentences, min_count=1)   #Word2Vec modelini eğitme


word = "ama"  # Analiz etmek istenilen kelime

# Kelimenin en yakın 10 kelimesini bul
similar_words = word2vec_model.wv.most_similar(word, topn=10)
print(similar_words)


# Adım 3 TSNE algoritması kullanılarak görselleştirme işlemi

# Kelimenin vektörünü alma
word_vector = word2vec_model.wv[word]

# En yakın 10 kelimenin vektörlerini alma
similar_word_vectors = [word2vec_model.wv[word] for word, _ in similar_words]

# Tüm vektörleri birleştirme
vectors = np.vstack([word_vector] + similar_word_vectors)


# TSNE uygulama
perplexity_value = min(vectors.shape[0] // 2, 50)  # Veri kümenizdeki kelime sayısının yarısı veya maksimum 50
tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
tsne_vectors = tsne.fit_transform(vectors)

print("Vektörlerin Şekli:", vectors.shape)
print("Perplexity Değeri:", perplexity_value)


# Adım 2 ?
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

kelimeler = ['merhaba', 'dünya', 'python', 'programlama', 'öğreniyorum', 'veri', 'analizi', 'yapmak', 'iş', 'geliştirici',
            'kitap', 'yazılım', 'proje', 'ödev', 'sınıf', 'öğrenci', 'okul', 'üniversite', 'öğretmen', 'öğrenmek',
            'kodlama', 'hata', 'debug', 'algoritma', 'fonksiyon', 'değişken', 'liste', 'sözlük', 'döngü', 'koşul',
            'modül', 'paket', 'dosya', 'okuma', 'yazma', 'yapı', 'metot', 'nesne', 'string', 'integer', 'float',
            'boolean', 'tuple', 'index', 'karakter', 'liste', 'sözlük', 'hafıza', 'performans']

frekans = {word: np.random.randint(1, 100) for word in kelimeler}

# WordCloud oluşturun
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(frekans)

# WordCloud'u görselleştirin
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

