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
