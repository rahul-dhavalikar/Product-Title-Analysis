from __future__ import unicode_literals
import csv
import gensim
import re
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

nlp = spacy.load('en')

tfidf_list = []

def loadWordVectors(word2VecFilePath):
    print("Loading Word Vectors............")
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(word2VecFilePath, binary=True)
    print("Loaded Word Vectors.............")
    return w2v_model

def contains_number(s):
    regex = re.compile("\d")
    if regex.search(s):
        return 1;
    return 0;

def containsColor(s):
    if "COLOR" in s:
        return 1
    else:
        return 0

def containsNew(s):
    if "new" in s:
        return 1
    else:
        return 0

def extract_features(filename,w2v_model, word2VecFlag = True, lengthAndBooleanFlag = True, tfidfLabelIntersectFlag = True,
                     numericFlag = True, denseEntropySKUFlag = True):
    index2word_set = set(w2v_model.index2word)
    build_tfidf()
    le1,le2,le3,le4=labelencoder()

    features = []
    with open(filename) as file:
        reader = csv.reader(file)
        for row in reader:

#             print("len(row)===========================");
            skuid = row[1]
            title = row[2]
            price = row[7]

            cat1 = row[3]
            cat2 = row[4]
            cat3 = row[5]
            ptype =row[8]
            current_feature1 = []
            if(lengthAndBooleanFlag):
                current_feature1 = [len(title), contains_number(title)]

            #word2vec
            data_word2vec = vector_title(title,w2v_model)
            
            if(word2VecFlag):
                current_feature1.extend(data_word2vec)

            if(lengthAndBooleanFlag):    
                # contains color
                current_feature1.extend([containsColor(title)])

                # has new
                current_feature1.extend([containsNew(title)])
            if(numericFlag):
                # price
                current_feature1.extend([price])

            if(tfidfLabelIntersectFlag): 
                #label_encoder for cat1,2,3,ptype
                current_feature1.extend(le1.transform([cat1]))
                current_feature1.extend(le2.transform([cat2]))
                current_feature1.extend(le3.transform([cat3]))
                current_feature1.extend(le4.transform([ptype]))

                for word in tfidf_list:
                    if word in title:
                        current_feature1.extend([1])
                        #tfidfCount = tfidfCount + 1
                    else:
                        current_feature1.extend([0])
            
                cur_title = title
                intersecting_words_list = list(set(cur_title.split(' ')) & set(tfidf_list))

    #           print("Computing interesecting list")
                current_feature1.extend([len(intersecting_words_list)])
    #           print("Computed Intersecting Feature")
    

            #number of repeating words
#             print("Computing number of repeating words")
            if(numericFlag):
                current_feature1.extend([num_of_repeating_words(title)])
#             print("Computed number of repeating words")
            
            #num of words
#             print("Computing number of words")
                current_feature1.extend([num_of_words(title)])
#             print("Computed number of words")
            
            #num of all and non digit words
#             print("Computing number of all digit and non digit words")
                current_feature1.extend([num_of_all_digit_words(title)])
                current_feature1.extend([num_of_non_digit_words(title)])
#             print("Computed number of all digit and non digit words")
            
            #contains measurement unit
#             print("Computing number measurement units")
            if(lengthAndBooleanFlag):
                current_feature1.extend([contains_measurement(title)])
#             print("Computed number measurement units")
            
            #get dense features
#             print("Computing dense features")
            if(denseEntropySKUFlag):
                current_feature1.extend(get_dense_features(title))
#             print("Computed dense features")
            
            #entropy of chars
                current_feature1.extend([entropy_chars(title)])

            #sku_id features
                sku_f1, sku_f2, sku_f3 = sku_id_features(skuid)
                current_feature1.extend(sku_f1)
                current_feature1.extend(sku_f2)
                current_feature1.extend(sku_f3)
            
            #append to features
            features.append(current_feature1)

    return np.asarray(features)


# Word2vec title
def vector_title(title, w2v_model):

    title = word_tokenize(title)
    #title = title.split(" ")
    vec = np.zeros(300)
    for w in title:
        try:
            vec += w2v_model[w]
        except:
            pass

    return vec

def build_tfidf():

    #print("TFIDF Compute********************")
    filename = "Data/training/data_train_clean.csv"
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(min_df=1)
    title = []
    with open(filename) as file:

        reader = csv.reader(file)
        for row in reader:
            title.append((str(row[2])))
    X = vectorizer.fit_transform(title)
    idf = vectorizer.idf_
    dd = dict(zip(vectorizer.get_feature_names(), idf))
    import operator
    sorted_x = sorted(dd.items(), key=operator.itemgetter(1))
    sorted_x = list((sorted_x))

    for i in range(0, 2700):
        #print(sorted_x[i][0], sorted_x[i][1])
        tfidf_list.append(sorted_x[i][0])

        
def labelencoder():
    filename = "Data/training/data_train_clean.csv"
    cat1=[]
    cat2=[]
    cat3=[]
    ptype=[]
    with open(filename) as file:
            reader = csv.reader(file)
            for row in reader:
                cat1.append((str(row[3])))
                cat2.append((str(row[4])))
                cat3.append((str(row[5])))
                ptype.append((str(row[8])))

    cat1=list(set(cat1))
    cat2=list(set(cat2))
    cat3=list(set(cat3))
    ptype=list(set(ptype))

    le1 = preprocessing.LabelEncoder()
    le2 = preprocessing.LabelEncoder()
    le3 = preprocessing.LabelEncoder()
    le4 = preprocessing.LabelEncoder()

    le1.fit(cat1)
    le2.fit(cat2)
    le3.fit(cat3)
    le4.fit(ptype)

    return le1,le2,le3,le4

def avg_feature_vector(sentence, model, num_features, index2word_set):
    words = sentence.split()
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

def find_word2vec_similarity(sentence1, sentence2, w2v_model,index2word_set):

    s1_afv = avg_feature_vector(sentence1, model=w2v_model, num_features=300, index2word_set=index2word_set)
    s2_afv = avg_feature_vector(sentence2, model=w2v_model, num_features=300, index2word_set=index2word_set)
    sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
    
    return sim
    
def num_of_repeating_words(title):
    title = title.split(' ')
    d = dict()
    count = 0
    for word in title:
        if word in d:
            count = count+1
        else:
            d[word] = 1

    return count    

def num_of_words(title):
    tlist = title.split(' ')
    return len(tlist)

def num_of_all_digit_words(title):
    tlist = title.split(' ')
    count = 0
    for w in tlist:
        if w.isdigit():
            count += 1

    return count    

def num_of_non_digit_words(title):
    tlist = title.split(' ')
    count = 0
    for w in tlist:
        if not w.isdigit():
            count += 1

    return count

def contains_measurement(title):
    mo=re.match(r'[^\d]*(\d+).*?(ml|cm|oz|gb|inch|inches|xl|kg|mg|pcs|pc|tb|m|ft|mah|a|vac)', title)
    
    if mo:
        return 1
    else:
        return 0
    
def get_dense_features(sent):
    """ratio: noun/nb_words, oov+stop, adj,num, avg char per word"""
    SAFEDIV=2e-6
    doc = nlp(sent.lower().decode('utf8'))
    nb_noun = 0
    nb_adj = 0
    nb_oov = 0
    nb_stop = 0
    nb_bracket = 0
    nb_nonalpha = 0
    nb_num = 0
    nb_upper = 0
    nb_nonascii = 0
    for i in doc:
        if i.pos_ == 'NOUN': nb_noun += 1
        if i.pos_ == 'ADJ': nb_adj += 1
        if i.is_oov: nb_oov += 1
        if i.is_stop: nb_stop += 1
        if i.is_bracket: nb_bracket += 1
        if not i.is_alpha: nb_nonalpha += 1
        if i.is_digit or i.like_num or i.pos_ == 'NUM': nb_num += 1
        if i.orth_.isupper(): nb_upper += 1
        if not i.is_ascii: nb_nonascii += 1
    nb_words = len(sent.split())
    nb_chars = len(sent.replace(' ',''))
    return (nb_words,nb_chars,nb_oov+nb_stop,nb_chars /(nb_words+SAFEDIV),nb_noun / (nb_words+SAFEDIV),nb_adj/(nb_words+SAFEDIV) ,                    (nb_oov+nb_stop)/(nb_words+SAFEDIV),(nb_bracket + nb_nonalpha + nb_nonascii)/(nb_words+SAFEDIV),nb_num/(nb_words+SAFEDIV))    

def entropy_chars(title):
    l_input = title.split(' ')

    if (len(l_input) < 3):
        return 0
    
    counter = CountVectorizer(ngram_range=(1,1), analyzer='word')
    X = counter.fit_transform(l_input)

    char_freq_df = pd.DataFrame({'word': counter.get_feature_names(), 'occurrences':np.asarray(X.sum(axis=0)).ravel().tolist()})
    count_dict_word = char_freq_df.set_index('word').to_dict()['occurrences']
    N_words = char_freq_df.occurrences.sum()

    res = entropy_chars_helper(l_input,count_dict=count_dict_word,n=N_words)    
    return res
    
def entropy_chars_helper(terms,count_dict,n):
    s = 0.0
    for word in terms:
        if word in count_dict:
            s+=-(1.*count_dict[word]/n)*np.log(1.*count_dict[word]/n)
           # print s;
    return s

def sku_id_features(sku_id):
    sku_id_list = list(sku_id)
    
    f = list()
    f1 = list()
    f2 = list()
    f3 = list()
    
    for ch in sku_id_list:
        f.append(ord(ch))
        
    f1 = f[1:6]
    f2 = f[6:8]
    f3 = f[8:10]
    
    return f1,f2,f3