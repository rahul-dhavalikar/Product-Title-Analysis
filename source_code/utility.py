# coding=utf-8
import re
import webcolors
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import wordnet
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

stop_words = set(stopwords.words('english'))

num_re = re.compile(r'[\.\+\-\$\u20ac]?\d+((\.\d+)|(,\d+))*(st|nd|rd|th)?', re.I | re.UNICODE)
size_re = re.compile(r'[\.\+\-\$\u20ac]?\d+((\.\d+)|(,\d+)|(x\d+))*(ml|cm|oz|gb|inch|inches|xl|kg|mg|pcs|pc|tb|m|ft|mah|a|vac)', re.I | re.UNICODE)

colorList = webcolors.CSS3_NAMES_TO_HEX.keys()
# print len(colorList), colorList
colorspattern = '|'.join(colorList)
colorregex = re.compile(colorspattern)

toker = TreebankWordTokenizer()
lemmer = wordnet.WordNetLemmatizer()

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    return cleantext

def cleanText(inText,removeStopWord = True):
    inText = inText.lower()
    inText = inText.replace("/", " / ")
    inText = re.sub(r"[^A-z0-9\s-]", "", inText)
    inText = textReplaceNum(textReplaceSize(inText))
    if removeStopWord:
        inText = removeStopWords(inText)
    # replace all color words
    inText = replaceColors(inText)
    tokens = toker.tokenize(inText)
    inText = " ".join([lemmer.lemmatize(z) for z in tokens])

    return inText

def removeStopWords(inText):

    word_tokens = word_tokenize(inText)
    #filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []

    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    result = " ".join(filtered_sentence)
    return result

def cleanDescription(raw_description):
    cleanOutput = cleanhtml(raw_description)
    cleanOutput = " ".join(cleanOutput.split())
    cleanOutput = cleanText(cleanOutput)

    return cleanOutput

def normalizePrice(price, country):
    if country == 'sg':
        return price
    elif country == 'my':
        return price*0.32
    else:
        return price*0.027

def textReplaceSize(s):
    """
    100ml -> 100 ml
    """
    s = str(s)
    s = u' '.join([u' '.join(re.findall(r'[A-Za-z]+|\d+', w)) if size_re.match(w) else w for w in s.strip().split()])
    return s

def textReplaceNum(s):
    """
    123 -> 0
    """
    s = str(s)
    s = re.sub(r'(\w+)-(\w+)', r'\1 - \2', s) # fixed - by digit and named
    s = u' '.join(["0" if num_re.match(w) else w for w in s.strip().split()])
    return s


def replaceColors(inText):
    token = "COLOR"
    return colorregex.sub(token, inText)

def splitTrainTest(inputData, dataLabels, testSizeRatio, randomState = None):
    trainingData , testingData , trainingLabels, testingLabels = train_test_split(inputData,dataLabels, test_size = testSizeRatio , random_state = randomState)
    return trainingData, testingData, trainingLabels, testingLabels

def splitKFold(inputData, dataLabels, K , shuffleData = True , randomState = None):
    kf = KFold(n_splits=K, shuffle = shuffleData, random_state = randomState)
    result = kf.split(inputData,dataLabels)
    return result
