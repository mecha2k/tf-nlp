import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)

lemmatizer = WordNetLemmatizer()

words = [
    "policy",
    "doing",
    "organization",
    "have",
    "going",
    "love",
    "lives",
    "fly",
    "dies",
    "watched",
    "has",
    "starting",
]

print("표제어 추출 전 :", words)
print("표제어 추출 후 :", [lemmatizer.lemmatize(word) for word in words])

print(lemmatizer.lemmatize("dies", "v"))
print(lemmatizer.lemmatize("watched", "v"))
print(lemmatizer.lemmatize("has", "v"))


stemmer = PorterStemmer()

sentence = (
    "This was not the map we found in Billy Bones's chest, but an accurate copy, complete in all "
    "things--names and heights and soundings--with the single exception of the red crosses and the written notes."
)
tokenized_sentence = word_tokenize(sentence)

print("어간 추출 전 :", tokenized_sentence)
print("어간 추출 후 :", [stemmer.stem(word) for word in tokenized_sentence])

words = ["formalize", "allowance", "electricical"]
print("어간 추출 전 :", words)
print("어간 추출 후 :", [stemmer.stem(word) for word in words])

porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()

words = [
    "policy",
    "doing",
    "organization",
    "have",
    "going",
    "love",
    "lives",
    "fly",
    "dies",
    "watched",
    "has",
    "starting",
]
print("어간 추출 전 :", words)
print("포터 스테머의 어간 추출 후:", [porter_stemmer.stem(w) for w in words])
print("랭커스터 스테머의 어간 추출 후:", [lancaster_stemmer.stem(w) for w in words])
