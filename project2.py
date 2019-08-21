import numpy as np
import pandas as pd
import networkx as nx
import nltk
from nltk.corpus import stopwords
import re
df = pd.read_csv("tennis_articles_v4.csv")
from nltk.tokenize import sent_tokenize
sentences = []
for s in df['article_text']:
  sentences.append(sent_tokenize(s))

sentences = [y for x in sentences for y in x]
#print(sentences[:5])
word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()
#print(len(word_embeddings))
# remove punctuations, numbers and special characters
clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

# make alphabets lowercase
clean_sentences = [s.lower() for s in clean_sentences]
stop_words = stopwords.words('english')
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new
# remove stopwords from the sentences
clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()
sentence_vectors = []
for i in clean_sentences:
  if len(i) != 0:
    v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
  else:
    v = np.zeros((100,))
  sentence_vectors.append(v)
sim_mat = np.zeros([len(sentences), len(sentences)])
from sklearn.metrics.pairwise import cosine_similarity
for i in range(len(sentences)):
  for j in range(len(sentences)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)
ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
#Extract top 10 sentences as the summary
ind=1
summary=[]
for i in range(10):
  summary.append(ranked_sentences[i][1])
  ind+=1
text=' '.join([str(elem) for elem in summary]) 
def voice(mytext,lang):
  from gtts import gTTS
  import os
  language =lang
  myobj = gTTS(text=mytext, lang=language, slow=False) 
  myobj.save("project2.mp3")
  os.system("mpg321 project2.mp3")
from googletrans import Translator
translator = Translator()
te=translator.translate(text,dest='te')
hindi=translator.translate(text,dest='hi')
lang=int(input('''select languge
           for telugu-1
           for english-2
           for hindi-3
           '''))
if lang==1:
  print(te.text)
  voice(te.text,'te')
elif lang==2:
  print(text)
  voice(text,'en')
elif lang==3:
  print(hindi.text)
  voice(hindi.text,'hi')
