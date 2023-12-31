import json
import string
import random 
import nltk
import numpy as num
from nltk.stem import WordNetLemmatizer 
import tensorflow as tensorF
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Dropout
nltk.download("punkt")
nltk.download("wordnet")

ourData = {"intents": [
            
            {"tag": "age",
              "patterns": ["How old are you?"],
              "responses": ["25"]
            },
            {"tag": "greeting",
              "patterns": [ "Hi!", "Sup?", "How's it going?", "Hey!", "Hello!", "How are you?", "Salve!", "What's up?"],
              "responses": ["I'm ok", "All quiet on the western front...", "I'm great :)", "Not bad, you?", "Not too bad..."],
             },
            {"tag": "banter",
              "patterns": [ "You mad?", "Cope", "Seethe", "Get gud", "Where is your god now, Christian?"],
              "responses": ["take this L", "ratio", "cope", "scrub!", "How are the Vestal Virgins doing?"]
            },
            {"tag": "goodbye",
              "patterns": [ "bye", "later", "take care", "see ya later", "ttyl", "Vale!", "Vale", "Take care!"],
              "responses": ["Bye, mistress", "take care, my lady", "bye, priestess!"]
            },
            {"tag": "name",
              "patterns": ["what's your name?", "who are you?"],
              "responses": ["My name is Rookie. I am a scrub in the employ of Domina Maurenus."]
            },
            {"tag": "ginna_inquiry",
              "patterns": ["Do you know who Ginna is?", "How is Ginna?"],
              "responses": ["Of course. She's a wonderful woman who hails from Mexico.", "Ask her, lol!"]
            },
            {"tag": "ginna_general",
              "patterns": ["Tell me about Ginna."],
              "responses": ["She's a dramaturgist who has the keys to Dublin.", "She's beautiful, but her mind is much more..."]
            }
            
]}

lm = WordNetLemmatizer() #for getting words
# lists
ourClasses = []
newWords = []
documentX = []
documentY = []
# Each intent is tokenized into words and the patterns and their associated tags are added to their respective lists.
for intent in ourData["intents"]:
    for pattern in intent["patterns"]:
        ourTokens = nltk.word_tokenize(pattern)
        newWords.extend(ourTokens)
        documentX.append(pattern)
        documentY.append(intent["tag"])
    
    
    if intent["tag"] not in ourClasses:# add unexisting tags to their respective classes
        ourClasses.append(intent["tag"])

newWords = [lm.lemmatize(word.lower()) for word in newWords if word not in string.punctuation] # set words to lowercase if not in punctuation
newWords = sorted(set(newWords))# sorting words
ourClasses = sorted(set(ourClasses))# sorting classes

trainingData = [] # training list array
outEmpty = [0] * len(ourClasses)
# BOW model
for idx, doc in enumerate(documentX):
    bagOfwords = []
    text = lm.lemmatize(doc.lower())
    for word in newWords:
        bagOfwords.append(1) if word in text else bagOfwords.append(0)
    
    outputRow = list(outEmpty)
    outputRow[ourClasses.index(documentY[idx])] = 1
    trainingData.append([bagOfwords, outputRow])

random.shuffle(trainingData)
trainingData = num.array(trainingData, dtype=object)# coverting our data into an array afterv shuffling

x = num.array(list(trainingData[:, 0]))# first trainig phase
y = num.array(list(trainingData[:, 1]))# second training phase

# defining some parameters
iShape = (len(x[0]),)
oShape = len(y[0])

# the deep learning model
ourNewModel = Sequential()
ourNewModel.add(Dense(128, input_shape=iShape, activation="relu"))
ourNewModel.add(Dropout(0.5))
ourNewModel.add(Dense(64, activation="relu"))
ourNewModel.add(Dropout(0.3))
ourNewModel.add(Dense(oShape, activation = "softmax"))
lr_schedule = tensorF.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9)
md = tensorF.keras.optimizers.Adam(learning_rate=lr_schedule)
ourNewModel.compile(loss='categorical_crossentropy',
              optimizer=md,
              metrics=["accuracy"])
print(ourNewModel.summary())
ourNewModel.fit(x, y, epochs=200, verbose=1)

def ourText(text): 
    newtkns = nltk.word_tokenize(text)
    newtkns = [lm.lemmatize(word) for word in newtkns]
    return newtkns

def wordBag(text, vocab): 
    newtkns = ourText(text)
    bagOwords = [0] * len(vocab)
    for w in newtkns: 
        for idx, word in enumerate(vocab):
            if word == w: 
                bagOwords[idx] = 1
    return num.array(bagOwords)

def pred_class(text, vocab, labels): 
    bagOwords = wordBag(text, vocab)
    ourResult = ourNewModel.predict(num.array([bagOwords]))[0]
    newThresh = 0.2
    yp = [[idx, res] for idx, res in enumerate(ourResult) if res > newThresh]

    yp.sort(key=lambda x: x[1], reverse=True)
    newList = []
    for r in yp:
        newList.append(labels[r[0]])
    return newList

def getRes(firstlist, fJson): 
    tag = firstlist[0]
    listOfIntents = fJson["intents"]
    for i in listOfIntents: 
        if i["tag"] == tag:
            ourResult = random.choice(i["responses"])
            break
    return ourResult

while True:
    newMessage = input("")
    if newMessage == "Exit program" or newMessage == "exit program":
        break
    intents = pred_class(newMessage, newWords, ourClasses)
    ourResult = getRes(intents, ourData)
    print("Rookie: " + ourResult)