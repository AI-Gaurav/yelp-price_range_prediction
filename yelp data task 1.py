import json
import pandas as pd
from pandas.io.json import json_normalize 
from io import StringIO
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import nltk
import seaborn as sn
nltk.download('stopwords')
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense,Dropout,Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,precision_recall_fscore_support


review_json_path = 'data/yelp_academic_dataset_review.json'
business_json_path = 'data/yelp_academic_dataset_business.json'
transformer_model_path = "models/transformer_model_balanced_small"
lstm_model_path = "models/lstm_model_balanced_small"

def load_business_data(business_json_path):
    df  = pd.read_json(business_json_path,lines=True)
    df = df[df["attributes"].notnull()]
    df = pd.concat([df.drop(['attributes'], axis=1), pd.json_normalize(df['attributes'])], axis=1)
    df = df[df["business_id"].notnull()]
    df = df[df["RestaurantsPriceRange2"].notnull()]
    df = df[['business_id','RestaurantsPriceRange2']]
    df.rename(columns={'RestaurantsPriceRange2': 'PriceRange'}, inplace=True)
    df = df[df["PriceRange"]!='None']
    df.PriceRange = df.PriceRange.astype(float)
    return df

def load_reviews(review_json_path):
    reviews = []
    with open(review_json_path) as fl:
        for i, line in enumerate(fl):
            reviews.append(json.loads(line))
            if i+1 >= 100000:
                break
    df_reviews = pd.DataFrame(reviews)
    df_reviews = df_reviews[['business_id','text']]
    return df_reviews

df_business = load_business_data(business_json_path)
df_reviews = load_reviews(review_json_path)

df_reviews = pd.merge(df_business, df_reviews, on='business_id', how='inner')

classes = df_reviews['PriceRange'].unique()

"""#Preprocessing

"""

STOPWORDS = set(stopwords.words('english'))

def get_text(index):
    example = df_reviews[df_reviews.index == index][['text', 'PriceRange']].values[0]
    if len(example) > 0:
        print(example[0])
        print('PriceRange:', example[1])

def clean_text(text):
    text = ' '.join(word for word in text.lower().split() if word not in STOPWORDS) # remove stopwors from text
    return text

df_reviews['text'] = df_reviews['text'].apply(clean_text)

vocab_size = 50000

embed_dim = 100
tokenizer = Tokenizer(num_words=vocab_size, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df_reviews['text'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = tokenizer.texts_to_sequences(df_reviews['text'].values)

maxlen = 200  # Only consider the first 200 words of each review

X = keras.preprocessing.sequence.pad_sequences(X, maxlen=maxlen)

print('Shape of data tensor:', X.shape)

Y = df_reviews['PriceRange'].values
Y =Y-1
print('Shape of label tensor:', Y.shape)

x_train, x_val, y_train, y_val = train_test_split(X,Y, random_state = 4,stratify = Y) #75-25 split

y_train = tf.keras.utils.to_categorical(y_train,num_classes=4)
y_val = tf.keras.utils.to_categorical(y_val,num_classes=4)

def class_weights(y):
    weights = {}
    y= np.argmax(y,axis=1)
    cl_w = class_weight.compute_class_weight('balanced',
                                                 np.unique(y),
                                                 y)
    for i,label in enumerate(np.unique(y)):
        weights[int(label)] = cl_w[i]
    return weights

cls_weights = class_weights(y_train)
cls_weights

x_train.shape, x_val.shape, y_train.shape, y_val.shape

"""#transformer

"""

# from tensflow keras example
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# from tensflow keras example
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

embed_dim = 100  # Embedding size for each token
num_heads = 5  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(32, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(4, activation="softmax")(x)

model_transformer = keras.Model(inputs=inputs, outputs=outputs)

model_transformer.summary()

model_transformer.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
history = model_transformer.fit(
    x_train, y_train, batch_size=1024, epochs=40, validation_data=(x_val, y_val),class_weight=cls_weights
)

plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

model_transformer.save(transformer_model_path)

"""#Keras

"""

model_lstm = Sequential()
model_lstm.add(Embedding(vocab_size, embed_dim, input_length=X.shape[1]))
model_lstm.add(Dropout(0.2))
model_lstm.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2,return_sequences=True)))
model_lstm.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
model_lstm.add(Dense(64, activation='relu'))
model_lstm.add(Dense(y_train.shape[1], activation='softmax'))
model_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs = 20
batch_size = 1024
history_keras = model_lstm.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,validation_data=(x_val, y_val),callbacks=[EarlyStopping(monitor='val_accuracy', patience=7, min_delta=0.001,restore_best_weights=True)],class_weight=cls_weights)

model.summary()



plt.title('Loss')
plt.plot(history_keras.history['loss'], label='train')
plt.plot(history_keras.history['val_loss'], label='test')
plt.legend()
plt.show()

model_lstm.save(lstm_model_path)

"""#Load models

"""

model_lstm = tf.keras.models.load_model(lstm_model_path)
model_transformer = tf.keras.models.load_model(transformer_model_path)

"""#Evaluation Metrics"""

y_true = np.argmax(y_val,axis = 1)
ypred_transformer = np.argmax(model_transformer.predict(x_val),axis=1)
ypred_lstm = np.argmax(model_lstm.predict(x_val),axis=1)

def eval_metrics(y_true,ypred):
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true,ypred,average ='weighted')
    conf_matrix = confusion_matrix(y_true,ypred,normalize='true')
    accuracy = accuracy_score(y_true,ypred)
    return accuracy,precision, recall, fscore, conf_matrix

accuracy_transformer,precision_transformer, recall_transformer, fscore_transformer, conf_matrix_transformer = eval_metrics(y_true,ypred_transformer)
accuracy_transformer,precision_transformer, recall_transformer, fscore_transformer, conf_matrix_transformer

cm_transformer = pd.DataFrame(conf_matrix_transformer,columns=[1,2,3,4],index=[1,2,3,4])

sn.heatmap(cm_transformer,annot=True,cmap='Blues')

accuracy_lstm, precision_lstm, recall_lstm, fscore_lstm, conf_matrix_lstm = eval_metrics(y_true,ypred_lstm)
accuracy_lstm, precision_lstm, recall_lstm, fscore_lstm, conf_matrix_lstm

cm_lstm = pd.DataFrame(conf_matrix_lstm,columns=[1,2,3,4],index=[1,2,3,4])

sn.heatmap(cm_lstm,annot=True,cmap='Blues')



"""#Prediction"""

def predict(text,model):
    new_review = [text]
    seq = tokenizer.texts_to_sequences(new_review)
    padded = pad_sequences(seq, maxlen=maxlen)
    pred = model.predict(padded)
    labels = [1,2,3,4]
    return labels[np.argmax(pred)]

review_example = "first time oasis, moved street. usually go brake masters, referred friend. dropped car oil change told would done within hours tops, ended taking half time! manager rusty helped dropped picked up, completely wonderful whole time, nice guy. cost completely fair, rusty went additional maintenance would due future, along price quotes each. far best customer service i've ever auto shop! guys really go beyond take care clients."

predict(review_example,model_lstm)