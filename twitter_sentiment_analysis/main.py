# %%
import pandas as pd
# %%
data = pd.read_csv("Tweets.csv")
# %%
data.info()
# %%
data.dropna(inplace=True)
data.head()
# %%
data.drop("textID", axis=1, inplace=True)
# %%
data["sentiment"].replace(["neutral", "negative"],"0", inplace=True)
data["sentiment"].replace(["positive"],"1", inplace=True)
# %%
# Hatalı değerleri düzelttikten sonra veri türünü değiştirme
data["sentiment"] = pd.to_numeric(data["sentiment"], errors='coerce', downcast='integer')
# %%
data.head()
# %%
from sklearn.model_selection import train_test_split

X=data["text"]
y=data["sentiment"]

X_train,X_test,y_train,y_test = train_test_split(X,y, random_state=16)


# %%
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense,Embedding,LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
# %%
tokenizer = Tokenizer()

tokenizer.fit_on_texts(X_train)
# %%
# X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
# %%
from tensorflow.keras.utils import pad_sequences
# %%
num_words = len(tokenizer.word_index) + 1
# print(num_words)

X_train_pad = pad_sequences(X_train, maxlen=100)
X_test_pad = pad_sequences(X_test, maxlen=100)
# %%
model = Sequential()
# %%
# Gömme katmanı (Embedding layer)
# Bu katman, her kelimenin bir vektör temsilini oluşturur.
model.add(Embedding(input_dim=num_words, output_dim=64, input_length=100))

# LSTM katmanları
# İlk LSTM katmanı, return_sequences=True ile konfigüre edilmiştir.
# Bu, diğer LSTM katmanlarından (eğer varsa) önceki LSTM katmanının tüm zaman adımlarının çıkışını döndürmesini sağlar.
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64))

# Tam bağlı (fully connected) gizli katman
model.add(Dense(64, activation='relu'))

# Çıkış katmanı
model.add(Dense(1, activation='sigmoid'))

# Modeli derleme
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model özeti
model.summary()
# %%
callbackss = EarlyStopping(monitor="val_loss", patience=35)

model.fit(X_train_pad,y_train,validation_split=0.2,epochs = 100, batch_size=156,callbacks=callbackss)
# %%
model.save("twitter_model.h5")
# %%
y_pred = model.predict(X_test_pad)
# %%
y_pred
# %%
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Modeli yükleyin (bu, daha önce eğittiğiniz modelin dosya yolunu göstermelidir)
model = load_model('twitter_model.h5')  # 'twitter_model.h5' kısmını kendi modelinizin dosya yoluna güncelleyin

cumle = input("::")
# Yeni bir tokenizer oluşturun
tokenizer = Tokenizer()
tokenizer.fit_on_texts([cumle])

# Metni tokenize etme
sequences = tokenizer.texts_to_sequences([cumle])

# Pad etme
max_len = 100  # Bu, modelin eğitildiği pad_sequences ile aynı olmalı
padded_sequence = pad_sequences(sequences, maxlen=max_len)

# Modeli kullanarak tahmin yapma
prediction = model.predict(padded_sequence)

# Tahminin pozitif mi negatif mi olduğunu kontrol etme
if prediction[0][0] >= 0.5:
    print("Positive")
else:
    print("Negative")
# %%
