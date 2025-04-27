#Bu kod, kullanıcı etkileşimini sağlayan bir Streamlit uygulamasıdır. Kullanıcı bir resim yüklediğinde, uygulama resmi işler, model ile tahmin yapar ve sonuç olarak sıtma hastalığı olup olmadığını (enfekte veya enfekte olmamış) gösterir. 
import streamlit as st   #Web uygulaması geliştirmek için kullanılır.
from tensorflow.keras.models import load_model # type: ignore  #Keras: Makine öğrenimi modelini yüklemek için kullanılır.
from PIL import Image    #Resimleri işlemek için kullanılır.
import numpy as np       #Sayısal işlemler için kullanılır.

model=load_model("malaria.h5")   #Eğitilmiş sıtma modelini malaria.h5 dosyasından yükler. Bu model, resimleri sınıflandırmak için kullanılır.

def process_image(img):
        img=img.resize((128,128))
        img=np.array(img)
        img=img/255.0
        img=np.expand_dims(img,axis=0) # 
        return img
 

st.title("Sıtma Hastalığı Tahmin:palm_tree:")  #Uygulamanın başlığı ve kullanıcıya bilgi veren bir açıklama ekler.
st.write("Resim sec ve model hastalık olup olmadığını tahmin etsin")

file=st.file_uploader('Bir resim sec', type=['jpg','jpeg','png'])  #Kullanıcının JPEG veya PNG formatında bir resim yüklemesine olanak tanır. Yüklenen dosya file değişkenine atanır.

if file is not None:
    img=Image.open(file) #Açılan resim, process_image fonksiyonu kullanılarak işlenir ve image değişkenine atanır. Bu, modelin girdi formatına uygun hale getirir.

    st.image(img,caption='yuklenen resim')
    image=process_image(img)
    prediction=model.predict(image) #model.predict(image) ifadesi, işlenmiş resmi modelin tahmin fonksiyonuna gönderir. Model, resmi analiz eder ve her bir sınıf için tahmin edilen olasılıkları içeren bir dizi döndürür.
# np.argmax(prediction), bu dizideki en yüksek değerin indeksini bulur ve predicted_class değişkenine atar. Bu, modelin en yüksek olasılıkla tahmin ettiği sınıfı temsil eder.
    predicted_class=np.argmax(prediction)
    class_names=['uninfected','parasitized']
    st.write(class_names[predicted_class])
    #class_names listesi, modelin tahmin ettiği sınıfların isimlerini içerir.
    #st.write(class_names[predicted_class]), modelin tahmin ettiği sınıfın ismini Streamlit uygulamasında gösterir.