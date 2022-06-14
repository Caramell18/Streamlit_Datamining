from dis import dis
import imp
from pyparsing import And
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.title('------Aplikasi Projek Data Mining------')
st.markdown ('* **Nama Kolom harus sesuai pada dataset**')
st.markdown ('* **Web Ini Untuk Menghitung Akurasi**')
st.markdown ('* **Untuk saat ini web ini hanya bisa menggunakan dataset yang disediakan**')

#========================= Input Dataset ==========================
uploaded_file = st.file_uploader("Masukan Dataset")
#=======================================================================

#============================= Inputan ===================================

if uploaded_file is not None :    
    df = pd.read_csv(uploaded_file)
    st.write(df)

    sel_col,displ_col = st.columns(2)
    displ_col.write('Nama Kolom :')
    displ_col.write(df.columns)
    input_tabel1 = sel_col.text_input('Masukan kolom Chart pertama','tahun')
    input_tabel2 = sel_col.text_input('Masukan kolom Chart kedua','bulan')
    chart_select = sel_col.selectbox('Pilih bentuk Chart :', ('Bar Chart', 'Line Chart'))
    if input_tabel1 or input_tabel2 and uploaded_file  is not None :
        st.subheader('Berikut data berdasarkan kota dari data di atas :')
        chart_var_dist = pd.DataFrame(np.random.randn(20,2),columns=[input_tabel1,input_tabel2])
        # chart_var_dist = pd.DataFrame(df[input_tabel].value_counts())
        if chart_select == 'Bar Chart':
            st.bar_chart(chart_var_dist)
        elif chart_select == 'Line Chart' :
            st.line_chart(chart_var_dist)

        #========================PARAMETER============================    
        sel_col,displ_col = st.columns(2)
        params = dict()
        K = sel_col.slider('Parameter K', 1,100)
        params['K'] = K
        p = displ_col.slider('Parameter P', 1,100)
        params['p'] = p
        metric = st.selectbox(
        "pilih Metric",
        ('minkowski', 'euclidean')
        )

        dataset = df
        x = dataset.iloc[:, [1,5]].values
        y = dataset.iloc[:, -1].values
                
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        classifier = KNeighborsClassifier(n_neighbors = params['K'], metric = metric,p = params['p'])
        classifier.fit(x_train,y_train)
        y_pred = classifier.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)

        acc = accuracy_score(y_test,y_pred)
        
        st.write(f'Akurasi = ',acc)
        
        #===============================Tabel Pyplot===============================
        st.write('Berikut Tabel nya :')
        sel_col,displ_col = st.columns(2)
        pca = PCA(2)
        x_projected = pca.fit_transform(x)
        x1 = x_projected[:,0]
        x2 = x_projected[:,1]
        fig = plt.figure()
        plt.scatter(x1,x2,c=y,alpha=0.8, cmap='viridis')
        plt.colorbar()
        sel_col.pyplot(fig)
