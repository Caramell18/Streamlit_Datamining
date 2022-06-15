from dis import dis
import imp
from pyparsing import And
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import streamlit as st
import numpy as np
import pandas as pd

st.sidebar.header('Menu')
page = st.sidebar.selectbox("Pilih Page Dataset",('KNN','Random Forest'))

if page == 'KNN':
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
        input_tabel = sel_col.text_input('Masukan kolom Chart pertama','VendorID')
        chart_select = sel_col.selectbox('Pilih bentuk Chart :', ('Bar Chart', 'Line Chart'))
        
        if input_tabel and uploaded_file  is not None :
            st.subheader('Berikut data berdasarkan kota dari data di atas :')
            # chart_var_dist = pd.DataFrame(np.random.randn(20,2),columns=[input_tabel1,input_tabel2])
            chart_var_dist = pd.DataFrame(df[input_tabel].value_counts()).head(30)
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
            x = dataset.iloc[:, [3,5]].values
            y = dataset.iloc[:, [8]].values
                    
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

            from mitosheet import *; register_analysis("id-rltampzvvc");
    
            # Imported taxi_data.csv
            import pandas as pd
            taxi_data = pd.read_csv(r'E:\datamining\Datamining_Tugas\data\taxi_data.csv')

            # Filtered passenger_count
            taxi_data = taxi_data[taxi_data['passenger_count'].notnull()]

            # Filtered RatecodeID
            taxi_data = taxi_data[taxi_data['RatecodeID'].notnull()]

            # Filtered DOLocationID
            taxi_data = taxi_data[taxi_data['DOLocationID'].notnull()]

            import plotly.express as px

            # Filter the dataframe so that it does not crash the browser
            taxi_data_filtered = taxi_data.head(1000)

            # Construct the graph and style it. Further customize your graph by editing this code.
            # See Plotly Documentation for help: https://plotly.com/python/plotly-express/
            fig = px.histogram(taxi_data_filtered, x=['passenger_count', 'RatecodeID'], y='DOLocationID')
            fig.update_layout(
                title='passenger_count, RatecodeID, DOLocationID (first 1000 rows) histogram', 
                xaxis = dict(
                    rangeslider = dict(
                        visible=True, 
                        thickness=0.05
                    )
                ), 
                yaxis = dict(

                ), 
                barmode='group', 
                paper_bgcolor='#0a0a0a', 
                showlegend=True
            )
            fig.show(renderer="iframe")
            
            st.write(fig)
            
else:
    st.title('------Aplikasi Projek Data Mining------')
    st.markdown ('* **Nama Kolom harus sesuai pada dataset**')
    st.markdown ('* **Untuk saat ini web ini hanya bisa menggunakan dataset yang disediakan**')
    st.markdown ('* **Web ini hanya untuk menghitung Mean Absulute Error, Mean Squared Error, dan R Squared Score**')


    #========================= Input Dataset ==========================
    uploaded_file = st.file_uploader("Masukan Dataset")
    #=======================================================================

    #============================= Inputan ===================================
    if uploaded_file is not None :    
        df = pd.read_csv(uploaded_file)
        st.write(df)
        sel_col, displ_col = st.columns(2)
        displ_col.write('--Nama Kolom Tabel--')
        displ_col.write(df.columns)
        input_tabel = sel_col.text_input(' Masukan kolom Chart pertama  ',)
        chart_select = sel_col.selectbox('Pilih bentuk Chart :', ('Bar Chart', 'Line Chart'))
        input_hitungan = sel_col.text_input('Masukan kolom yang akan di hitung ','PULocationID')
        metric = sel_col.selectbox("Pilih Metric",('minkowski', 'euclidean'))

        if input_tabel and uploaded_file  is not None :
            st.subheader('Berikut data berdasarkan kota dari data di atas :')
            chart_var_dist = pd.DataFrame(df[input_tabel].value_counts()).head(30)
            if chart_select == 'Bar Chart':
                st.bar_chart(chart_var_dist)
            elif chart_select == 'Line Chart' :
                st.line_chart(chart_var_dist)

            #========================PARAMETER============================
            sel_col, displ_col = st.columns(2)   
            max_depth = sel_col.slider('max_depth', 1,100)

            n_estimmators = sel_col.slider('n_estimators', 1,100)

            regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimmators)

            dataset = df
            x = dataset[[input_hitungan]]
            y = dataset[['trip_distance']]

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
            sc = StandardScaler()
            x_train = sc.fit_transform(x_train)
            x_test = sc.transform(x_test)

            #============================KLASIFIKASI==================================#
            regr.fit(x,y)
            pred = regr.predict(y)

            displ_col.subheader('Mean Absolute Error')
            displ_col.write(mean_absolute_error(y,pred))

            displ_col.subheader('Mean Squared Error')
            displ_col.write(mean_squared_error(y,pred))

            displ_col.subheader('R Squared Score Error')
            displ_col.write(r2_score(y,pred))
