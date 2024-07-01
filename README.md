# Laporan Proyek Machine Learning - Rafi Rachmad Ramadhan

## Domain Proyek

Dewasa ini, teknologi semakin berkembang dengan pesat sehingga dalam kehidupan sehari-hari kita tidak akan luput dari penggunaan barang-barang elektronik misalnya, televisi, komputer, mesin cuci dan sebagainya. Semakin banyak barang-barang elektronik yang kita gunakan semakin besar pula resiko-resiko yang kita tanggung ketika terjadi keruskan. Terjadinya kerusakan bisa disebabkan oleh apa saja yang bisa mengakibatkan kebakaran salah satunya. Untuk itu perlu adanya alat yang mampu untuk mendeteksi asap secara dini yang disebabkan oleh kerusakan-kerusakan barang elektronik atau lainnya, sehingga bencana kebakaran dapat diatasi sejak dini. Alat tersebut adalah Smoke detector. 

- Saya mengambil topik ini karena untuk mendeteksi secara dini segumpalan asap, mengapa? karena untuk menghidari atau mengatasi kebakaran yang besar dengan cara membuat smoke detector menggunkan algoritma machine learning.

## Business Understanding

Katakanlah kita mengetahui bahwa fire alarm dengan karakteristik tertentu akan menyala dan tidak. Anda juga sedang dalam sebuah bangunan yang sangat luas dan tentu saja banyak barang-barang elektronik disana. Mungkin ketika diruangan kita, kita tahu bahwa ada asap disana, kita bisa mendeteksinya lebih dini, kita bisa membenarkannya atau melarikan diri bila keadaan memburuk. Lantas bagaimana keadaan orang yang berada diruangan lain. jika terjadi keadaan mendesak yang tiba-tiba terjadi seperti kebakaran. kemungkinan kita bisa keliling memberi tahu, tetapi pertanyaanya apakah sempat mengelilingi bangunan yang luas ini.

Tentu saja dari situlah masalah muncul. Oleh karena itu, penting bagi kita untuk mengetahui dan memprediksi apakah disekitar kita ada asap. Prediksi inilah yang digunakan untuk mengetahui apakah ada asap atau tidak sehingga kita bisa lebih wwaspada

### Problem Statements

Berdasarkan kondisi yang telah diuraikan sebelumnya, saya akan mengembangkan sebuah sistem prediksi asap untuk menjawab permasalahan berikut.

- Dari serangkaian karakteristik yang membuat alarm berbunyi, apa yang paling berpengaruh terhadap alarm yang berbunyi?
- Apakah alarm akan berbunyi dengan karakteristik tertentu?  

### Goals

Untuk  menjawab pertanyaan tersebut, saya akan membuat predictive modelling dengan tujuan atau goals sebagai berikut:
- Mengetahui apa yang paling berkorelasi terhadap fire alarm.
- Membuat model machine learning yang dapat memprediksi apakah ada asap seakurat mungkin berdasarkan karakteristik yang ada.

**Solution statements**:
untuk meraih goals ini adalah solusi yang saya berikan.
- Seperti yang dijelaskan sebelumnya bahwasannya tujuan saya adalah memklasifikasikan dengan target fire alarm yang menyala atau tidak.
- Pada kasus klasifikasi ini saya menggunakan 2 algoritmayaitu K-NN dan Random Forest
- Pada kasus klasifikasi ini saya mengevaluasi model dengan menggunkan metrik accuracy

## Data Understanding
Data yang saya gunakan adalah [Smoke Detection Dataset](https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset) yang diunduh dari kaggle. Dataset ini memiliki 62630 kumpulan karakteristik atau baris. 

### Variabel-variabel pada Smoke Detection Dataset adalah sebagai berikut:    
- UTC            : merupakan waktu universal terkodinasi
- Temperature[C] : merupakan suhu disekitar alat
- Humidity[%]    : merupakan konsentrasi kandungan dari uap air yang ada di udara
- TVOC[ppb]      : merupakan total semua kandungan komponen bahan kimia organik yang dapat menguap dan dapat mencemari udara, baik pada saat proses produksi, aplikasi sampai dengan barang jadi dan digunakan oleh end user.
- eCO2[ppm]      : merupakan perkiraan konsentrasi karbon dioksida yang dihitung dari konsentrasi TVOC yang diketahui.
- Raw H2         : merupakan raw hydrogen
- Raw Ethanol    : merupakan raw ethanol
- Pressure[hPa]  : merupakan tekanan udara dengan satuan hPa
- PM1.0          : merupakan partikel udara berukuran 10 mikrometer atau lebih kecil
- PM2.5          : merupakan partikel berukuran 2.5 mikron (mikrometer)
- NC0.5          : merupakan number concentration 0.5
- NC1.0          : merupakan number concentration 1.0
- NC2.5          : merupakan number concentration 2.5
- CNT            : merupakan suatu molekul silinder karbon dengan diameter ukuran nanometer

**Explaratory Data Analysis dan Visualisasi Data**:
- Informasi data
![info](https://user-images.githubusercontent.com/111117217/190332298-ad4aac75-2c3a-4d1c-8724-bd705ea73bdb.PNG)
- Deskripsi data
![desk](https://user-images.githubusercontent.com/111117217/190332282-79e751a2-6275-4788-b91e-e2daf898f50f.PNG)
![desk](https://user-images.githubusercontent.com/111117217/190332287-204afa0d-e266-47a9-bd0a-9a2f4b734364.PNG)
- menghapus variabel unnamed dan UTC karena tidak penting
![drop](https://user-images.githubusercontent.com/111117217/190332291-b437ab82-62b2-45ab-8d33-f6d6027af30a.PNG)
- menangani outliers dengan teknik IQR method
![out](https://user-images.githubusercontent.com/111117217/190332305-89cf0352-efec-455f-9620-74bfb69fafb6.PNG)
- histogram masing-masing fitur
![hist](https://user-images.githubusercontent.com/111117217/190332294-5578269b-00e9-4971-bce1-6f9ab63eda50.png)
- Heatmap korelasi
dari heatmap dibawah ini terlihat bahwasannya CNT dan TVOC yang paling berkorelasi dengan fire alarm
![kore](https://user-images.githubusercontent.com/111117217/190332299-5d00c417-4d79-4a75-9e50-d8363c859330.PNG)

## Data Preparation
Pada bagian ini saya menerapkan teknik data preparation yaitu Pembagian dataset dengan fungsi train_test_split dari library sklearn dan Standarisasi.

**Pembagian dataset**
- Pada proses pembagian dataset saya membaginya dengan perbandingan train set dan test set sebesar 90:10. saya membaginya menggunkan fungsi train_test_split. dan hasilnya terbagi sebagai berikut.
![traintest](https://user-images.githubusercontent.com/111117217/190332310-09ee74f0-1ee2-4ec4-85a4-a44add84e63a.PNG)
- Alasan saya membagi dataset adalah mempertahankan sebagian data yang ada untuk menguji seberapa baik generalisasi model terhadap data baru.

**Standarisasi**
- Pada proses standarisasi saya menggunakan StandarScaler() dari sklearn. pada proses ini saya hanya melakukan standarisasi pada data tarin, dimana nantinya akan membuat mean dan standar deviasinya menjadi 0 dan 1.
![traintest](https://user-images.githubusercontent.com/111117217/190332301-a5247d59-6ced-477a-a20f-5750902898f9.PNG)
- Alasan saya melakukan standarisasi adalah membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. 

## Modeling
Pada tahapan ini saya menggunkan dua model untuk menyelesaikan permaslaahan klasifikasi ini.

**KNN** 
- Untuk algoritma K-NN mempunyai kelebihan dimana ia mudah dipahami dan digunakan, namun disisi lain ia mempunyai kekurangan jika dihadapkan pada jumlah fitur atau dimensi yang besar (kutukan dimensi).
- Pada model ini saya memberikan k parameter sebesar 10.

**Random Forest**
- Untuk algoritma Random forest mempunyai kelebihan yaitu cukup sederhana tetapi memiliki stabilitas yang mumpuni, namun disisi lain training bisa berjalan lambat, tergantung pada parameter yang digunakan dan tidak bisa memperbaiki model yang dihasilkan secara berulang
- pada model ini saya menggunakan default parameter.

Karena saya mengevaluasi modelnya hanya dengan menggunkan matrik accuracy, jadi saya lihat dari akurasinya. Didapati hasil yang menunjukkan bahwa kedua model memiliki akurasi yang benar-benar sama. Jadi, Keduanya bisa dikatakan sama-sama baik.

## Evaluation
Pada proyek kali ini saya menggunakan metrik accuracy untuk kasus klasifikasi.
- Metrik accuracy ini diggunkan untuk menghitung seberapa banyak target dari data test yang bisa diprediksi dengan benar oleh model yang mana keluaran dari metrik ini adalah bentuk %.
- Metrik accuracy ini bekerja dengan menghitung rasio banyaknya data yang berhasil di prediksi dengan benar dengan keseluruan data test.
- Berdasarkan hasil yang telah diperoleh dari hasil evaluasi menunjukan bahwasannya kedua algoritma menghasilkan accuracy yang sama, dimana menunjukan hasil 100%.
![acc](https://user-images.githubusercontent.com/111117217/190334213-cd5ad0a7-2696-4f67-aa0c-680f6bb33900.PNG)

## Kesimpulan
Dari proyek smoke detection ini bisa disimpulkan bahwasannya pada kasus ini kedua model menghasilkan accuracy yang sama sama sempurna yaitu 100%, dimana hal ini menunjukan bahwasannya kedua model bisa di pilih salah satunya untun melanjutkannya ke proses deployment.

## Referensi
- Pradeep Kumar G, Rahul R, and Ravindharan N, “[Early Forest Fire Detection Using Machine Learning Algorithms](https://www.ijntr.org/download_data/IJNTR07040009.pdf),” International Journal of New Technology and Research, vol. 7, no. 4, pp. 1–5, Apr. 2021. 