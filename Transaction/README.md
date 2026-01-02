# Analisis Ujian Akhir Semester

## Mata Kuliah: Machine Learning

**Nama**: Zahrani Cahya Priesa
**NIM**: 1103223074
**Topik Proyek**: *Fraud Detection menggunakan Machine Learning*

---

## 1. Latar Belakang

Fraud detection merupakan salah satu permasalahan klasik dalam Machine Learning yang memiliki karakteristik **data tidak seimbang (imbalanced data)**, di mana jumlah transaksi normal jauh lebih banyak dibandingkan transaksi fraud. Kondisi ini menuntut pemilihan metode preprocessing, algoritma, serta metrik evaluasi yang tepat agar model tidak bias terhadap kelas mayoritas.

Proyek ini bertujuan untuk membangun dan membandingkan beberapa algoritma Machine Learning dalam mendeteksi transaksi fraud, serta menganalisis performa masing-masing model berdasarkan metrik evaluasi yang relevan.

---

## 2. Dataset

Dataset yang digunakan merupakan dataset transaksi keuangan yang memiliki karakteristik:

* Data numerik hasil transformasi (misalnya PCA / feature engineering)
* Label biner:

  * `0` : Transaksi normal
  * `1` : Transaksi fraud
* Distribusi kelas **sangat tidak seimbang**

Karakteristik ini menjadikan fraud detection sebagai kasus *binary classification* dengan fokus pada minimisasi *false negative*.

---

## 3. Tahapan Preprocessing

Tahapan preprocessing yang dilakukan pada notebook meliputi:

### 3.1 Data Loading

* Dataset dimuat dari file eksternal (Google Drive / local path)
* Dilakukan pengecekan struktur data dan tipe fitur

### 3.2 Train-Test Split

* Data dibagi menjadi data latih dan data uji
* Pembagian dilakukan sebelum oversampling untuk menghindari *data leakage*

### 3.3 Handling Imbalanced Data (SMOTE)

* Digunakan **SMOTE (Synthetic Minority Over-sampling Technique)** pada data latih
* Tujuan:

  * Menyeimbangkan jumlah kelas fraud dan non-fraud
  * Meningkatkan kemampuan model dalam mengenali kelas minoritas

### 3.4 Feature Scaling

* Standardisasi fitur menggunakan **StandardScaler**
* Scaling diterapkan khusus untuk model yang sensitif terhadap skala fitur seperti:

  * Logistic Regression
  * MLP Neural Network

---

## 4. Model yang Digunakan

Pada proyek ini digunakan beberapa algoritma Machine Learning untuk dibandingkan performanya:

### 4.1 Logistic Regression (LR)

* Model baseline untuk klasifikasi biner
* Mudah diinterpretasikan
* Sensitif terhadap skala fitur

### 4.2 Random Forest (RF)

* Algoritma ensemble berbasis decision tree
* Mampu menangani non-linearitas data
* Relatif robust terhadap outlier dan noise

### 4.3 XGBoost (Extreme Gradient Boosting)

* Algoritma boosting dengan performa tinggi
* Efektif untuk dataset tabular
* Memiliki kompleksitas komputasi dan penggunaan memori yang lebih besar

### 4.4 Multi-Layer Perceptron (MLP)

* Model Neural Network sederhana
* Mampu mempelajari pola non-linear
* Membutuhkan tuning hyperparameter dan scaling data

---

## 5. Proses Training dan Evaluasi

### 5.1 Training Model

* Setiap model dilatih menggunakan data latih
* Untuk model tertentu digunakan data hasil scaling

### 5.2 Metrik Evaluasi

Evaluasi model tidak hanya menggunakan akurasi, tetapi juga:

* **Confusion Matrix**
* **Precision**
* **Recall**
* **F1-Score**

Alasan:

* Akurasi saja tidak cukup pada kasus imbalanced data
* Recall pada kelas fraud sangat penting untuk meminimalkan transaksi fraud yang lolos

---

## 6. Hasil dan Analisis

Secara umum hasil eksperimen menunjukkan bahwa:

* Model baseline (Logistic Regression) mampu memberikan performa awal yang cukup baik, namun masih terbatas dalam menangkap pola kompleks
* Random Forest memberikan peningkatan performa karena kemampuannya menangani non-linearitas
* XGBoost menunjukkan performa paling optimal dalam mendeteksi fraud, terutama dari sisi **recall dan F1-score**, meskipun membutuhkan optimasi memori
* MLP mampu mempelajari pola data, namun sensitif terhadap parameter dan waktu training

Model ensemble dan boosting cenderung lebih unggul dibandingkan model linear pada kasus fraud detection.

---

## 7. Kesimpulan

Berdasarkan eksperimen yang dilakukan:

1. Fraud detection merupakan permasalahan klasifikasi dengan tantangan utama berupa **imbalanced data**
2. Penggunaan SMOTE terbukti membantu meningkatkan performa model dalam mendeteksi kelas minoritas
3. Pemilihan metrik evaluasi yang tepat (Recall dan F1-score) sangat penting
4. Model XGBoost memberikan performa terbaik secara keseluruhan pada proyek ini

---

## 8. Saran Pengembangan

Beberapa pengembangan yang dapat dilakukan ke depannya:

* Hyperparameter tuning (Grid Search / Random Search)
* Penggunaan teknik imbalance lain (ADASYN, class weight)
* Implementasi model berbasis Deep Learning yang lebih kompleks
* Evaluasi menggunakan ROC-AUC dan Precision-Recall Curve

---

## 9. Repository GitHub

Struktur repository yang disarankan:

```
Fraud-Detection-ML/
│── Fraud_Detection.ipynb
│── README.md
│── dataset/
│── results/
│── requirements.txt
```

Dokumen ini dapat digunakan sebagai **README.md** pada repository GitHub untuk keperluan Ujian Akhir Semester Machine Learning.

---

**© 2026 – Zahrani Cahya Priesa**

