# ml-scratch

Implementasi mini-library machine learning (terinspirasi oleh scikit-learn) untuk kebutuhan klasifikasi, regresi, clustering, dan reduksi dimensi. Di dalamnya ada Logistic Regression, Ridge/Lasso, Decision Tree (Gini/Entropy), Random Forest, KNN, SVM (Linear), KMeans, PCA, StandardScaler/MinMaxScaler/Encoder, VarianceThreshold/SelectKBest, K-Fold cross-validation, serta metrik Accuracy, F1, MSE, dan R². Proyek ini dibangun sepenuhnya dari nol hanya menggunakan `numpy` tanpa dependency ML eksternal apa pun.

## Struktur Singkat

- `base/` — BaseEstimator & Mixin (ClassifierMixin, RegressorMixin, TransformerMixin).
- `linear_model/` — LinearRegression, LogisticRegression, Ridge, Lasso.
- `tree/` — DecisionTreeClassifier, DecisionTreeRegressor.
- `ensemble/` — RandomForestClassifier, RandomForestRegressor.
- `neighbors/` — KNeighborsClassifier, KNeighborsRegressor.
- `svm/` — SVC (linear kernel).
- `cluster/` — KMeans.
- `decomposition/` — PCA.
- `preprocessing/` — StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder.
- `feature_selection/` — VarianceThreshold, SelectKBest.
- `model_selection/` — train_test_split, KFold, cross_val_score.
- `metrics/` — accuracy_score, precision_recall_f1, mean_squared_error, r2_score, confusion_matrix.

## Fitur utama

| Kategori            | Komponen                 | Keterangan                             |
| ------------------- | ------------------------ | -------------------------------------- |
| **Klasifikasi**     | Logistic Regression      | Solver MGD, visualisasi lintasan bobot |
|                     | SVM                      | Multi-kernel (Linear, RBF, Polynomial) |
|                     | Decision Tree            | Gini & Entropy, visualisasi pohon      |
| **Seleksi Fitur**   | Forward Selection        | Greedy penambahan fitur terbaik        |
|                     | Backward Elimination     | Greedy penghapusan fitur terburuk      |
|                     | Backward-Forward         | Kombinasi eliminasi + re-evaluasi      |
| **Reduksi Dimensi** | PCA                      | Eigen-decomposition covariance matrix  |
| **Preprocessing**   | Encoder & Scaler         | Normalisasi & encoding kategorikal     |
| **Validasi**        | K-Fold / StratifiedKFold | Cross-validation standar & stratified  |
| **Metrik**          | Accuracy, F1-Score       | Evaluasi performa klasifikasi          |
