
# Team Number – Project Title

## Team Info
- 22471A05XX — Mutyam Srilakshmi ( [LinkedIn](https://www.linkedin.com/in/mutyam-srilakshmi-9065b8341 ) )
_Work Done: Dataset preprocessing, SMOTEENN balancing, EDA graphs (outlier removal, normalization, class distribution), report writing.

- 22471A05XX — Grandhisila Durga Vyshnavi ( [LinkedIn](linkedin.com/in/g-durga-vyshnavi-a83a87324) )
_Work Done: BiLSTM model implementation, training & validation accuracy plotting, performance evaluation, confusion matrix and ROC curve generation.

- 22471A05XX — Shaik Kagaji Rabia Basri( [LinkedIn](linkedin.com/in/shaik-kagaji-rabia-basri-766182280) )
_Work Done: HREF ensemble integration (Random Forest, AdaBoost, SVM), stacking meta-learner design, final hybrid model testing, result comparison table.


---

## Abstract
Cardiovascular disease (CVD) continues to be
a leading global cause of mortality, indicating a need
for new methods to maintain a high precision for CVD
diagnosis and early CVD detection. We introduce here a
novel hybridized ensemble model by improving a a Bidirec
tional Long Short-Term Memory (BiLSTM) based neural
network through a Hybrid Refined Ensemble Framework
(HREF) for improving classification. The hybrid model was
applied to the non-standardized Cleveland dataset, which
contains clinical-related cardiology parameters relevant to
cardiovascular health. The dataset was thoroughly pre
processed prior to training, involving outlier removal,
missing value transformation, feature normalization, and
subsequent utilization of SMOTEENN for the purpose
of class balancing. The BiLSTM part is designed, to
some extent, to capture intricate relationships behind the
features, and the ensemble part through HREF boosts
the prediction reliability through the combined output of
numerous ensembles. From experiments carried out on the
hybrid model, accuracy was 94.7 with ROC-AUC 0.9474
and good precision and recall scores. The findings of
this research indicate that deep learning combined with
ensemble refinement is a reliable and effective approach to
support the early detection of heart disease in real clinical
settings.

---

## Paper Reference (Inspiration)
👉 **[Paper Title Heart Disease Prediction through Hybrid LSTM and HREF Models
  – Author Names Dr. M. Sireesha, Srilakshmi Mutyam, Durga Vyshnavi Grandhisila, Rabia Basri Shaik Kagaji, Geeta Padole, S. Vasundhara
 ](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10494316)**
Original conference/IEEE paper used as inspiration for the model.

---

## Our Improvement Over Existing Paper
We improved the preprocessing pipeline using outlier removal (Z-score), StandardScaler normalization, and SMOTEENN balancing.
We combined deep learning (BiLSTM) with an advanced ensemble framework (HREF) using stacking, which improved stability and reduced overfitting.
We achieved improved final accuracy (94.7%) and strong ROC-AUC (0.9474) on the Cleveland dataset.

---

## About the Project
This project predicts whether a person has heart disease using clinical health parameters such as age, cholesterol, blood pressure, chest pain type, etc.
Early heart disease detection helps doctors take preventive action and reduce risk of severe complications.
Input: Patient clinical features
→ Preprocessing: outlier removal + scaling + balancing
→ Model: BiLSTM + Ensemble classifiers (RF, AdaBoost, SVM)
→ HREF stacking meta-learner
→ Output: Heart Disease / No Heart Disease prediction with evaluation metrics
---

## Dataset Used
👉 **[Cleveland Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)**

**Dataset Details:**
The project uses the Cleveland Heart Disease Dataset, which is a widely used benchmark dataset for heart disease prediction.
Total Records: 303 patient samples
Total Features: 14 columns (13 input features + 1 target)
Type: Tabular clinical dataset
Source: UCI / Kaggle Cleveland Heart Disease dataset

---

## Dependencies Used
Python
NumPy, Pandas
Matplotlib, Seaborn
Scikit-learn
TensorFlow / Keras
Imbalanced-learn (SMOTEENN)

---

## EDA & Preprocessing
Outlier removal: Z-score (±3)
Missing values: imputation/cleaning
Feature normalization: StandardScaler
Class balancing: SMOTEENN
Train-test split: 80:20 stratified
Graphs included:
Before/After outlier removal
Before/After normalization
Class distribution before/after SMOTEENN

---

## Model Training Info
Deep learning model: BiLSTM
Ensemble models: Random Forest, AdaBoost, SVM
Final hybrid model: HREF + BiLSTM
Training monitored using:
Training accuracy
Validation accuracy
Dropout layers were used to reduce overfitting.

---

## Model Testing / Evaluation
Evaluation metrics used:
Accuracy
Precision
Recall
F1-score
ROC-AUC
Confusion Matrix
These metrics help verify performance in medical classification tasks.

---

## Results
Final Hybrid Model (HREF + BiLSTM) achieved:
Accuracy: 94.7%
Precision: 91.2%
Recall: 94.1%
F1-score: 92.6%
ROC-AUC: 0.9474
Confusion matrix showed only 2 errors (1 false positive, 1 false negative).
---

## Limitations & Future Work
Limitations
Dataset is small (303 records), so generalization to large real-world hospital datasets may be limited.
Model performance may vary with different populations and clinical conditions.
Future Work
Use larger datasets with more clinical parameters.
Explore attention models and transformers for tabular medical data.
Deploy as a real-time clinical decision support tool with privacy-preserving methods.

---

## Deployment Info
Model can be deployed using:
Flask / FastAPI backend
Web interface for patient input
Prediction output with probability score
Future scope includes hospital-ready integration and secure cloud deployment.

---
