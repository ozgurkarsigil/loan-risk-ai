import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    auc
)
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
df=pd.read_csv(r"C:\Users\ozgur\Desktop\Loan_default.csv")
A=df["Default"]==1/df["Default"]
#print(A.value_counts())
X=df[["Income","LoanTerm","LoanAmount","InterestRate"]]
Y=df["Default"]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
model=LogisticRegression()
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
"""
print("Accuracy:",accuracy_score(Y_test,y_pred))
print(confusion_matrix(Y_test, y_pred))
print(df["Income"].mean())
print(df["Income"].std())
print(df["Income"].median())
print(df["Income"].min())
print(df["Income"].max())
print(df["LoanAmount"].mean())
print(df["LoanAmount"].std())
print(df["LoanAmount"].max())
print(df["LoanAmount"].min())
print(df["LoanTerm"].mean())
print(df["LoanTerm"].std())
print(df["LoanTerm"].max())
print(df["LoanTerm"].min())
print(df["InterestRate"].mean())
print(df["InterestRate"].std())
print(df["InterestRate"].max())
print(df["InterestRate"].min())
"""
#print(df["LoanAmount"].describe())
df["loan_amount_quantile"]=pd.qcut(df["LoanAmount"],q=4,labels=["Q1","Q2","Q3","Q4"])
#print(df.groupby("loan_amount_quantile")["Default"].agg(count="count",default_rate="mean"))
K=df[["CreditScore","MonthsEmployed","DTIRatio"]]
X2=pd.concat([X,K],axis=1)
X_train2,X_test2,y_train2,y_test2=train_test_split(X2,Y,test_size=0.2,random_state=42,stratify=Y)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train2)
X_test_scaled=scaler.transform(X_test2)
model=LogisticRegression()
model.fit(X_train_scaled,y_train2)
y_pred2=model.predict(X_test_scaled)
#print("Accuracy:",accuracy_score(y_test2,y_pred2))
#print("Confusion:", confusion_matrix(y_test2,y_pred2))
y_prob = model.predict_proba(X_test_scaled)[:, 1]

#print(roc_auc_score(y_test2, y_prob))
#print(y_prob.min())
#print(y_prob.max())
#print(y_prob.mean())
Thresholds=[ 0.30,0.25,0.22,0.17,0.14]
"""
for i in range(5):
    y_pred=(y_prob>=Thresholds[i]).astype(int)
    print(Thresholds[i])
    print("Accuracy:",accuracy_score(y_test2,y_pred))
    print("Confusion:", confusion_matrix(y_test2
    ap = average_precision_score(y_test, y_prob),y_pred))
    print("f11:", f1_score(y_test2,y_pred))
    print("recall:", recall_score(y_test2,y_pred))
"""
precision, recall, thresholds = precision_recall_curve(y_test2, y_prob)
ap = average_precision_score(y_test2, y_prob)
pr_auc = auc(recall, precision)
#print("Average Precision:", ap)
#print("PR-AUC:", pr_auc)
"""
import matplotlib.pyplot as plt

plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()
"""
f1_scores = 2 * precision[:-1] * recall[:-1] / (
    precision[:-1] + recall[:-1] + 1e-10
)

best_index = np.argmax(f1_scores)

best_threshold = thresholds[best_index]
best_f1 = f1_scores[best_index]

#print("Best threshold:", best_threshold)
#print("Best F1:", best_f1)
#print("Precision:", precision[best_index])
#print("Recall:", recall[best_index])
"""
from sklearn.ensemble import RandomForestClassifier
modelt=RandomForestClassifier(n_estimators=100,random_state=42)
modelt.fit(X_train2,y_train2)
yt_pred=modelt.predict(X_test2)
y_probt = modelt.predict_proba(X_test_scaled)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test2, y_probt)
ap = average_precision_score(y_test2, y_probt)
f1_scores = 2 * precision[:-1] * recall[:-1] / (
    precision[:-1] + recall[:-1] + 1e-10
)

best_index = np.argmax(f1_scores)

best_threshold = thresholds[best_index]
pr_auc = auc(recall, precision)
best_f1 = f1_scores[best_index]

print("Accuracy:",accuracy_score(y_test2,yt_pred))
print("Confusion:", confusion_matrix(y_test2,yt_pred))
print("f11:", f1_score(y_test2,yt_pred))
print("recall:", recall_score(y_test2,yt_pred))
print("Best F1:", best_f1)
print("Precision:", precision[best_index])
print("Recall:", recall[best_index])
print("Average Precision:", ap)
print("PR-AUC:", pr_auc)

from sklearn.ensemble import GradientBoostingClassifier
modelg=GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,random_state=42)
modelg.fit(X_train2,y_train2)
yg_pred=modelg.predict(X_test2)
yg_prob=modelg.predict_proba(X_test_scaled)[:,1]
precision,recall,thresholds=precision_recall_curve(y_test2,yg_prob)
ap=average_precision_score(y_test2,yg_prob)
f1_scores = 2 * precision[:-1] * recall[:-1] / (
    precision[:-1] + recall[:-1] + 1e-10
)
best_index = np.argmax(f1_scores)

best_threshold = thresholds[best_index]
pr_auc = auc(recall, precision)
best_f1 = f1_scores[best_index]
print("Accuracy:",accuracy_score(y_test2,yg_pred))
print("Confusion:", confusion_matrix(y_test2,yg_pred))
print("f11:", f1_score(y_test2,yg_pred))
print("recall:", recall_score(y_test2,yg_pred))
print("Best F1:", best_f1)
print("Precision:", precision[best_index])
print("Recall:", recall[best_index])
print("Average Precision:", ap)
print("PR-AUC:", pr_auc)
"""
"""
from sklearn.naive_bayes import GaussianNB
modelgn=GaussianNB()
modelgn.fit(X_train2,y_train2)
y_predgn=modelgn.predict(X_test2)
y_probgn=modelgn.predict_proba(X_test_scaled)[:,1]
precision,recall,thresholds=precision_recall_curve(y_test2,y_probgn)
ap=average_precision_score(y_test2,y_probgn)
f1_scores = 2 * precision[:-1] * recall[:-1] / (
    precision[:-1] + recall[:-1] + 1e-10
)
best_index = np.argmax(f1_scores)

best_threshold = thresholds[best_index]
pr_auc = auc(recall, precision)
best_f1 = f1_scores[best_index]
print("Accuracy:",accuracy_score(y_test2,y_predgn))
print("Confusion:", confusion_matrix(y_test2,y_predgn))
print("f11:", f1_score(y_test2,y_predgn))
print("recall:", recall_score(y_test2,y_predgn))
print("Best F1:", best_f1)
print("Precision:", precision[best_index])
print("Recall:", recall[best_index])
print("Average Precision:", ap)
print("PR-AUC:", pr_auc)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
modelLDA=LinearDiscriminantAnalysis()
modelLDA.fit(X_train2,y_train2)
y_predLDA=modelLDA.predict(X_test2)
y_probLDA=modelLDA.predict_proba(X_test_scaled)[:,1]
precision,recall,thresholds=precision_recall_curve(y_test2,y_probLDA)
ap=average_precision_score(y_test2,y_probLDA)
f1_scores = 2 * precision[:-1] * recall[:-1] / (
    precision[:-1] + recall[:-1] + 1e-10
)
best_index = np.argmax(f1_scores)

best_threshold = thresholds[best_index]
pr_auc = auc(recall, precision)
best_f1 = f1_scores[best_index]
print("Accuracy:",accuracy_score(y_test2,y_predLDA))
print("Confusion:", confusion_matrix(y_test2,y_predLDA))
print("f11:", f1_score(y_test2,y_predLDA))
print("recall:", recall_score(y_test2,y_predLDA))
print("Best F1:", best_f1)
print("Precision:", precision[best_index])
print("Recall:", recall[best_index])
print("Average Precision:", ap)
print("PR-AUC:", pr_auc)

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
modelLDA=QuadraticDiscriminantAnalysis()
modelLDA.fit(X_train2,y_train2)
y_predLDA=modelLDA.predict(X_test2)
y_probLDA=modelLDA.predict_proba(X_test_scaled)[:,1]
precision,recall,thresholds=precision_recall_curve(y_test2,y_probLDA)
ap=average_precision_score(y_test2,y_probLDA)
f1_scores = 2 * precision[:-1] * recall[:-1] / (
    precision[:-1] + recall[:-1] + 1e-10
)
best_index = np.argmax(f1_scores)

best_threshold = thresholds[best_index]
pr_auc = auc(recall, precision)
best_f1 = f1_scores[best_index]
print("Accuracy:",accuracy_score(y_test2,y_predLDA))
print("Confusion:", confusion_matrix(y_test2,y_predLDA))
print("f11:", f1_score(y_test2,y_predLDA))
print("recall:", recall_score(y_test2,y_predLDA))
print("Best F1:", best_f1)
print("Precision:", precision[best_index])
print("Recall:", recall[best_index])
print("Average Precision:", ap)
print("PR-AUC:", pr_auc)
"""
from sklearn.neural_network import MLPClassifier
modelMLP=MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
modelMLP.fit(X_train_scaled,y_train2)
y_predMLP=modelMLP.predict(X_test_scaled)
y_probMLP=modelMLP.predict_proba(X_test_scaled)[:,1]
precision,recall,thresholds=precision_recall_curve(y_test2,y_probMLP)
ap=average_precision_score(y_test2,y_probMLP)
f1_scores = 2 * precision[:-1] * recall[:-1] / (
    precision[:-1] + recall[:-1] + 1e-10
)
best_index = np.argmax(f1_scores)

best_threshold = thresholds[best_index]
pr_auc = auc(recall, precision)
best_f1 = f1_scores[best_index]
print("Best threshold:", best_threshold)
"""
print("Accuracy:",accuracy_score(y_test2,y_predLDA))
print("Confusion:", confusion_matrix(y_test2,y_predLDA))
print("f11:", f1_score(y_test2,y_predLDA))
print("recall:", recall_score(y_test2,y_predLDA))
print("Best F1:", best_f1)
print("Precision:", precision[best_index])
print("Recall:", recall[best_index])
print("Average Precision:", ap)
print("PR-AUC:", pr_auc)
"""
"""
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Bilgisayarı çok zorladı kullanılamadı
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

logistic_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

mlp_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=300,
        random_state=42
    ))
])

logistic_scores = cross_val_score(
    logistic_pipeline,
    X,
    Y,
    cv=cv,
    scoring="average_precision",
    n_jobs=-1
)

mlp_scores = cross_val_score(
    mlp_pipeline,
    X,
    y,
    cv=cv,
    scoring="average_precision",
    n_jobs=-1
)

print("Logistic AP scores:", logistic_scores)
print("Logistic Mean AP:", logistic_scores.mean())
print("Logistic Std:", logistic_scores.std())

print()

print("MLP AP scores:", mlp_scores)
print("MLP Mean AP:", mlp_scores.mean())
print("MLP Std:", mlp_scores.std())
"""
import os
import joblib
joblib.dump(Pipeline,"modelMLP.pkl")
print("Kaydedildi mi:", os.path.exists("modelMLP.pkl"))
print("Konum:", os.path.abspath("modelMLP.pkl"))
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import logging

app = FastAPI()

THRESHOLD = 0.1693532262785989

pipeline = joblib.load("modelMLP.pkl")


class LoanRequest(BaseModel):
    Income: int
    LoanAmount: int
    CreditScore: int
    LoanTerm: int
    InterestRate: float
    MonthsEmployed: int
    DTIRatio: float
    
    
class PredictionResponse(BaseModel):
    default_probability: float
    predicted_class: int
    risk_level: str


logging.basicConfig(level=logging.INFO)


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: LoanRequest):

    logging.info("Yeni prediction isteği geldi.")

    try:
        customer_df = pd.DataFrame([{
        "Income": customer.Income,
        "LoanTerm": customer.LoanTerm,
        "LoanAmount": customer.LoanAmount,
        "InterestRate": customer.InterestRate,
        "CreditScore": customer.CreditScore,
        "MonthsEmployed": customer.MonthsEmployed,
        "DTIRatio": customer.DTIRatio
            }])

        probability = pipeline.predict_proba(customer_df)[0][1]

        prediction = int(probability >= THRESHOLD)

        logging.info(
            f"Tahmin tamamlandı: "
            f"probability={probability:.4f}, "
            f"prediction={prediction}"
        )

        return PredictionResponse(
            default_probability=round(float(probability), 4),
            predicted_class=prediction,
            risk_level="high" if prediction == 1 else "low"
        )

    except Exception as e:

        logging.exception("Tahmin sırasında hata oluştu.")

        raise HTTPException(
            status_code=500,
            detail="Tahmin sırasında hata oluştu."
        )
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib

mlp_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=300,
        random_state=42
    ))
])

mlp_pipeline.fit(X_train2, y_train2)

joblib.dump(mlp_pipeline, "modelMLP.pkl")
loaded_model = joblib.load("modelMLP.pkl")

print(loaded_model)
print(hasattr(loaded_model, "predict_proba"))
print(loaded_model.named_steps["model"])
