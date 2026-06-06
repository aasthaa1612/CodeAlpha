import os,json,joblib
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.svm import SVR

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
MODEL_DIR=os.path.join(BASE_DIR,"saved_model")
DATA_DIR=os.path.join(BASE_DIR,"..","data")
os.makedirs(MODEL_DIR,exist_ok=True);os.makedirs(DATA_DIR,exist_ok=True)
MODEL_PATH=os.path.join(MODEL_DIR,"car_model.pkl")
SCALER_PATH=os.path.join(MODEL_DIR,"scaler.pkl")
ENCODERS_PATH=os.path.join(MODEL_DIR,"encoders.pkl")
METRICS_PATH=os.path.join(MODEL_DIR,"metrics.json")
DATA_PATH=os.path.join(DATA_DIR,"car_data.csv")

BRANDS=["Maruti","Hyundai","Honda","Toyota","Ford","Mahindra","Tata","Chevrolet","Renault","Volkswagen","BMW","Audi","Mercedes","Kia","Nissan"]
CAT_COLS=["fuel_type","seller_type","transmission","brand"]
NUM_COLS=["year","present_price","kms_driven","owner"]

def generate_dataset(n=600):
    np.random.seed(42);rows=[]
    for _ in range(n):
        brand=np.random.choice(BRANDS)
        luxury=brand in ["BMW","Audi","Mercedes"]
        year=np.random.randint(2005,2025)
        age=2025-year
        present=round(np.random.uniform(15,60,) if luxury else np.random.uniform(3,20),2)
        kms=int(np.random.uniform(5000,age*18000+5000))
        fuel=np.random.choice(["Petrol","Diesel","CNG"],p=[.5,.4,.1])
        seller=np.random.choice(["Dealer","Individual"],p=[.6,.4])
        trans=np.random.choice(["Manual","Automatic"],p=[.35,.65] if luxury else [.7,.3])
        owner=np.random.choice([0,1,2,3],p=[.5,.3,.15,.05])
        dep=0.15*age+0.05*owner+0.08*(kms/100000)
        fuel_f=1.1 if fuel=="Diesel" else 0.9 if fuel=="CNG" else 1.0
        trans_f=1.15 if trans=="Automatic" else 1.0
        sell=round(max(0.5,present*(1-dep)*fuel_f*trans_f+np.random.normal(0,0.8)),2)
        rows.append({"brand":brand,"year":year,"present_price":present,"kms_driven":kms,"fuel_type":fuel,"seller_type":seller,"transmission":trans,"owner":owner,"selling_price":sell})
    df=pd.DataFrame(rows);df.to_csv(DATA_PATH,index=False)
    return df

def load_data():
    if os.path.exists(DATA_PATH):return pd.read_csv(DATA_PATH)
    return generate_dataset()

def train_and_evaluate():
    df=load_data()
    X=df.drop("selling_price",axis=1);y=df["selling_price"]
    encoders={}
    for c in CAT_COLS:
        le=LabelEncoder();X[c]=le.fit_transform(X[c]);encoders[c]=le
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    scaler=StandardScaler();X_train_s=scaler.fit_transform(X_train);X_test_s=scaler.transform(X_test)
    models={"Linear Regression":LinearRegression(),"Ridge Regression":Ridge(alpha=1.0),"Random Forest":RandomForestRegressor(n_estimators=100,random_state=42),"Gradient Boosting":GradientBoostingRegressor(n_estimators=100,random_state=42),"SVR":SVR(kernel="rbf")}
    results={};best_r2=-999;best_name=None;best_model=None
    print("="*60+"\n  CAR PRICE PREDICTION - MODEL COMPARISON\n"+"="*60)
    for name,model in models.items():
        model.fit(X_train_s,y_train);y_pred=model.predict(X_test_s)
        r2=r2_score(y_test,y_pred);mae=mean_absolute_error(y_test,y_pred);rmse=np.sqrt(mean_squared_error(y_test,y_pred))
        cv=cross_val_score(model,X_train_s,y_train,cv=5,scoring="r2")
        results[name]={"r2":round(r2*100,2),"mae":round(mae,2),"rmse":round(rmse,2),"cv_mean":round(cv.mean()*100,2),"cv_std":round(cv.std()*100,2)}
        print(f"\n  {name}: R²={r2*100:.2f}% | MAE={mae:.2f} | RMSE={rmse:.2f} | CV={cv.mean()*100:.2f}%")
        if r2>best_r2:best_r2=r2;best_name=name;best_model=model
    joblib.dump(best_model,MODEL_PATH);joblib.dump(scaler,SCALER_PATH);joblib.dump(encoders,ENCODERS_PATH)
    feat_imp={}
    if hasattr(best_model,"feature_importances_"):feat_imp={c:round(float(v),4) for c,v in zip(X.columns,best_model.feature_importances_)}
    metrics={"best_model":best_name,"best_r2":round(best_r2*100,2),"all_results":results,"feature_names":list(X.columns),"cat_cols":CAT_COLS,"num_cols":NUM_COLS,"feature_importances":feat_imp,"brands":BRANDS,"dataset_info":{"total_samples":len(df),"train_samples":len(X_train),"test_samples":len(X_test),"num_features":len(X.columns),"price_range":[round(float(y.min()),2),round(float(y.max()),2)],"avg_price":round(float(y.mean()),2)}}
    with open(METRICS_PATH,"w") as f:json.dump(metrics,f,indent=2)
    print(f"\n{'='*60}\n  [BEST] {best_name} (R²={best_r2*100:.2f}%)\n  [SAVE] {MODEL_PATH}\n{'='*60}")
    return metrics

def predict(features:dict):
    model=joblib.load(MODEL_PATH);scaler=joblib.load(SCALER_PATH);encoders=joblib.load(ENCODERS_PATH)
    with open(METRICS_PATH) as f:metrics=json.load(f)
    df=pd.DataFrame([features])
    for c in CAT_COLS:
        if c in df.columns:df[c]=encoders[c].transform(df[c])
    ordered=metrics["feature_names"];df=df[ordered]
    scaled=scaler.transform(df);pred=model.predict(scaled)[0]
    return {"predicted_price":round(float(pred),2),"model_used":metrics["best_model"]}

if __name__=="__main__":train_and_evaluate()
