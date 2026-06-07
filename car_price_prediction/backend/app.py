import os,json
from flask import Flask,request,jsonify,send_from_directory
from flask_cors import CORS
import pandas as pd,numpy as np
from model import train_and_evaluate,predict,MODEL_PATH,METRICS_PATH,load_data,BRANDS,CAT_COLS

FRONTEND=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","frontend")
app=Flask(__name__,static_folder=FRONTEND,static_url_path="")
CORS(app)

def ensure_model():
    if not os.path.exists(MODEL_PATH):print("\n[*] No saved model -- training now...");train_and_evaluate()
    else:print("\n[OK] Model found -- ready!")

@app.route("/")
def index():return send_from_directory(FRONTEND,"index.html")

@app.route("/api/predict",methods=["POST"])
def api_predict():
    data=request.get_json()
    required=["brand","year","present_price","kms_driven","fuel_type","seller_type","transmission","owner"]
    missing=[k for k in required if k not in data]
    if missing:return jsonify({"error":f"Missing: {', '.join(missing)}"}),400
    try:
        features={"brand":data["brand"],"year":int(data["year"]),"present_price":float(data["present_price"]),"kms_driven":int(data["kms_driven"]),"fuel_type":data["fuel_type"],"seller_type":data["seller_type"],"transmission":data["transmission"],"owner":int(data["owner"])}
    except(ValueError,TypeError):return jsonify({"error":"Invalid input types."}),400
    return jsonify(predict(features))

@app.route("/api/metrics")
def api_metrics():
    if not os.path.exists(METRICS_PATH):return jsonify({"error":"Model not trained."}),404
    with open(METRICS_PATH) as f:return jsonify(json.load(f))

@app.route("/api/dataset")
def api_dataset():
    df=load_data()
    return jsonify(df.to_dict(orient="records"))

@app.route("/api/dataset/stats")
def api_stats():
    df=load_data();stats={}
    for brand in df["brand"].unique():
        sub=df[df["brand"]==brand]
        stats[brand]={"count":len(sub),"avg_price":round(sub["selling_price"].mean(),2),"avg_kms":int(sub["kms_driven"].mean()),"avg_year":int(sub["year"].mean())}
    return jsonify(stats)

@app.route("/api/retrain",methods=["POST"])
def api_retrain():return jsonify(train_and_evaluate())

if __name__=="__main__":ensure_model();app.run(debug=True,port=5001)
