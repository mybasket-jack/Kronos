from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

# Append current directory to import from model.py
sys.path.append(os.path.dirname(__file__))
from model import Kronos, KronosTokenizer, KronosPredictor

app = FastAPI(title="Kronos Prediction Microservice")

print("Loading Kronos Tokenizer...")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
print("Loading Kronos-base Model...")
model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
print("Initializing Predictor...")
predictor = KronosPredictor(model, tokenizer, max_context=512)
print("Kronos Microservice Ready.")

class KLineItem(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float

class PredictRequest(BaseModel):
    market_code: str
    history: list[KLineItem]
    pred_len: int = 5

@app.get("/")
def health_check():
    return {"status": "ok", "service": "kronos-server"}

@app.post("/predict")
def predict_trend(req: PredictRequest):
    if len(req.history) < 10:
        raise HTTPException(status_code=400, detail="Not enough history data (min 10 bars)")
    
    # Convert to DataFrame
    data = []
    for item in req.history:
        data.append({
            "timestamps": pd.to_datetime(item.timestamp),
            "open": item.open,
            "high": item.high,
            "low": item.low,
            "close": item.close,
            "volume": item.volume,
            "amount": item.amount
        })
    df = pd.DataFrame(data)
    
    # Kronos predicts the next N days
    x_df = df[['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_timestamp = df['timestamps']
    
    # Generate mock future timestamps (assume business days)
    last_date = x_timestamp.iloc[-1]
    y_timestamp = []
    current_date = last_date
    while len(y_timestamp) < req.pred_len:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:  # Monday to Friday
            y_timestamp.append(current_date)
            
    try:
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=pd.Series(y_timestamp),
            pred_len=req.pred_len,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=False
        )
        
        # Prepare response
        pred_list = []
        for i in range(len(pred_df)):
            pred_list.append({
                "timestamp": y_timestamp[i].strftime("%Y-%m-%d"),
                "open": float(pred_df.iloc[i]['open']),
                "high": float(pred_df.iloc[i]['high']),
                "low": float(pred_df.iloc[i]['low']),
                "close": float(pred_df.iloc[i]['close'])
            })
            
        return {"status": "success", "predictions": pred_list}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
