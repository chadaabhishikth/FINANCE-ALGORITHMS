import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import concurrent.futures # For parallel processing

class MarketScreener:
    def __init__(self, tickers):
        self.tickers = tickers
        self.results = []

    def analyze_stock(self, ticker):
        """Analyzes a single stock and returns a summary."""
        try:
            # 1. Fetch
            df = yf.Ticker(ticker).history(period="5y")
            if len(df) < 500: return None
            
            df.index = pd.to_datetime(df.index).tz_localize(None)
            data = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

            # 2. Features
            horizons = [2, 5, 20, 60]
            for h in horizons:
                data[f"Ratio_{h}"] = data["Close"] / data["Close"].rolling(h).mean()
                data[f"Trend_{h}"] = data["Close"].shift(1).pct_change().rolling(h).apply(lambda x: (x > 0).sum())
            
            data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
            data.dropna(inplace=True)
            
            predictors = [f"Ratio_{h}" for h in horizons] + [f"Trend_{h}" for h in horizons] + ["Volume"]

            # 3. PROFICIENCY CHECK (Backtest)
            # We train on the past and test on the last 100 days to see if this stock is even 'predictable'
            train_bt = data.iloc[:-100]
            test_bt = data.iloc[-100:]
            
            model = RandomForestClassifier(n_estimators=200, min_samples_split=100, random_state=1, n_jobs=-1)
            model.fit(train_bt[predictors], train_bt["Target"])
            preds = model.predict(test_bt[predictors])
            acc_score = precision_score(test_bt["Target"], preds, zero_division=0)

            # 4. CURRENT PREDICTION
            model.fit(data[predictors], data["Target"])
            last_row = data.iloc[-1:][predictors]
            prob_up = model.predict_proba(last_row)[0][1]

            return {
                "Ticker": ticker,
                "Price": data['Close'].iloc[-1],
                "Backtest_Accuracy": acc_score,
                "AI_Confidence": prob_up
            }
        except Exception as e:
            return None

    def run(self):
        print(f"Scanning {len(self.tickers)} stocks... (Please wait)")
        
        # This uses Multi-threading to analyze stocks 10x faster
        with concurrent.futures.ThreadPoolExecutor() as executor:
            self.results = list(executor.map(self.analyze_stock, self.tickers))
        
        # Filter out errors and sort by confidence
        self.results = [r for r in self.results if r is not None]
        report = pd.DataFrame(self.results).sort_values(by="AI_Confidence", ascending=False)
        
        print("\n" + "="*50)
        print("TOP OPPORTUNITIES DETECTED")
        print("="*50)
        
        # Only show stocks with >60% confidence and >50% backtest reliability
        top_picks = report[(report['AI_Confidence'] >= 0.60) & (report['Backtest_Accuracy'] > 0.50)]
        
        if top_picks.empty:
            print("No high-probability setups found today.")
        else:
            print(top_picks.to_string(index=False))
        print("="*50)

# --- LIST OF STOCKS TO WATCH ---
watch_list = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", # Big Tech
    "AMD", "INTC", "NFLX", "PYPL", "V", "MA", "JPM",        # Finance/Semi
    "SPY", "QQQ", "IWM", "DIA"                              # Index ETFs
]

if __name__ == "__main__":
    screener = MarketScreener(watch_list)
    screener.run()
