import finnhub
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google.colab import userdata


#  Finnhub client
try:
    finnhub_client = userdata.get('finhubAPI')
    if not finnhub_client:
         raise ValueError("Finnhub API key not found in Colab Secrets.")
except Exception as e:
    print(f"Error retrieving API key from Colab Secrets: {e}")
    print("Please make sure you have stored your Finnhub API key in Colab Secrets under the name 'finhubAPI'.")
   
    exit()

# Initialize Finnhub Client
try:
    finnhub_client = finnhub.Client(api_key=finnhub_client)
    print("Initialized.")
except Exception as e:
    print(f"Error initializing Finnhub client: {e}")
    exit() 

# Get Stock Symbol from User 
stock_symbol = input("Please enter the stock ticker symbol (AAPL, NVDA, MSFT, META, AMZN): ").strip().upper()

if not stock_symbol:
     print("No symbol entered. Exiting.")
     exit()

#  Embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

#  Fetch headlines with fallback
def fetch_news_headlines(ticker):
    # first try company_news over the last 30 days
    to_date   = datetime.utcnow().date()
    from_date = to_date - timedelta(days=30)
    raw = []
    try:
        raw = finnhub_client.company_news(
            ticker,
            _from=from_date.isoformat(),
            to=to_date.isoformat()
        )
    except:
        raw = []
    
    if not raw:
        raw = finnhub_client.general_news('technology')
        ticker = ticker.upper()
        raw = [item for item in raw if 'NVIDIA' in item.get('headline','').upper()]
    data = []
    for item in raw:
        dt = item.get('datetime')
        if dt is None:
            continue
        title = item.get('headline') or item.get('summary') or ""
        data.append({
            "title": title,
            "date": pd.to_datetime(dt, unit='s')
        })
    return pd.DataFrame(data)

#  Embedding & reflexivity detection
def compute_embeddings(df):
    return model.encode(df['title'].tolist())

def detect_reflexivity(df, embeddings, threshold=0.8):
    flags = [False]*len(embeddings)
    for i in range(1, len(embeddings)):
        if cosine_similarity([embeddings[i]],[embeddings[i-1]])[0][0] > threshold:
            flags[i] = True
    df['reflexive'] = flags
    return df

#  Pipeline
def run_reflexivity_pipeline(ticker):
    print(f"Fetching headlines for {ticker}…")
    df = fetch_news_headlines(ticker)
    if df.empty:
        print("No headlines found.")
        return df
    print(f"Found {len(df)} headlines; computing embeddings…")
    embs = compute_embeddings(df)
    print("Detecting reflexive clusters…")
    return detect_reflexivity(df, embs)

#  Execute example
df = run_reflexivity_pipeline(stock_symbol)
df.head(50)
