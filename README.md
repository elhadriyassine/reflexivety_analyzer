# reflexivety_analyzer
Stock News Reflexivity Analyzer
This script is an NLP-powered tool that analyzes financial news headlines for a specific stock to identify and flag repetitive or "reflexive" news. By converting headlines into numerical vectors and measuring their similarity, it helps filter out redundant information from a news feed. This is my first attempt to detect repeated narratives as a part of applying the concepts of theory of reflexivity, as a tool to analyse the market .

Features
  - Custom Stock Ticker: The user can enter a stock symbol (e.g., AAPL, NVDA, MSFT) to fetch relevant news.
  - Intelligent Data Fetching: The script attempts to retrieve 30 days of company-specific news. If none is available, it falls back to a         general technology news feed and filters for the stock's company name.
  - Semantic Similarity: It uses a pre-trained SentenceTransformer model (all-MiniLM-L6-v2) to create a semantic embedding for each headline.     This allows it to understand the meaning, not just the keywords.
  - Reflexivity Detection: Using cosine similarity between consecutive headlines, the script identifies and flags news stories that are           essentially the same.

Dependencies: All required Python libraries are automatically installed at runtime.

How to Run

-- Get a Finnhub API Key: You'll need a free API key from Finnhub to access their news data.

-- Set up Google Colab Secrets: This script is designed to run in a Google Colab environment. You must store your Finnhub API key in the Colab Secrets panel.

-- Click the key icon on the left-hand sidebar.

-- Click + New secret.

-- For the name, use finhubAPI. For the value, paste your API key. Make sure the Notebook access checkbox is enabled.

-- Execute the Script: Run the code in your notebook. You'll be prompted to enter a stock ticker symbol.

 What to Expect 
When you run the script, you'll be guided through the process in the console:
* A message confirming the Finnhub client has been initialized.
* A prompt will ask you to enter a stock symbol.
+ Status updates will inform you that the script is fetching headlines, computing embeddings, and detecting reflexive clusters.
*The final output is a pandas DataFrame displaying the top 10 news headlines. Each row will have a title and a reflexive flag (True if it's a near-duplicate, False otherwise).
