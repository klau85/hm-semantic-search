# Build a Semantic Product Recommender with LLMs

This repo contains:
* Text data cleaning (code in the notebook `data-exploration.ipynb`)
* Semantic (vector) search and how to build a vector database (code in the notebook `vector-search.ipynb`). This allows users to find the most similar products to a natural language query (e.g., "a red shirt").
* Creating a web application using Gradio for users to get products recommendations (code in the file `gradio-dashboard.py`).

In order to create your vector database, you'll need to create a .env file in your root directory containing your OpenAI API key.

The data for this project can be downloaded from Kaggle https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations.  
Data folder structure:
```
./data/
- images/
- articles.csv
```

