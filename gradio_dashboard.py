import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
import gradio as gr
import torch
import os

load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"

articles = pd.read_csv('data/articles_full_desc.csv', dtype={'article_id': str})

async def recommend_articles(query):
    # embeddings = HuggingFaceEmbeddings(
    #     model_name=os.getenv('HUGGING_FACE_EMBEDDING'),
    #     model_kwargs={'device': device}
    # )
    embeddings = OpenAIEmbeddings()
    db_articles = Milvus(
        embedding_function=embeddings,
        collection_name=os.getenv('MILVIUS_COLLECTION'),
        connection_args={
            "uri": os.getenv('ZILLIZ_CLOUD_URI'),
            "token": os.getenv('ZILLIZ_CLOUD_API_KEY'),
            "secure": True,
        }
    )

    recs = db_articles.similarity_search_with_score(query, k=24)

    rec_data = {
        "article_id": [rec[0].metadata['article_id'] for rec in recs],
        "score": [rec[1] for rec in recs]
    }
    recs_df = pd.DataFrame(rec_data)

    articles_rec = pd.merge(
        recs_df,
        articles,
        on='article_id',
        how='left'
    )

    results = []
    for _, row in articles_rec.iterrows():
        image_path = 'data/images/' + str(row['article_id'])[:3] + '/' + str(row['article_id']) + '.jpg'
        caption = f"Score: {row['score']:.2f}\n{row['full_description']}"
        results.append((image_path, caption))

    return results

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown('# H&M articles recommender')

    with gr.Row():
        user_query = gr.Textbox(label="Enter article description", placeholder="e.g., A white shirt for man...")
        submit_button = gr.Button(value="Find recommendations")

    gr.Markdown("## Recommendations")

    output = gr.Gallery(label="Recommended articles", rows=3, columns=8)

    submit_button.click(
        fn=recommend_articles,
        inputs=[user_query],
        outputs=output,
    )

if __name__ == "__main__":
    dashboard.launch()