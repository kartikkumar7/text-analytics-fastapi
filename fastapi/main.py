
import uvicorn
from fastapi import FastAPI  
from ner_spacy import ner_spacy
# from word_embedding import embed_spacy
from transformer import summarize
from starlette.responses import Response
from pydantic import BaseModel

app = FastAPI(
    title = "NEW API",
    description = "NER extracts labels from the text"
)

# @app.get('/embed')
# def embed_text(text: str):
#     result = {}
#     message_embeddings = embed_spacy(text)
#     result = {"Word Embeddings from Spacy": message_embeddings }
#     return result


class Text(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}

@app.post("/summary")
def get_summary(text: Text):
    """Get summary from text"""
    # print(text.text)
    summary = summarize(text.text)
    # summary = summarize(text)
    result = {"Summary from transformers": summary}
    return result

@app.post("/ner")
def get_ner(text: Text):
    """Get ner from text"""
    # print(text.text)
    output_spacy = list(ner_spacy(text.text))
    # output_spacy = list(ner_spacy(text))
    result = {"NER from Spacy": output_spacy}
    return result


@app.get('/ner')
def ner_text(text: str):
    output_spacy = list(ner_spacy(text))
    result = {"NER from Spacy": output_spacy}
    return result

@app.get('/summary')
def summarize_text(text: str):
    summary = summarize(text)
    result = {"Summary from transformers": summary}
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)

