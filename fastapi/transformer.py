from transformers import pipeline
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

summarizer = pipeline(task="summarization", model="sshleifer/distilbart-cnn-6-6", tokenizer="sshleifer/distilbart-cnn-6-6",framework="pt")

def summarize(text):
    summary = summarizer(text, max_length=40, min_length=20, do_sample=False)
    return summary

# model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
# tokenizer = AutoTokenizer.from_pretrained("t5-base")

# # T5 uses a max_length of 512 so we cut the article to 512 tokens.
# def summarize(text):
#     inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
#     outputs = model.generate(inputs["input_ids"], max_length=60, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
#     return str(tokenizer.decode(outputs[0]))