# chatbot_engine.py

from .sentiment_model import get_sentiment
from .emotion_model import detect_emotion
from .retrieval import retrieve_law
from .vector_memory import store_memory, retrieve_memory, retrieve_rag
from transformers import pipeline
import torch

# -------- Load HF Pipeline --------
generator = pipeline(
    task="text-generation",
    model="google/flan-t5-small",
    device=0 if torch.cuda.is_available() else -1,
)

def summarize_rag_chunks(rag_text: str, max_words: int = 250) -> str:
    words = rag_text.split()
    return " ".join(words[:max_words])

def retrieve_memory_safe(user_input: str) -> str:
    memory = retrieve_memory(user_input)
    if memory:
        words = memory.split()
        return " ".join(words[:150])
    return "No past related conversation found."

def generate_response(user_input: str) -> str:
    sentiment = get_sentiment(user_input)
    emotion = detect_emotion(user_input)

    law = retrieve_law(user_input) or "Human behavior and relationships are shaped by emotional patterns, communication habits, and past experiences."
    memory = retrieve_memory_safe(user_input)
    rag_context = retrieve_rag(user_input)
    rag_context = summarize_rag_chunks(rag_context) if rag_context else "No additional reference found in the knowledge base."

    # -------- Prompt --------
    # Only the model sees instructions; the output won't include them
    prompt = f"""
User question: {user_input}

Detected emotion: {emotion}
Sentiment: {sentiment}

Relevant psychological insight:
{law}

Past conversation memory:
{memory}

solutions and advice based on the above information:
{rag_context}

"""

    outputs = generator(
        prompt,
        max_new_tokens=180,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3
    )

    # -------- Extract text --------
    result = outputs[0]["generated_text"].strip()

    # Store in memory
    store_memory(user_input, result)

    return result