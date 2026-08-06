"""
train_model.py

Trains the prompt -> best-model classifier and saves everything the Flask app needs:
  - model.pt          (trained PyTorch network weights)
  - vectorizer.pkl     (fitted TF-IDF vectorizer)
  - trigger_words.json (keyword lists used for the extra features)
  - config.json        (input/output sizes, model names)

Run this once (or any time you update data/prompts.csv or the trigger words) with:
    python train_model.py
"""

import json
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

random.seed(42)
torch.manual_seed(42)

MODEL_NAMES = {
    0: "DeepSeek",
    1: "ChatGPT",
    2: "Claude",
    3: "Gemini",
    4: "Perplexity",
    5: "Grok",
    6: "Kimi",
    7: "GLM",
}

TRIGGER_WORDS = {
    0: ["solve for", "equation", "derivative", "integral", "prove", "theorem", "matrix",
        "probability", "calculate", "eigenvalue", "eigenvector", "gaussian", "combinatorics",
        "differential", "induction", "linear algebra", "calculus", "vector", "polynomial",
        "quadratic", "taylor series", "limit of", "hypothesis testing", "determinant",
        "conditional probability", "l'hopital", "vector spaces", "complete the square",
        "trigonometric identity", "permutations", "combinations", "related rates",
        "normal distribution", "solid of revolution", "partial derivative",
        "null hypothesis", "inverse of a matrix", "system of inequalities",
        "pigeonhole principle", "mean value theorem", "convergence", "confidence interval"],
    1: ["write a story", "write a poem", "write a song", "brainstorm names", "tagline",
        "slogan", "caption", "write lyrics", "creative", "write a script", "haiku",
        "toast", "eulogy", "vows", "backstory", "dialogue", "headline", "jingle",
        "short story", "fairy tale", "monologue", "screenplay", "character", "plot twist",
        "fictional", "poem about", "song about", "roast", "limerick"],
    2: ["analyze", "critique", "think through", "pros and cons", "ethical", "argument",
        "weigh the", "reasoning", "logical fallacy", "evaluate", "tradeoffs", "implications",
        "counterargument", "moral dilemma", "review my", "philosophical", "flaws in",
        "credibility", "rhetorical", "unintended consequences", "fair to", "consistency",
        "strengths and weaknesses", "perspective on", "assess",
        "code review", "review this code", "audit this code", "large codebase",
        "legacy code", "refactor this large", "pull request", "security vulnerability",
        "debug this complex", "review this pull request", "codebase",
        "python", "javascript", "java", "c++", "rust", "code", "debug", "function", "algorithm",
        "sql", "api", "docker", "unit test", "data structure", "big o", "regex", "compile",
        "refactor", "syntax error", "recursion", "linked list", "binary tree", "hash table",
        "hash map", "trie", "compiler", "framework", "backend", "frontend",
        "database schema", "git", "dockerfile", "ci/cd", "websocket", "shell script",
        "graphql", "rest api", "segmentation fault", "null pointer", "async", "await",
        "pointers", "memory allocation", "garbage collection", "stack and a queue",
        "b-tree", "dynamic programming", "virtual memory", "time complexity"],
    3: ["explain how", "what is the difference between", "what are some", "beginner's guide",
        "give me an overview", "what causes", "how does", "basics of", "tips for",
        "how do", "what happens when", "why does", "how is", "simple explanation of"],
    4: ["latest", "recent", "current", "right now", "today", "this year", "up to date",
        "breaking", "newest", "search for", "find recent", "what's happening",
        "this quarter", "this week", "ongoing", "as of", "currently", "trending"],
    5: ["trending", "hot take", "viral", "meme", "twitter", "reddit", "internet reaction",
        "discourse", "unfiltered", "snarky", "sarcastic take", "online controversy",
        "streamers", "the tea", "roast this", "internet's reaction", "casual rundown",
        "blowing up", "social media", "current mood online", "internet trend"],
    6: ["summarize this entire", "read through this whole", "digest this long",
        "go through this entire", "long document", "this whole document",
        "multi-hundred page", "extract action items", "long transcript", "large codebase's",
        "200 page", "300 page", "whole report", "entire manuscript", "long white paper",
        "cross reference this long document", "massive pdf", "lengthy", "long form article",
        "whole podcast transcript", "entire syllabus"],
    7: ["translate", "translation", "bilingual", "localize", "in mandarin", "in spanish",
        "in french", "in german", "in japanese", "in korean", "in italian", "in portuguese",
        "multiple languages", "native korean speaker", "polite korean phrasing",
        "another language", "side by side"],
}

TEMPLATES = [
    "can you help me with {kw}",
    "I need help with {kw}",
    "please {kw} for me",
    "how do I approach {kw}",
    "what's the best way to handle {kw}",
    "give me some help on {kw}",
    "walk me through {kw}",
    "I'm stuck on {kw}, any ideas?",
]


def keyword_features(text, trigger_words):
    text_lower = text.lower()
    return [sum(text_lower.count(w) for w in words) for words in trigger_words.values()]


def main():
    df = pd.read_csv("data/prompts.csv")
    print(f"Loaded {len(df)} real examples from data/prompts.csv")

    synthetic_rows = []
    for label, words in TRIGGER_WORDS.items():
        for w in words:
            for t in TEMPLATES:
                synthetic_rows.append({"prompt": t.format(kw=w), "label": label})
    synthetic_df = pd.DataFrame(synthetic_rows)
    df = pd.concat([df, synthetic_df], ignore_index=True).drop_duplicates(subset="prompt").reset_index(drop=True)
    print(f"Total examples after augmentation: {len(df)}")

    MAX_FEATURES = 500
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X_tfidf = vectorizer.fit_transform(df["prompt"]).toarray()
    X_keywords = np.array([keyword_features(p, TRIGGER_WORDS) for p in df["prompt"]])
    X = np.hstack([X_tfidf, X_keywords])
    y = df["label"].values

    VOCAB_SIZE = X_tfidf.shape[1]
    NUM_KEYWORD_FEATURES = X_keywords.shape[1]
    INPUT_SIZE = X.shape[1]
    NUM_MODELS = len(MODEL_NAMES)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=16, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=16, shuffle=False)

    model = nn.Sequential(
        nn.Linear(INPUT_SIZE, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, NUM_MODELS),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}  |  Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    print(f"Test accuracy: {correct/total*100:.1f}% ({correct}/{total})")

    torch.save(model.state_dict(), "model.pt")
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("trigger_words.json", "w") as f:
        json.dump(TRIGGER_WORDS, f)
    with open("config.json", "w") as f:
        json.dump({
            "input_size": INPUT_SIZE,
            "vocab_size": VOCAB_SIZE,
            "num_keyword_features": NUM_KEYWORD_FEATURES,
            "num_models": NUM_MODELS,
            "model_names": MODEL_NAMES,
            "temperature": 2.4,
        }, f)
    print("\nSaved model.pt, vectorizer.pkl, trigger_words.json, config.json")


if __name__ == "__main__":
    main()