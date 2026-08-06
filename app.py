"""
app.py
"""

import json
import os
import pickle

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()
app = Flask(__name__)

with open("config.json") as f:
    CONFIG = json.load(f)
with open("trigger_words.json") as f:
    TRIGGER_WORDS = json.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

MODEL_NAMES = {int(k): v for k, v in CONFIG["model_names"].items()}
INPUT_SIZE = CONFIG["input_size"]
NUM_MODELS = CONFIG["num_models"]
TEMPERATURE = CONFIG["temperature"]

classifier = nn.Sequential(
    nn.Linear(INPUT_SIZE, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, NUM_MODELS),
)
classifier.load_state_dict(torch.load("model.pt", map_location="cpu"))
classifier.eval()


def keyword_features(text):
    text_lower = text.lower()
    return [sum(text_lower.count(w) for w in words) for words in TRIGGER_WORDS.values()]


def classify(prompt_text):
    tfidf_vec = vectorizer.transform([prompt_text]).toarray()
    kw_vec = np.array([keyword_features(prompt_text)])
    vector = np.hstack([tfidf_vec, kw_vec])
    input_tensor = torch.tensor(vector, dtype=torch.float32)
    with torch.no_grad():
        output = classifier(input_tensor)
        probs = F.softmax(output / TEMPERATURE, dim=1).squeeze()
    scores = {MODEL_NAMES[i]: float(probs[i]) for i in range(NUM_MODELS)}
    best_model = max(scores, key=scores.get)
    return best_model, scores


PROVIDER_SPECS = {
    "Gemini": {
        "kind": "gemini",
        "native_key_env": "GEMINI_API_KEY",
        "submodels": [
            {"label": "3.1 Pro", "id": "gemini-3.1-pro"},
            {"label": "3.6 Flash", "id": "gemini-3.6-flash"},
            {"label": "3.5 Flash-Lite", "id": "gemini-3.5-flash-lite"},
        ],
        "default_submodel": "gemini-3.6-flash",
        "opencode": None,
    },
    "Claude": {
        "kind": "anthropic",
        "native_key_env": "ANTHROPIC_API_KEY",
        "submodels": [
            {"label": "Opus 5", "id": "claude-opus-5"},
            {"label": "Sonnet 5", "id": "claude-sonnet-5"},
            {"label": "Haiku 4.5", "id": "claude-haiku-4-5-20251001"},
            {"label": "Fable 5", "id": "claude-fable-5"},
        ],
        "default_submodel": "claude-sonnet-5",
        "opencode": None,
    },
    "ChatGPT": {
        "kind": "openai",
        "native_url": "https://api.openai.com/v1/chat/completions",
        "native_key_env": "OPENAI_API_KEY",
        "submodels": [
            {"label": "GPT-5.6 Sol", "id": "gpt-5.6-sol"},
            {"label": "GPT-5.6 Terra", "id": "gpt-5.6-terra"},
            {"label": "GPT-5.6 Luna", "id": "gpt-5.6-luna"},
            {"label": "GPT-5.5", "id": "gpt-5.5"},
        ],
        "default_submodel": "gpt-5.6-terra",
        "opencode": None,
    },
    "DeepSeek": {
        "kind": "openai",
        "native_url": "https://api.deepseek.com/chat/completions",
        "native_key_env": "DEEPSEEK_API_KEY",
        "submodels": [{"label": "DeepSeek Chat", "id": "deepseek-chat"}],
        "default_submodel": "deepseek-chat",
        "opencode": {
            "url": "https://opencode.ai/zen/go/v1/chat/completions",
            "submodels": [
                {"label": "DeepSeek V4 Flash", "id": "deepseek-v4-flash"},
                {"label": "DeepSeek V4 Pro", "id": "deepseek-v4-pro"},
            ],
            "default_submodel": "deepseek-v4-flash",
        },
    },
    "Perplexity": {
        "kind": "openai",
        "native_url": "https://api.perplexity.ai/chat/completions",
        "native_key_env": "PERPLEXITY_API_KEY",
        "submodels": [
            {"label": "Sonar", "id": "sonar"},
            {"label": "Sonar Pro", "id": "sonar-pro"},
            {"label": "Sonar Reasoning", "id": "sonar-reasoning"},
            {"label": "Sonar Reasoning Pro", "id": "sonar-reasoning-pro"},
            {"label": "Sonar Deep Research", "id": "sonar-deep-research"},
        ],
        "default_submodel": "sonar",
        "opencode": None,
    },
    "Grok": {
        "kind": "openai",
        "native_url": "https://api.x.ai/v1/chat/completions",
        "native_key_env": "XAI_API_KEY",
        "submodels": [{"label": "Grok 4", "id": "grok-4"}],
        "default_submodel": "grok-4",
        "opencode": {
            "url": "https://opencode.ai/zen/go/v1/chat/completions",
            "submodels": [{"label": "Grok 4.5", "id": "grok-4.5"}],
            "default_submodel": "grok-4.5",
        },
    },
    "Kimi": {
        "kind": "openai",
        "native_url": "https://api.moonshot.ai/v1/chat/completions",
        "native_key_env": "MOONSHOT_API_KEY",
        "submodels": [{"label": "Moonshot v1 128k", "id": "moonshot-v1-128k"}],
        "default_submodel": "moonshot-v1-128k",
        "opencode": {
            "url": "https://opencode.ai/zen/go/v1/chat/completions",
            "submodels": [
                {"label": "Kimi K3", "id": "kimi-k3"},
                {"label": "Kimi K2.7 Code", "id": "kimi-k2.7-code"},
                {"label": "Kimi K2.6", "id": "kimi-k2.6"},
                {"label": "Kimi K2.5", "id": "kimi-k2.5"},
            ],
            "default_submodel": "kimi-k3",
        },
    },
    "GLM": {
        "kind": "openai",
        "native_url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "native_key_env": "ZHIPU_API_KEY",
        "submodels": [{"label": "GLM-4.6", "id": "glm-4.6"}],
        "default_submodel": "glm-4.6",
        "opencode": {
            "url": "https://opencode.ai/zen/go/v1/chat/completions",
            "submodels": [
                {"label": "GLM 5.2", "id": "glm-5.2"},
                {"label": "GLM 5.1", "id": "glm-5.1"},
                {"label": "GLM 5", "id": "glm-5"},
            ],
            "default_submodel": "glm-5.2",
        },
    },
}


def call_openai_style(url, api_key, model, message):
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": message}]},
        timeout=30,
    )
    if resp.status_code != 200:
        detail = resp.json().get("error", {}).get("message", resp.text) if resp.text else resp.text
        raise RuntimeError(f"{resp.status_code}: {detail}")
    return resp.json()["choices"][0]["message"]["content"]


def call_gemini(api_key, model, message):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    resp = requests.post(
        url,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json={"contents": [{"parts": [{"text": message}]}]},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def call_anthropic(api_key, model, message):
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={"model": model, "max_tokens": 1024, "messages": [{"role": "user", "content": message}]},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


def chat_with_provider(provider_name, message, api_key_override=None, submodel_override=None, source="regular"):
    spec = PROVIDER_SPECS[provider_name]

    if source == "opencode":
        if not spec.get("opencode"):
            return f"{provider_name} isn't available through OpenCode Zen -- switch to Regular in Settings."
        api_key = api_key_override or os.environ.get("OPENCODE_API_KEY")
        if not api_key:
            return f"{provider_name} (OpenCode) isn't configured -- add a key in Settings or set OPENCODE_API_KEY in .env."
        valid_ids = [m["id"] for m in spec["opencode"]["submodels"]]
        model = submodel_override if submodel_override in valid_ids else spec["opencode"]["default_submodel"]
        try:
            return call_openai_style(spec["opencode"]["url"], api_key, model, message)
        except Exception as e:
            raise RuntimeError(f"{provider_name} (OpenCode) {e}")

    api_key = api_key_override or os.environ.get(spec["native_key_env"])
    if not api_key:
        return f"{provider_name} isn't configured -- add a key in Settings or set {spec['native_key_env']} in .env."
    valid_ids = [m["id"] for m in spec["submodels"]]
    model = submodel_override if submodel_override in valid_ids else spec["default_submodel"]

    try:
        if spec["kind"] == "gemini":
            return call_gemini(api_key, model, message)
        elif spec["kind"] == "anthropic":
            return call_anthropic(api_key, model, message)
        else:
            return call_openai_style(spec["native_url"], api_key, model, message)
    except requests.HTTPError as e:
        raise RuntimeError(f"{provider_name} {e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:
        raise RuntimeError(f"{provider_name}: {e}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json(silent=True) or {}
    prompt_text = (data.get("prompt") or "").strip()
    if not prompt_text:
        return jsonify({"error": "Please enter a prompt."}), 400
    best_model, scores = classify(prompt_text)
    return jsonify({"prompt": prompt_text, "best_model": best_model, "scores": scores})


@app.route("/chat", methods=["POST"])
def chat_route():
    data = request.get_json(silent=True) or {}
    model_name = data.get("model")
    message = (data.get("message") or "").strip()
    user_api_key = (data.get("api_key") or "").strip() or None
    submodel = (data.get("submodel") or "").strip() or None
    source = data.get("source") or "regular"
    if model_name not in PROVIDER_SPECS:
        return jsonify({"error": f"Unknown model: {model_name}"}), 400
    if not message:
        return jsonify({"error": "Please enter a message."}), 400
    try:
        reply = chat_with_provider(
            model_name, message,
            api_key_override=user_api_key,
            submodel_override=submodel,
            source=source,
        )
    except Exception as e:
        reply = f"Something went wrong talking to {model_name}: {e}"
    return jsonify({"model": model_name, "reply": reply})


@app.route("/provider_specs")
def provider_specs_route():
    out = {}
    for name, spec in PROVIDER_SPECS.items():
        out[name] = {
            "submodels": spec["submodels"],
            "default_submodel": spec["default_submodel"],
            "opencode": spec["opencode"],
        }
    return jsonify(out)


@app.route("/key_status")
def key_status():
    opencode_ready = bool(os.environ.get("OPENCODE_API_KEY"))
    out = {}
    for name, spec in PROVIDER_SPECS.items():
        out[name] = {
            "regular": bool(os.environ.get(spec["native_key_env"])),
            "opencode": opencode_ready if spec.get("opencode") else False,
        }
    return jsonify(out)


@app.route("/opencode_models", methods=["POST"])
def opencode_models_route():
    data = request.get_json(silent=True) or {}
    api_key = (data.get("api_key") or "").strip()
    if not api_key:
        return jsonify({"error": "Missing api_key"}), 400
    try:
        resp = requests.get(
            "https://opencode.ai/zen/go/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        resp.raise_for_status()
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 502


if __name__ == "__main__":
    app.run(debug=True, port=5050)