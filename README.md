AI Router

This application's purpose is to determine the best model to answer a user's simple or complex question, with a variety of models, 
both free and paid. The user starts by entering a prompt or topic they would like to discuss about. When the app has determined the best model to be used for the user's prompt, it will then proceed to open up that model in a chat within the app, 
provided to user has placed their API key in the settings.

Users may enter their API keys and corresponding models (ex: Claude Sonnet 4.6) in order to access chats to those models. 
The application also supports OpenCode Zen API keys. Keys are stored to the browser only for privacy purposes. 
The entire application is completely FREE standalone. Users just have to pay for their API keys.

Currently, we offer the following models, and plan to add more in the future, provided users have API keys to access them:
- DeepSeek (OpenCode supported)
- Grok (OpenCode Supported)
- GLM (OpenCode Supported)
- Kimi (OpenCode Supported)
- ChatGPT
- Claude
- Perplexity
- Gemini

## How to Start

### Checkout

```bash
git clone https://github.com/APandit1-cpu/AI_Routing.git
cd AI_Routing
```

### Prerequisites

- Python 3.10+ (with `pip`)
- The repo already includes the trained artifacts (`model.pt`, `vectorizer.pkl`, `trigger_words.json`, `config.json`), so no training is required to run the app.

Install the dependencies:

```bash
pip install -r requirements.txt
```

### API Keys (optional but recommended)

The app runs without keys, but chatting with a provider requires one. Keys can be entered in the app's **Settings** (stored in your browser only) or set in a `.env` file in the project root:

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
DEEPSEEK_API_KEY=...
PERPLEXITY_API_KEY=...
XAI_API_KEY=...            # Grok
MOONSHOT_API_KEY=...       # Kimi
ZHIPU_API_KEY=...          # GLM
OPENCODE_API_KEY=...       # OpenCode Zen (DeepSeek, Grok, Kimi, GLM)
```

### Command to Start

```bash
python app.py
```

Then open http://localhost:5050 in your browser.

### Retraining the Classifier (optional)

If you update `prompts.csv` or the trigger words, retrain with:

```bash
python train_model.py
```

### Screenshots

TODO
