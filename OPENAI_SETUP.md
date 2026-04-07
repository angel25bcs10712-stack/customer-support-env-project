# OpenAI API Key Setup Instructions

## 1. Get Your API Key
1. Visit: https://platform.openai.com/account/api-keys
2. Sign in with your OpenAI account (create one if you don't have it)
3. Click "Create new secret key"
4. Copy the API key immediately (you won't be able to see it again!)

## 2. Set Environment Variables
Run these commands in your terminal:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your_api_key_here"
$env:MODEL_NAME="gpt-4o-mini"
$env:API_BASE_URL="https://api.openai.com/v1"  # Optional, defaults to OpenAI

# Or set them permanently (Windows)
setx OPENAI_API_KEY "your_api_key_here"
setx MODEL_NAME "gpt-4o-mini"
```

## 3. Test Your Setup
```bash
python inference.py
```

## Important Notes
- Keep your API key secret and never commit it to version control
- You need credits in your OpenAI account to use the API
- Start with gpt-4o-mini for testing (cheaper than gpt-4)
- Monitor your API usage at https://platform.openai.com/usage