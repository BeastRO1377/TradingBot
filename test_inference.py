# test_inference_gguf.py (This loads the single, optimized GGUF file)
from mlx_lm import load, generate
import json

# --- THIS IS THE ONLY CHANGE THAT MATTERS ---
# We are loading the self-contained, quantized GGUF file you already created.
MODEL_PATH = "./trading-llama.gguf"

def get_llama_response(prompt: str) -> dict:
    print(f"Loading the single, optimized GGUF model from: {MODEL_PATH}...")
    
    # The mlx_lm.load function is designed to handle GGUF files directly.
    # It will be fast and memory-efficient.
    model, tokenizer = load(MODEL_PATH)
    
    print("Optimized model loaded successfully.")
    print("Generating response...")

    # The generate call is simple. It will be fast because the model is small.
    response_text = generate(model, tokenizer, prompt, verbose=False, max_tokens=256)
    
    print("Generation complete.")

    try:
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError("JSON object not found in the response")
            
        json_part = response_text[start:end]
        return json.loads(json_part)
    except Exception as e:
        print(f"--- ERROR: Failed to parse JSON from model output ---")
        print(f"Error details: {e}")
        print(f"Raw model response received:\n---\n{response_text}\n---")
        return {"error": "Failed to parse JSON", "raw_response": response_text}

# --- MAIN TEST ---
if __name__ == "__main__":
    test_prompt = """
ПОЛЬЗОВАТЕЛЬ:
Проанализируй торговый сигнал.

Данные сигнала:
{
  "symbol": "ETHUSDT",
  "side": "Buy",
  "source": "squeeze",
  "metrics": {
    "pct_5m": 4.2,
    "vol_change_pct": 380.0
  }
}

ЗАДАЧА:
Верни JSON с ключами "confidence_score", "justification", и "action" ("EXECUTE" или "REJECT").
"""
    
    result = get_llama_response(test_prompt)
    
    print("\n--- Parsed JSON Response ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))