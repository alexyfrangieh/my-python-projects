import google.generativeai as genai
import os

# Ensure your API key is configured (it should be, since you fixed it!)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("Listing available Gemini models and their capabilities:\n")

try:
    for m in genai.list_models():
        # Check if the model supports generateContent (for text/chat)
        # Or generateContent(image) for vision
        if 'generateContent' in m.supported_generation_methods:
            print(f"Model: {m.name}")
            print(f"  Description: {m.description}")
            print(f"  Supported methods: {m.supported_generation_methods}")
            print(f"  Input Token Limit: {m.input_token_limit}")
            print(f"  Output Token Limit: {m.output_token_limit}")
            print("-" * 30)
except Exception as e:
    print(f"An error occurred while listing models: {e}")
    print("Please check your API key and network connectivity.")
