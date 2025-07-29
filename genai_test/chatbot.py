import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
chat = model.start_chat(history=[]) # Initialize chat history

print("Welcome to the Gemini Chatbot! Type 'exit' to quit.")

while True:
    user_message = input("You: ")
    if user_message.lower() == 'exit':
        break
    try:
        response = chat.send_message(user_message)
        print("Bot:", response.text)
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please try again.")

print("Goodbye!")
