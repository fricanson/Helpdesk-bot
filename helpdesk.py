import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")# Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')

# Load Hugging Face models
def load_intent_model():
    tokenizer = AutoTokenizer.from_pretrained("Venkatesh4342/distilroberta-helpdesk-intent")
    model = AutoModelForSequenceClassification.from_pretrained("Venkatesh4342/distilroberta-helpdesk-intent")
    return tokenizer, model

def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained("Venkatesh4342/distilbert-helpdesk-sentence-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("Venkatesh4342/distilbert-helpdesk-sentence-sentiment")
    return tokenizer, model

intent_tokenizer, intent_model = load_intent_model()
sentiment_tokenizer, sentiment_model = load_sentiment_model()

# Prediction functions
def predict_intent(text):
    inputs = intent_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = intent_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    intent_id = probs.argmax().item()
    return intent_model.config.id2label[intent_id]

def predict_sentiment(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = sentiment_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_id = probs.argmax().item()
    return sentiment_model.config.id2label[sentiment_id]

# Gemini AI function
def generate_response(intent, sentiment, query):
    prompt = f"""
    As a helpdesk assistant for a company that provides water pumps, water treatment, swimming pools, and renewable energy services, respond to the following query:
    Query: {query}
    Detected Intent: {intent}
    Detected Sentiment: {sentiment}
    Provide a helpful and appropriate response based on the intent and sentiment. Include specific information about our services:
    1. Water Pumps: We offer a range of water pumps for industrial, commercial, and domestic use.
    2. Water Treatment: We supply equipment and solutions for treating water, removing contaminants, and making it safe for various purposes.
    3. Swimming Pools: We provide products and services for designing, building, and maintaining swimming pools.
    4. Renewable Energy: We offer solar solutions, generators, and other renewable energy products.
    Tailor your response to address the specific service area mentioned or implied in the query.
    """
    response = gemini_model.generate_content(prompt)
    return response.text

class ChatApp:
    def __init__(self, master):
        self.master = master
        master.title("Davis & Shirtliff Helpdesk")
        master.geometry("600x400")

        self.chat_display = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=70, height=20)
        self.chat_display.pack(padx=10, pady=10)
        self.chat_display.config(state=tk.DISABLED)

        self.input_field = tk.Entry(master, width=50)
        self.input_field.pack(side=tk.LEFT, padx=10)
        self.input_field.bind("<Return>", self.send_message)

        self.send_button = tk.Button(master, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.LEFT)

        self.display_message("Welcome to Davis & Shirtliff Helpdesk!\nWe specialize in Water Pumps, Water Treatment, Swimming Pools, and Renewable Energy solutions.\nHow can I assist you today?", "Assistant")

    def display_message(self, message, sender):
        self.chat_display.config(state=tk.NORMAL)
        if sender == "User":
            self.chat_display.insert(tk.END, f"You: {message}\n\n", "user")
        elif sender == "Assistant":
            self.chat_display.insert(tk.END, f"Assistant: {message}\n\n", "assistant")
        else:
            self.chat_display.insert(tk.END, f"{sender}: {message}\n\n", "system")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def send_message(self, event=None):
        user_input = self.input_field.get()
        if user_input:
            self.display_message(user_input, "User")
            self.input_field.delete(0, tk.END)
            threading.Thread(target=self.process_input, args=(user_input,)).start()

    def process_input(self, user_input):
        intent = predict_intent(user_input)
        sentiment = predict_sentiment(user_input)
        self.display_message(f"Detected Intent: {intent}", "System")
        self.display_message(f"Detected Sentiment: {sentiment}", "System")
        response = generate_response(intent, sentiment, user_input)
        self.display_message(response, "Assistant")

def main():
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()