# main.py

from fastapi import FastAPI, HTTPException, Form
import json
import torch
import random
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import subprocess

app = FastAPI()

# Load the chatbot model and data
model = None
FILE = "data.pth"
intents = None
previous_user_input = None

# Load the model and intents when the app starts
@app.on_event("startup")
async def load_model_and_intents():
    global model, intents, input_size, hidden_size, output_size, all_words, tags, model_state

    data = torch.load(FILE)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size)
    model.load_state_dict(model_state)
    model.eval()

    with open('intents.json', 'r') as json_data:
        intents = json.load(json_data)

# Function to handle the training process
def handle_training(user_main_input, user_provided_info):
    global previous_user_input

    existing_intent = next((intent for intent in intents['intents'] if user_main_input in intent['patterns']), None)

    if existing_intent:
        if user_provided_info not in existing_intent['responses']:
            existing_intent['responses'].append(user_provided_info)
    else:
        new_intent = {
            "tag": f"user_defined_{len(intents['intents']) + 1}",
            "patterns": [user_main_input],
            "responses": [user_provided_info]
        }
        intents['intents'].append(new_intent)

    with open('intents.json', 'w') as json_file:
        json.dump(intents, json_file, indent=2)

    print("Rerunning train.py to update the model...")
    subprocess.run(["python", "train.py"])

    previous_user_input = None

# API endpoint to handle both chatting and teaching
@app.post("/chat")
async def chat(message: str = Form(...), teach: bool = Form(False)):
    global model, input_size, hidden_size, output_size, all_words, tags, model_state, intents, previous_user_input

    if not model or not intents:
        raise HTTPException(status_code=500, detail="Chatbot model not loaded. Please try again later.")

    sentence = message

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])

    # Process the user's message using the chatbot model
    output = model(torch.from_numpy(X))
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return {"response": random.choice(intent['responses'])}
    else:
        if teach and previous_user_input:
            handle_training(previous_user_input, message)
            return {"response": "Thank you for the information. I'll use that to improve my understanding."}
        else:
            previous_user_input = message
            return {"response": "I'm still learning. Could you provide more details or teach me about it?"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
