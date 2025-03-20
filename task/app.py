import torch
from flask import Flask, render_template, request, jsonify
from transformers import BertForSequenceClassification, AutoTokenizer

app = Flask(__name__)

# Configuration
teacher_model_name = "bert-base-uncased"
num_labels = 2
max_length = 128

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

# Load the teacher model to obtain its configuration
teacher_model = BertForSequenceClassification.from_pretrained(teacher_model_name, num_labels=num_labels)
# Modify the configuration for the student model (using 6 layers)
student_config = teacher_model.config
student_config.num_hidden_layers = 6
student_config.num_labels = num_labels  # Ensure the config contains the correct number of labels

# Initialize the student model using the modified configuration
model = BertForSequenceClassification(student_config)
# Load the saved weights (using strict=False to ignore missing keys like classifier weights)
model.load_state_dict(torch.load("model/student_model_lora.pth"), strict=False)
model.eval()  # Set the model to evaluation mode

def classify_text(text):
    """Tokenize and classify input text, returning 'Toxic' or 'Non-toxic'."""
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return "Toxic" if predicted_class == 1 else "Non-toxic"

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    text = ""
    if request.method == "POST":
        text = request.form.get("text", "")
        if text:
            result = classify_text(text)
    return render_template("index.html", result=result, text=text)

if __name__ == "__main__":
    app.run(debug=True)
