from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle
import os
import re
import speech_recognition as sr

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# -----------------------------
# 1️⃣ Define custom metrics
# -----------------------------
def recall(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    return true_positives / (possible_positives + tf.keras.backend.epsilon())

def precision(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    return true_positives / (predicted_positives + tf.keras.backend.epsilon())

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

# -----------------------------
# 2️⃣ Register custom metrics
# -----------------------------
tf.keras.utils.get_custom_objects()['f1'] = f1
tf.keras.utils.get_custom_objects()['precision'] = precision
tf.keras.utils.get_custom_objects()['recall'] = recall

# -----------------------------
# 3️⃣ Load model and tokenizer
# -----------------------------
model = load_model("model.keras")

with open("tokenizer.pkl", "rb") as f:  # make sure the file name matches
    tokenizer = pickle.load(f)

MAXLEN = 100

# -----------------------------
# 4️⃣ In-memory storage for posts
# -----------------------------
posts = []

# -----------------------------
# 5️⃣ Clean and preprocess text
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# 6️⃣ Prediction function
# -----------------------------
def predict_hate(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAXLEN)
    pred = model.predict(padded)[0]
    
    result_class = pred.tolist().index(max(pred.tolist()))
    
    print(f"🧠 Text: {text}\n🔹 Prediction: Class {result_class} | Probabilities: {pred}")
    return result_class

# -----------------------------
# 7️⃣ Convert audio to text
# -----------------------------
def audio_to_text(filepath):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filepath) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return ""
        except sr.RequestError:
            return ""

# -----------------------------
# 8️⃣ Flask routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    global posts
    result = None
    text = ""

    if request.method == "POST":
        # Text input
        if "text" in request.form and request.form["text"].strip():
            text = request.form["text"].strip()

        # Audio input
        elif "audio" in request.files and request.files["audio"].filename != "":
            audio_file = request.files["audio"]
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], audio_file.filename)
            audio_file.save(filepath)
            text = audio_to_text(filepath)
            os.remove(filepath)

        # Predict and store
        if text:
            result = predict_hate(text)
            
            # Only save and display neutral posts (label 2)
            if result == 2:
                posts.append({"text": text, "label": result})

    return render_template("index.html", result=result, text=text, posts=posts)

# -----------------------------
# 9️⃣ Run app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
