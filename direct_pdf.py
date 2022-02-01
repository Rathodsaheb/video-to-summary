# importing required modules
from flask import Flask,render_template, url_for
from flask import request

app = Flask(__name__)
@app.route("/", methods=['GET'])
def home():
    return render_template("index.html")

import speech_recognition as sr
from moviepy.editor import * #AudioFileClip
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("t5-base")

model = AutoModelWithLMHead.from_pretrained("t5-base")

device = torch.device('cpu')


def summary(raw):
    file_name = raw  #Taking input from user
    audioclip = AudioFileClip(file_name)
    audioclip.write_audiofile("new_converted_trail.wav")  #convert video file into audio

    r = sr.Recognizer()
    audio = sr.AudioFile("new_converted_trail.wav")
    with audio as source:
        audio_file = r.record(source)
    result = r.recognize_google(audio_file)  #using speech recognition converting into text
    # encode the text into tensor of integers using the appropriate tokenizer
    inputs = tokenizer.encode("summarize: " + str(result), return_tensors="pt", truncation=True)
    #Generate summary from training given text data on t5-base model
    outputs = model.generate(inputs, max_length=1500, min_length=50, length_penalty=2.0, num_beams=5,
                             early_stopping=True)
    out = tokenizer.decode(outputs[0])
    o = out.split('>',1)
    out1 = o[1].split('<')
    final_out = out1[0]
    # print(tokenizer.decode(outputs[0]))
    #l = Label(text=out1[0], wraplength= 500,justify="center" ,font="comicsansms 13 bold", pady=15)
    #l.pack(fill=Y)
    #return out1[0].capitalize()
    return final_out

@app.route("/home1", methods=['Get', 'POST'])
def index():
    errors = []
    results = {}
    if request.method == "POST":
        # get url that the user has entered
        try:
            num = request.form['myfile']
            r= num
        except:
            errors.append(
                "Unable to get URL. Please make sure it's valid and try again."
            )
            return render_template('home.html', errors=errors)
        if r:
            results1 = summary(r)
    return render_template('index.html',results1=results1, errors=errors)



if __name__ == "__main__":
    app.run(debug=True)
