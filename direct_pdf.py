# importing required modules
import speech_recognition as sr
from moviepy.editor import * #AudioFileClip
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("t5-base")

model = AutoModelWithLMHead.from_pretrained("t5-base")

device = torch.device('cpu')

audioclip = AudioFileClip('SP.mp4')
# getting video for only starting 10 seconds
clip = audioclip.subclip(0, 10)
abc=clip.write_audiofile("new_converted_trail.wav")  #convert video file into audio

r = sr.Recognizer()
#audio = sr.AudioFile("new_converted_trail.wav")
with clip as source:
    audio_file = r.record(source)
#audio_file=r.listen(clip)
result = r.recognize_google(audio_file)  #using speech recognition converting into text
    # encode the text into tensor of integers using the appropriate tokenizer
inputs = tokenizer.encode("summarize: " + str(result), return_tensors="pt", truncation=True)
    #Generate summary from training given text data on t5-base model
outputs = model.generate(inputs, max_length=1500, min_length=50, length_penalty=2.0, num_beams=5,
                            early_stopping=True)
out = tokenizer.decode(outputs[0])
o = out.split('>',1)
out1 = o[1].split('<')
print(out1)
    # print(tokenizer.decode(outputs[0]))
    #l = Label(text=out1[0], wraplength= 500,justify="center" ,font="comicsansms 13 bold", pady=15)
    #l.pack(fill=Y)
    #return out1[0].capitalize()
