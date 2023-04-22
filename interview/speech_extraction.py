import whisper
import os

# Load the model
model = whisper.load_model("base")

file_name = "HoererapperatStig.m4a"
transciption = model.transcribe(str(os.getcwd())+f"/interview/{file_name}", fp16=False)["text"].lower()

print(transciption)

# Save the transcription to a text file
with open("transcription.txt", "w") as f:
    f.write(transciption)
