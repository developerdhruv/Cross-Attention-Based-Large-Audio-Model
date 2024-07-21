from lam import LAM
text = ["this is a music of india and it is very popular"]
audio_path = 'test_file.wav'
model = LAM()
hello = model(text, audio_path)
print(hello)
