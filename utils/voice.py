import speech_recognition as sr
import pyttsx3


# Setup voice engine
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('utils', 'voices[0].id')


# Speaks a string of text
def speak(text):
    engine.say(text)
    engine.runAndWait()


# Listens to mic input
def listen(model):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)

        model_response = None

        try:
            statement = r.recognize_google(audio, language='en-in')
            print(f"User: {statement}\n")
            model_response = model.evaluate(statement)

        except Exception as e:
            model_response = "Pardon me, please say that again"

        # Print bot response and speak back to user
        print("Bot: " + model_response)
        speak(model_response)
