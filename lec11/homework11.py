import speech_recognition as sr

def transcribe_wavefile(filename, language):
    '''
    Use sr.Recognizer.AudioFile(filename) as the source,
    recognize from that source,
    and return the recognized text.
     
    @params:
    filename (str) - the filename from which to read the audio
    language (str) - the language of the audio
     
    @returns:
    text (str) - the recognized speech
    '''
    # 1. Initialize the recognizer
    r = sr.Recognizer()
    
    # 2. Open the audio file and record the data
    with sr.AudioFile(filename) as source:
        audio = r.record(source)
    
    # 3. Recognize using Google Speech API
    try:
        text = r.recognize_google(audio, language=language)
    except sr.UnknownValueError:
        # If audio is not understood, return empty string (still a string)
        text = ""
    except sr.RequestError:
        # If API is unreachable, return error message
        text = "API Error"
        
    return text
        
