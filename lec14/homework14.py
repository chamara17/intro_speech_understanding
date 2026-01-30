from gtts import gTTS
import speech_recognition as sr
import librosa
import soundfile as sf
import os

def synthesize(text, language, filename):
    '''
    Use gTTS to synthesize text and save to filename.
    
    @params:
    text (str) - the text to synthesize
    language (str) - the language code (e.g., 'en', 'ja')
    filename (str) - the filename to save the audio to
    '''
    # Initialize Google Text-to-Speech with the text and language
    tts = gTTS(text=text, lang=language)
    # Save the file to the specified filename
    tts.save(filename)

def make_a_corpus(texts, languages, filenames):
    '''
    Synthesize a list of texts to files, convert them to wav, 
    and then recognize them to check accuracy.
    
    @params:
    texts (list) - list of strings to say
    languages (list) - list of language codes
    filenames (list) - list of base filenames (without extensions)
    
    @returns:
    recognized_texts (list) - list of recognized strings
    '''
    r = sr.Recognizer()
    recognized_texts = []
    
    # Iterate through the lists simultaneously
    for text, lang, fname in zip(texts, languages, filenames):
        
        # 1. Define filenames
        # The test passes 'testfile', so we add extensions
        mp3_filename = fname + ".mp3"
        wav_filename = fname + ".wav"
        
        # 2. Synthesize speech to MP3
        # We reuse our synthesize function
        synthesize(text, lang, mp3_filename)
        
        # 3. Convert MP3 to WAV
        # Librosa loads the MP3, Soundfile writes the WAV
        y, fs = librosa.load(mp3_filename, sr=None)
        sf.write(wav_filename, y, fs)
        
        # 4. Recognize the speech
        with sr.AudioFile(wav_filename) as source:
            audio = r.record(source)
            try:
                # Use Google recognizer
                rec_text = r.recognize_google(audio, language=lang)
            except sr.UnknownValueError:
                rec_text = ""
            except sr.RequestError:
                rec_text = "API Error"
                
        recognized_texts.append(rec_text)
        
    return recognized_texts