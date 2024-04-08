from pydub import AudioSegment
from pydub.silence import split_on_silence

import speech_recognition as sr

import os 
import sys
import pydub
import argparse

class Transcriber:
    def __init__(self):
        self.__cwd = os.getcwd()
        self.__sr = sr.Recognizer()

    @staticmethod
    def parse_from_command_line():
        ap = argparse.ArgumentParser(
            description='A tool to transcribe video and audio files.')
        ap.add_argument('filename',
            help='File path for the source audio/video file')
        ap.add_argument('-o', '--output', default='transcript.txt',
            help='File path for the transcribed file.')
        ap.add_argument('-l', '--long', action='store_true',
            help="Transcribe longer files.")
        args = ap.parse_args()
        transcript = ""
        model = Transcriber()
        if args.long:
            filelist = model.split_audio_file(args.filename)
            transcript = model.transcribe_filelist(filelist)
        else:
            transcript = model.transcribe_file(args.filename)
        with open(args.output, 'w') as output_file:
            output_file.write(transcript)

    def split_audio_file(self, filepath, min_silence_len=500, keep_silence=500):
        sound = AudioSegment.from_file(filepath)

        chunks = split_on_silence(
            sound,
            min_silence_len = min_silence_len,
            silence_thresh = sound.dBFS-14,
            keep_silence=keep_silence
        )

        directory = os.path.join(self.__cwd, "audio")
        if not os.path.isdir(directory):
            os.mkdir(directory)

        filelist = []
        for i, audio_chunk in enumerate(chunks, start=1):
            chunk_filename = os.path.join(directory, f"audio-chunk-{i}.wav")
            audio_chunk.export(chunk_filename, format="wav")
            filelist.append(chunk_filename)
        return filelist

    def transcribe_file(self, filename):
        text = ""
        with sr.AudioFile(filename) as source:
            audio_listened = self.__sr.record(source)
            try:
                text = self.__sr.recognize_whisper(audio_listened)
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}. "
        return text

    def transcribe_filelist(self, filelist:str|list):
        if isinstance(filelist, str):
            filelist = [str]

        transcribed_text = ""
        for filename in filelist:
            transcribed_text += self.trasncribe_file(filename)
        return transcribed_text

if __name__ == '__main__':
    Transcriber.parse_from_command_line()