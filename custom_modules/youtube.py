from youtube_dl import YoutubeDL
import subprocess
import argparse
import os


class YoutubeDownloader:
    AUDIO_CONFIG = {
        'format': 'bestaudio/best',
        'outtmpl': '%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '256'
        }]
    }

    VIDEO_CONFIG = {
        'outtmpl': '%(title)s.%(ext)s'
    }

    @staticmethod
    def parse_from_command_line():
        ap = argparse.ArgumentParser(
            description='A tool to download youtube files as video or audio.')
        ap.add_argument('-f', '--file', default='%(title)s.%(ext)s',
            help='File name for the downloaded file.')
        ap.add_argument('-d', '--dir', default=None,
            help='Optional directory to store downloaded file.')
        ap.add_argument('-u', '--url', required=True,
            help='Youtube URL to download.')
        ap.add_argument('-a', '--audio', action='store_true',
            help="Add flag to download only mp3 audio")
        ap.add_argument('-na', '--no-audio', action='store_true',
            help="Add flag to download video without audio")
        args = ap.parse_args()
        if args.audio:
            YoutubeDownloader.download_audio(args.url, args.file, args.dir)
        else:
            YoutubeDownloader.download_video(args.url, args.file, args.dir, args.no_audio)

    @staticmethod
    def get_audio_config(filename='%(title)s.%(ext)s', directory=None):
        filepath = YoutubeDownloader._get_filepath(filename, directory)
        config =  YoutubeDownloader.AUDIO_CONFIG.copy()
        config['outtmpl'] = filepath
        if not filename.endswith('%(ext)s'):
            config['postprocessors'][0]['preferredcodec'] = filename.split('.')[-1]
        return config

    @staticmethod
    def get_video_config(filename='%(title)s.%(ext)s', directory=None):
        filepath = YoutubeDownloader._get_filepath(filename, directory)
        config = YoutubeDownloader.VIDEO_CONFIG.copy()
        config['outtmpl'] = filepath
        return config

    @staticmethod
    def _get_filepath(filename='%(title)s.%(ext)s', directory=None):
        if directory is None:
            return filename
        return os.path.join(directory, filename)

    @staticmethod
    def _get_object(url=None, config=VIDEO_CONFIG):
        with YoutubeDL(config) as library:
            library.download([url])
            
    @staticmethod
    def _get_metadata(url=None, config=VIDEO_CONFIG):
        with YoutubeDL(config) as library:
            return library.extract_info(url, download=False)

    @staticmethod
    def _merge_video_and_audio(audio_file, video_file, output_file):
        mix_command = f'ffmpeg -y -i "{audio_file}" -i "{video_file}" -c:v \
            copy -c:a aac -strict experimental "{output_file}"'
        subprocess.call(mix_command, shell=True)

    @staticmethod
    def download_audio(url=None, filename='%(title)s.%(ext)s', directory=None):
        config = YoutubeDownloader.get_audio_config(filename, directory)
        YoutubeDownloader._get_object(url, config)

    @staticmethod
    def download_video(url=None, filename='%(title)s.%(ext)s', directory=None, 
            exclude_audio=False):
        output_file = YoutubeDownloader._get_filepath(filename, directory)
        config = YoutubeDownloader.get_video_config(filename, directory)
        if exclude_audio:
            config['outtmpl'] = output_file
            YoutubeDownloader._get_object(url, config)
            return
        f_name = f"temp-{os.getpid()}"
        metafields = ['title', 'ext']
        print(url)
        metadata = YoutubeDownloader._get_metadata(url)
        print("-"*50)
        for field in metafields:
            output_file = output_file.replace(f'%({field})s', metadata[field])
        output_file = output_file.replace()
        audio_config = YoutubeDownloader.get_audio_config(filename=audio_file)
        video_config = YoutubeDownloader.get_video_config(filename=video_file)
        YoutubeDownloader._get_object(url, audio_config)
        YoutubeDownloader._get_object(url, video_config)
        YoutubeDownloader._merge_video_and_audio(audio_file, video_file, output_file)
        os.remove(audio_file)
        os.remove(video_file)      

if __name__ == '__main__':
    YoutubeDownloader.parse_from_command_line()