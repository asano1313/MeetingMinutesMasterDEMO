import streamlit as st
import os
import ffmpeg
import google.generativeai as genai
import time
from threading import Thread
from queue import Queue
from deepgram import DeepgramClient, PrerecordedOptions
import os
import json
from pydub import AudioSegment
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
import shutil

os.environ['DEEPGRAM_API_KEY'] = st.secrets.DEEPGRAM_API_KEY.key
os.environ['OPENAI_API_KEY'] = st.secrets.OPENAI_API_KEY.key


def get_audio_duration(file_path):
    audio = AudioSegment.from_file(file_path)
    duration_in_seconds = len(audio) / 1000.0  # ミリ秒から秒に変換
    duration_in_minuetes = duration_in_seconds / 60
    return duration_in_minuetes


#コンバータークラス
class FFmpegM4AConverter:
    """
    FFmpegを使用してさまざまな形式のオーディオファイルをM4Aファイルに変換するクラス。

    使用方法:
    1. インスタンスを作成します。必要に応じて、オーディオ設定を指定できます。
       シンプルな使い方
       converter = FFmpegM4AConverter()

       詳細な使い方
       converter = FFmpegM4AConverter(sample_rate=48000, bitrate=256000, channels=2, bits_per_sample=16)

    2. convertメソッドまたは__call__メソッドを使用して、ファイルを変換します。
       シンプルな使い方
       converter("input.wav", "output_directory")  # デフォルトでnormalizeが適用されます
       converter.convert("input.wav", "output_directory")  # デフォルトでnormalizeが適用されます

       詳細な使い方
       converter("input.mp3", "output_directory", normalize=False, vbr=True, metadata={"artist": "John Doe", "title": "Example"})

    対応する入力ファイル形式:
    - オーディオ形式: .aac, .ac3, .aif, .aiff, .alac, .amr, .ape, .flac, .m4a, .mp3, .ogg, .opus, .wav, ...
    - ビデオ形式: .avi, .flv, .mkv, .mov, .mp4, .mpeg, .webm, .wmv, ...

    変換されたM4Aファイルは、指定された出力ディレクトリに保存されます。
    """

    DEFAULT_SAMPLE_RATE = 44100
    DEFAULT_BITRATE = 192000
    DEFAULT_CHANNELS = 1
    DEFAULT_BITS_PER_SAMPLE = 16
    DEFAULT_ADJUST_VOLUME = True
    DEFAULT_TARGET_VOLUME = -10

    def __init__(self, sample_rate=None, bitrate=None, channels=None, bits_per_sample=None, adjust_volume=None, target_volume=None):
        self.sample_rate = sample_rate or self.DEFAULT_SAMPLE_RATE
        self.bitrate = bitrate or self.DEFAULT_BITRATE
        self.channels = channels or self.DEFAULT_CHANNELS
        self.bits_per_sample = bits_per_sample or self.DEFAULT_BITS_PER_SAMPLE
        self.adjust_volume = adjust_volume if adjust_volume is not None else self.DEFAULT_ADJUST_VOLUME
        self.target_volume = target_volume or self.DEFAULT_TARGET_VOLUME
        self.supported_extensions = self._get_supported_extensions()

    def _get_supported_extensions(self):
        return [
            '.3g2', '.3gp', '.aac', '.ac3', '.aif', '.aiff', '.alac', '.amr', '.ape',
            '.asf', '.au', '.avi', '.caf', '.dts', '.dtshd', '.dv', '.eac3', '.flac',
            '.flv', '.m2a', '.m2ts', '.m4a', '.m4b', '.m4p', '.m4r', '.m4v', '.mka',
            '.mkv', '.mod', '.mov', '.mp1', '.mp2', '.mp3', '.mp4', '.mpa', '.mpc',
            '.mpeg', '.mpg', '.mts', '.nut', '.oga', '.ogg', '.ogm', '.ogv', '.ogx',
            '.opus', '.ra', '.ram', '.rm', '.rmvb', '.shn', '.spx', '.tak', '.tga',
            '.tta', '.vob', '.voc', '.wav', '.weba', '.webm', '.wma', '.wmv', '.wv',
            '.y4m', '.aac', '.aif', '.aiff', '.aiffc', '.flac', '.iff', '.m4a', '.m4b',
            '.m4p', '.mid', '.midi', '.mka', '.mp3', '.mpa', '.oga', '.ogg', '.opus',
            '.pls', '.ra', '.ram', '.spx', '.tta', '.voc', '.vqf', '.w64', '.wav',
            '.wma', '.xm', '.3gp', '.a64', '.ac3', '.amr', '.drc', '.dv', '.flv',
            '.gif', '.h261', '.h263', '.h264', '.hevc', '.m1v', '.m4v', '.mkv', '.mov',
            '.mp2', '.mp4', '.mpeg', '.mpeg1video', '.mpeg2video', '.mpeg4', '.mpg',
            '.mts', '.mxf', '.nsv', '.nuv', '.ogg', '.ogv', '.ps', '.rec', '.rm',
            '.rmvb', '.roq', '.svi', '.ts', '.vob', '.webm', '.wmv', '.y4m', '.yuv'
        ]

    def _apply_filters(self, stream, normalize=False, equalizer=None):
        if normalize:
            stream = ffmpeg.filter(stream, 'dynaudnorm')
        if equalizer:
            stream = ffmpeg.filter(stream, 'equalizer', equalizer)
        return stream

    def _analyze_volume(self, input_file):
        try:
            stats = ffmpeg.probe(input_file)
            audio_stats = next((s for s in stats['streams'] if s['codec_type'] == 'audio'), None)
            if audio_stats:
                volume_mean = float(audio_stats['tags']['volume_mean'])
                volume_max = float(audio_stats['tags']['volume_max'])
                return volume_mean, volume_max
            else:
                print("No audio stream found in the input file.")
        except ffmpeg.Error as e:
            print(f"Error occurred during volume analysis: {e.stderr}")
        return None, None

    def _adjust_volume(self, stream, volume_mean, volume_max, target_volume):
        if volume_mean is not None and volume_max is not None:
            volume_adjustment = target_volume - volume_max
            stream = ffmpeg.filter(stream, 'volume', volume=f'{volume_adjustment}dB')
        return stream

    def _convert(self, input_file, output_path, normalize=False, equalizer=None, vbr=False, metadata=None):
        stream = ffmpeg.input(input_file)

        if normalize:
            stream = self._apply_filters(stream, normalize=True)
        else:
            if self.adjust_volume:
                volume_mean, volume_max = self._analyze_volume(input_file)
                if volume_mean is not None and volume_max is not None:
                    stream = self._adjust_volume(stream, volume_mean, volume_max, self.target_volume)

        stream = self._apply_filters(stream, equalizer=equalizer)

        kwargs = {
            'acodec': 'aac',
            'ar': self.sample_rate,
            'ac': self.channels,
        }
        if vbr:
            kwargs['vbr'] = 5
        else:
            kwargs['b:a'] = self.bitrate

        output_stream = ffmpeg.output(stream, output_path, **kwargs)

        try:
            # '-y' オプションを追加して、出力ファイルの自動上書きを許可
            ffmpeg.run(output_stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            print("Conversion completed successfully.")
        except ffmpeg.Error as e:
            stdout = e.stdout.decode('utf-8') if e.stdout else "No stdout"
            stderr = e.stderr.decode('utf-8') if e.stderr else "No stderr"
            print(f"Error occurred during conversion: {stderr}")
            print(f"FFmpeg stdout: {stdout}")

    def convert(self, input_file, output_dir, normalize=True, equalizer=None, vbr=False, metadata=None):
        _, extension = os.path.splitext(input_file)
        if extension.lower() not in self.supported_extensions:
            raise ValueError(f"Unsupported file extension: {extension}")

        output_file = os.path.splitext(os.path.basename(input_file))[0] + ".m4a"
        output_path = os.path.join(output_dir, output_file)
        os.makedirs(output_dir, exist_ok=True)

        self._convert(input_file, output_path, normalize, equalizer, vbr, metadata)
        return output_path

    def __call__(self, input_file, output_dir, normalize=True, equalizer=None, vbr=False, metadata=None):
        return self.convert(input_file, output_dir, normalize, equalizer, vbr, metadata)


#文字おこし
# audio_file_path = os.path.join(folder_path, FILE_NAME)
# json_file_path = os.path.join(folder_path, 'deepgram.json')
# transcript_file_path = os.path.join(folder_path, 'transcript.txt')

class DeepgramTranscriber:
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
      raise ValueError("api_key is required. set DEEPGRAM_API_KEY")

    def __init__(self, audio_file_path, json_file_path=None, transcript_file_path=None):
        self.audio_file_path = audio_file_path
        self.json_file_path = json_file_path
        self.transcript_file_path = transcript_file_path
        self.deepgram = DeepgramClient(self.api_key)

    def __call__(self):
        if self.json_file_path is None or self.transcript_file_path is None:
            return self.transcribe_with_no_save()
        else:
            return self.transcribe_with_save()

    def transcribe(self):
        print(self.audio_file_path)
        print(os.path.isfile(self.audio_file_path))
        with open(self.audio_file_path, 'rb') as buffer_data:
            payload = {'buffer': buffer_data}
            options = PrerecordedOptions(
                punctuate=True,
                model="nova-2",
                language="ja",
                # diarize=True,
                # utterances=True,
                # smart_format = True,
            )
            print('Requesting transcript...')
            print('Your file may take up to a couple minutes to process.')
            print('While you wait, did you know that Deepgram accepts over 40 audio file formats? Even MP4s.')
            print('To learn more about customizing your transcripts check out developers.deepgram.com')
            response = self.deepgram.listen.prerecorded.v('1').transcribe_file(payload, options)
            return response

    def transcribe_with_no_save(self):
        response = self.transcribe()
        # for u in response.results.utterances:
        #   print(f"[Speaker: {u.speaker}] {u.transcript}")
        return response.results.channels[0].alternatives[0].transcript

    def transcribe_with_save(self):
        response = self.transcribe()
        self._save_json(response)
        self._save_transcript(response)
        return response.results.channels[0].alternatives[0].transcript

    def _save_json(self, response):
        with open(self.json_file_path, 'w') as outfile:
            json.dump(response.to_json(indent=4), outfile)

    def _save_transcript(self, response):
        transcript = response.results.channels[0].alternatives[0].transcript
        with open(self.transcript_file_path, 'w') as outfile:
            outfile.write(transcript)

    def read_json(self):
        with open(self.json_file_path, 'r') as infile:
            return json.load(infile)

    def read_transcript(self):
        with open(self.transcript_file_path, 'r') as infile:
            return infile.read()
        
def transcribe(m4a_path):
    transcriber = DeepgramTranscriber(m4a_path)
    transcript = transcriber()
    return transcript


st.set_page_config(
    page_title="デモ",
    layout="wide", # wideにすると横長なレイアウトに
    initial_sidebar_state="expanded"
)

st.title("デモ")

st.sidebar.markdown("# ファイルアップロード")
uploaded_file = st.sidebar.file_uploader(
    "動画or音声ファイルをアップロードしてください", type=["mp4", "m4a"]
)

col1, col2, col3 = st.columns(3)

with col1:
    st.header("Gemini1.5")

with col2:
    st.header("G1.5F")

with col3:
    st.header("Gpt4o")

def genearate(audio_file, model_name, queue):

    genai.configure(api_key=st.secrets.GEMINI_API_KEY.key)
    audio_file = genai.upload_file(path=audio_file)

    while audio_file.state.name == "PROCESSING":
        st.write("wait")
        time.sleep(10)
        audio_file = genai.get_file(audio_file.name)

    if audio_file.state.name == "FAILED":
        raise ValueError(audio_file.state.name)
    st.write(audio_file.uri)        

    model = genai.GenerativeModel(
        model_name=model_name
    )

    prompt = """
面談音声の中から、私が指定する項目の内容で、議事録を作成してもらいます。
■導入から始めてください。
###指定する項目
■導入
・商談背景(なぜ今回の商談が行われたか)


■質問事項
・採用背景
・採用計画
・採用計画に対する進捗状況
・採用に伴う課題

■採用手法
・今までの採用手法
・現在利用している他社サービス
・他社サービスを利用しての所感

■条件
・給与
・年齢
・転職回数
・性別
・学歴
・資格
・その他

■求める人物像
・性格
・持ち合わせる経験
・活躍する人の傾向

■事業内容・業務内容
・事業内容
・具体的な業務
・１日の流れ

■会社風土
・設立背景
・どのような方が活躍するか
・会社の雰囲気・文化
・他社との違い


■選考フロー
・選考回数
・選考の流れ


■契約内容
・手数料
・返金規定

■次回アクション


■その他ユニークポイント
・他社との違いやアピールポイント等

■AI分析

・うまく関係値を気づけたか(アイスブレイク)
・自社の紹介、他社との違いを説明できたか(自社・自己紹介)
・相手の課題をヒアリングできたか(ヒアリング)
・サービスをしっかり提案できたいか(サービス提案)
・クロージングを行えたか(クロージング)
・次回の具体的なアクションを決めれたか(アクション設定)

■良かった点

■改善点

■まとめ

"""
    input_token = model.count_tokens([prompt, audio_file]).total_tokens
    response = model.generate_content(
        [prompt, audio_file], stream=True
    )

    genai.delete_file(audio_file.name)


    minutes = ""
    for chunk in response:
        minutes += chunk.text
        queue.put(minutes)
        time.sleep(0.05)

    output_token = response.usage_metadata.candidates_token_count

    if model_name == "models/gemini-1.5-pro-latest":
        st.write(1)
        input_price = 0.0000035 * input_token
        output_price = 0.0000105 * output_token
        price = input_price + output_price
        minutes += f"\n ## price(USD): ${format(price, '.6f')}"
        time.sleep(0.05)
        queue.put(minutes)
        time.sleep(0.05)
    if model_name == "models/gemini-1.5-flash-latest":
        st.write(2)
        input_price = 0.00000035 * input_token
        output_price = 0.00000105 * output_token
        price = input_price + output_price
        minutes += f"\n ## price(USD): ${format(price, '.6f')}"
        time.sleep(0.05)
        queue.put(minutes)
        time.sleep(0.05)       

    return "DONE"

def generate_gpt4o(m4a_path, model_name, queue):
    tran = transcribe(m4a_path=m4a_path)
    chat_client = ChatOpenAI(model=model_name, temperature=0)
    system_prompt = f"""
    以下の、面談音声の文字おこしの中から、私が指定する項目の内容で、議事録を作成してもらいます。
    ■導入から始めてください。
    ###面談音声
    {tran}
    ###指定する項目
    ■導入
    ・商談背景(なぜ今回の商談が行われたか)

    ■質問事項
    ・採用背景
    ・採用計画
    ・採用計画に対する進捗状況
    ・採用に伴う課題

    ■採用手法
    ・今までの採用手法
    ・現在利用している他社サービス
    ・他社サービスを利用しての所感

    ■条件
    ・給与
    ・年齢
    ・転職回数
    ・性別
    ・学歴
    ・資格
    ・その他

    ■求める人物像
    ・性格
    ・持ち合わせる経験
    ・活躍する人の傾向

    ■事業内容・業務内容
    ・事業内容
    ・具体的な業務
    ・１日の流れ

    ■会社風土
    ・設立背景
    ・どのような方が活躍するか
    ・会社の雰囲気・文化
    ・他社との違い


    ■選考フロー
    ・選考回数
    ・選考の流れ


    ■契約内容
    ・手数料
    ・返金規定

    ■次回アクション


    ■その他ユニークポイント
    ・他社との違いやアピールポイント等

    ■AI分析

    ・うまく関係値を気づけたか(アイスブレイク)
    ・自社の紹介、他社との違いを説明できたか(自社・自己紹介)
    ・相手の課題をヒアリングできたか(ヒアリング)
    ・サービスをしっかり提案できたいか(サービス提案)
    ・クロージングを行えたか(クロージング)
    ・次回の具体的なアクションを決めれたか(アクション設定)

    ■良かった点

    ■改善点

    ■まとめ
    """


    message=[("system", system_prompt)]
    deepgram_price = get_audio_duration(m4a_path) * 0.0043
    with get_openai_callback() as cb:
        response = chat_client.invoke(message)
        total_cost = deepgram_price + cb.total_cost
        cost = f"\n ## Price(USD): ${format(total_cost, '.6f')}"

    minutes = ""
    minutes += response.content
    minutes += cost
    queue.put(minutes)

    return "DONE"


if uploaded_file is not None:
    file_name = uploaded_file.name
    file_extension = os.path.splitext(file_name)[1]

    if file_extension != ".m4a":
        with st.spinner('音声ファイルに変換中...'):
            status_text = st.empty()
            save_dir = "uploaded_files"
            os.makedirs(save_dir, exist_ok=True)
            file_path = os.path.join(save_dir, file_name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            converter = FFmpegM4AConverter()
            inputfile = file_path
            outputfile = save_dir
            m4a_path = converter(file_path, outputfile)
        # status_text.success('処理が完了しました!')  
    else :
        save_dir = "uploaded_files"
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        m4a_path = file_path

    # audio_time = get_audio_duration(m4a_path)
    # st.write(audio_time)

    queue1 = Queue()
    queue2 = Queue()
    queue3 = Queue()
    future1 = Thread(target=genearate, args=(m4a_path, "models/gemini-1.5-pro-latest", queue1))
    future2 = Thread(target=genearate, args=(m4a_path, "models/gemini-1.5-flash-latest", queue2))
    future3 = Thread(target=generate_gpt4o, args=(m4a_path, "gpt-4o-2024-05-13", queue3))



    future1.start()
    future2.start()
    future3.start()


    minutes_container1 = col1.empty()
    minutes_container2 = col2.empty()
    minutes_container3 = col3.empty()

    while future1.is_alive() or future2.is_alive():
        if not queue1.empty():
            minutes_container1.write(queue1.get())
        if not queue2.empty():
            minutes_container2.write(queue2.get()) 
        if not queue3.empty():
            minutes_container3.write(queue3.get()) 
    future1.join()
    future2.join()
    future3.join()

    directory = 'uploaded_files'

    # ディレクトリが存在するか確認
    if os.path.exists(directory):
        # ディレクトリ内のすべてのファイルとフォルダを削除
        shutil.rmtree(directory)
        # 空のディレクトリを再作成
        os.makedirs(directory)
        print(f"{directory} ディレクトリ内のすべてのファイルが削除されました。")
    else:
        print(f"{directory} ディレクトリが見つかりません。")

    # tran = transcribe(m4a_path)
    # res = gpt4o_generate(tran, "gpt-4o-2024-05-13")
    # minutes_container3.write(res)
