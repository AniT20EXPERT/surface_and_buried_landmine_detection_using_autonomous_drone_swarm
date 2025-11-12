
import os
import torch
import pvporcupine
import pyaudio
import wave
import numpy as np
from dotenv import load_dotenv
import time
from vosk import Model, KaldiRecognizer
import json

# -------------------------------------------------------------
# SETUP
# -------------------------------------------------------------
load_dotenv()
ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY")

# Load Silero VAD
model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
(get_speech_timestamps, _, _, _, _) = utils

# Load Vosk model ONCE at startup
print("Loading Vosk model...")
vosk_model = Model("models/vosk-model-small-en-us-0.15")
print("Vosk model loaded")

# Create Porcupine wake word detector
porcupine = pvporcupine.create(
    access_key=ACCESS_KEY,
    keyword_paths=["models/Drone-Swarm_en_windows_v3_0_0.ppn"]
)

# Initialize PyAudio
pa = pyaudio.PyAudio()

audio_stream = pa.open(
    rate=porcupine.sample_rate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    input_device_index=2,  # change if needed
    frames_per_buffer=porcupine.frame_length
)


# -------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------
def record_until_silence(response, vad_model, sr=16000, max_duration=10):
    """
    Records from the mic until silence is detected.
    """
    if response:
        silence_limit = 2.3
    else:
        silence_limit = 1.5
    print("Listening for speech...")
    chunk_size = int(0.5 * sr)  # 0.5s chunks = 8000 samples
    buffer = []
    silence_start = None
    start_time = time.time()

    temp_chunk = np.array([], dtype=np.float32)

    while True:
        # read Porcupine-size frames continuously
        data = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
        pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        temp_chunk = np.concatenate((temp_chunk, pcm))

        # when we have ~0.5s of audio, check VAD
        if len(temp_chunk) >= chunk_size:
            audio_tensor = torch.tensor(temp_chunk).unsqueeze(0)
            speech = get_speech_timestamps(audio_tensor, vad_model)

            if len(speech) > 0:
                buffer.append(temp_chunk)
                silence_start = None
                print("Speaking...", end="\r")
            else:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > silence_limit:
                    print("\n Silence detected. Stopping recording.")
                    break

            temp_chunk = np.array([], dtype=np.float32)

        if time.time() - start_time > max_duration:
            print("\n⏱Max duration reached. Stopping.")
            break

    if len(buffer) == 0:
        return None

    return np.concatenate(buffer)


def save_wav(audio_data, filename, sr=16000):
    """Save audio data as WAV file"""
    # normalize to int16
    audio_data = (audio_data * 32767).astype(np.int16)
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_data.tobytes())
    # print(f"File saved: {filename}")


def transcribe_vosk(audio_path):
    """Transcribe audio file using pre-loaded Vosk model"""
    if not os.path.exists(audio_path):
        return "ERROR: Audio file not found"

    # print(f"🔎 File size: {os.path.getsize(audio_path)} bytes")

    try:
        wf = wave.open(audio_path, "rb")

        # Verify audio format
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            wf.close()
            raise ValueError("Audio must be 16kHz, 16-bit, mono")

        # Use the globally loaded model
        # rec = KaldiRecognizer(vosk_model, wf.getframerate())
        commands = intent_analyser("",True)

        rec = KaldiRecognizer(vosk_model, wf.getframerate(), json.dumps(commands))

        rec.SetWords(True)

        results = []
        print("Transcribing command...")

        # Process audio in chunks
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                results.append(result)
                print(".", end="", flush=True)

        # Get final result
        final_result = json.loads(rec.FinalResult())
        results.append(final_result)
        print()  # New line after dots

        wf.close()

        # Combine all text fragments
        text = " ".join([r.get("text", "") for r in results if r.get("text")])
        return text.strip()

    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return f"ERROR: {str(e)}"


def intent_analyser(input_text, get_commands=False):
    """
    Analyse natural language commands and map them to ROS-style service calls.
    If get_commands=True, returns all supported command strings.
    """
    commands = [
        # 1) start
        "start one", "start two", "start three", "start four", "start",

        # 2) scan
        "scan one", "scan two", "scan three", "scan four", "scan",

        # 3) pause scan
        "pause scan one", "pause scan two", "pause scan three", "pause scan four", "pause scan",

        # 4) resume scan
        "resume scan one", "resume scan two", "resume scan three", "resume scan four", "resume scan",

        # 5) restart scan
        "restart scan one", "restart scan two", "restart scan three", "restart scan four", "restart scan",

        # 6) mark
        "mark one", "mark two", "mark three", "mark four", "mark",

        # 7) pause mark
        "pause mark one", "pause mark two", "pause mark three", "pause mark four", "pause mark",

        # 8) resume mark
        "resume mark one", "resume mark two", "resume mark three", "resume mark four", "resume mark",

        # 9) generate path
        "generate path",

        # 10) start guidance
        "start guidance",

        # 11) pause guidance
        "pause guidance",

        # 12) user acceptance/rejection
        "yes", "no"
    ]

    # Return the list of commands if requested
    if get_commands:
        return commands

    # Parse command
    input_text = input_text.lower().strip()
    parts = input_text.split()

    word_to_num = {"one": 1, "two": 2, "three": 3, "four": 4}
    num = next((word_to_num[w] for w in parts if w in word_to_num), "ALL_DRONES")
    # Mapping rules
    # 1) start
    if parts[0] == "start" and len(parts) >= 1 and "guidance" not in input_text:
        return ["/initialise", num]

    # 2) scan
    if parts[0] == "scan":
        return ["/generate_scan_waypoints", num]

    # 3) pause scan
    if parts[:2] == ["pause", "scan"]:
        return ["/pause_drone", num]

    # 4) resume scan
    if parts[:2] == ["resume", "scan"]:
        return ["/resume_drone", num]

    # 5) restart scan
    if parts[:2] == ["restart", "scan"]:
        return ["/restart_scan", num]

    # 6) mark
    if parts[0] == "mark":
        return ["/mark_mines", num]

    # 7) pause mark
    if parts[:2] == ["pause", "mark"]:
        return ["/mark_mines_pause", num]

    # 8) resume mark
    if parts[:2] == ["resume", "mark"]:
        return ["/mark_mines_resume", num]

    # 9) generate path
    if parts[:2] == ["generate", "path"]:
        return ["/generate_path", {}]

    # 10) start guidance
    if parts[:2] == ["start", "guidance"]:
        return ["/start_guidance", {}]

    # 11) pause guidance
    if parts[:2] == ["pause", "guidance"]:
        return ["/pause_guidance", {}]

    # else case
    return ["/unknown", {}]




# -------------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------------
print("🚀 Listening for wake word... (Ctrl+C to stop)")
TO_EXECUTE = False
TEXT_4_TEXT_TO_SPEECH=""

try:
    while True:
        TO_EXECUTE = False
        TEXT_4_TEXT_TO_SPEECH=""
        pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
        pcm = memoryview(pcm)
        pcm = [int.from_bytes(pcm[i:i + 2], byteorder="little", signed=True)
               for i in range(0, len(pcm), 2)]

        keyword_index = porcupine.process(pcm)

        if keyword_index >= 0:
            print(f"Wake word detected! (index: {keyword_index})")
            print("*BEEP/BUZZER* Speak your command now...")
            time.sleep(0.3)  # Small pause to let user start speaking
            # Record speech until silence
            recorded_audio = record_until_silence(response=False, vad_model=model)
            if recorded_audio is not None:
                filename = "spoken_command.wav"
                save_wav(recorded_audio, filename)
                # print(f"Saved spoken command to {filename}")

                # Transcribe the audio
                # print("Sending to speech-to-text model...")
                text_transcribed = transcribe_vosk(filename)

                if text_transcribed:
                    print(f"Transcription: '{text_transcribed}'")
                    service, params = intent_analyser(text_transcribed,False)
                    print(f"service: {service}")
                    print(f"Arguments: {params}")
                    if service == "/unknown":
                        TEXT_4_TEXT_TO_SPEECH = "invalid command"
                        print(TEXT_4_TEXT_TO_SPEECH)
                        time.sleep(3)
                        continue
                    else:
                        TEXT_4_TEXT_TO_SPEECH = f"do i execute: {text_transcribed}, with service: {service}"
                        print(TEXT_4_TEXT_TO_SPEECH)
                        response_recorded_audio = record_until_silence(response=True, vad_model=model)
                        if response_recorded_audio is not None:
                            res_filename = "response_to_command.wav"
                            save_wav(response_recorded_audio, res_filename)
                            # print(f"response to command stored as {res_filename}")
                            text_transcribed = transcribe_vosk(res_filename)
                            if text_transcribed == "yes":
                                TEXT_4_TEXT_TO_SPEECH="Executing command demanded..."
                                print(TEXT_4_TEXT_TO_SPEECH)
                                TO_EXECUTE = True
                            elif text_transcribed == "no":
                                TEXT_4_TEXT_TO_SPEECH = "Requested command denied!!"
                                print(TEXT_4_TEXT_TO_SPEECH)
                                TO_EXECUTE = False
                            else:
                                TEXT_4_TEXT_TO_SPEECH = "invalid user response"
                                print(TEXT_4_TEXT_TO_SPEECH)
                                time.sleep(3)
                                continue


                else:
                    print("⚠️ No speech detected in recording")
            else:
                print("⚠️ No audio recorded")

except KeyboardInterrupt:
    print("\nExiting...")

finally:
    audio_stream.stop_stream()
    audio_stream.close()
    pa.terminate()
    porcupine.delete()
    print("Cleanup complete")