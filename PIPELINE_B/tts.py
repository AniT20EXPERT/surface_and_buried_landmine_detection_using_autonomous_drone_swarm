#note for windows os: download the espeakng msi installer and run it then only run the function: speak_espeak_ng

#note for linux os [on rpi]:-
#build mimic1 as per instructions on github: https://github.com/MycroftAI/mimic1
#then only run the function: speak_mimic

import subprocess
import os
import platform
import shutil
def speak_espeak_ng(text, output_file="output.wav", speed=150, pitch=50, voice="en-us"):
    """
    Generate speech using eSpeak NG, save to WAV, and play it.
    Works on Windows and Linux (e.g. Raspberry Pi).
    """
    if os.path.exists(output_file):
        os.remove(output_file)

    # Generate speech
    subprocess.run([
        "espeak-ng",
        "-v", voice,
        "-s", str(speed),
        "-p", str(pitch),
        "-w", output_file,
        text
    ], check=True)

    # Choose playback method based on OS
    system = platform.system().lower()
    if "windows" in system:
        # Use PowerShell to play the WAV
        subprocess.run([
            "powershell", "-c",
            f'(New-Object Media.SoundPlayer "{os.path.abspath(output_file)}").PlaySync();'
        ])
    elif "linux" in system:
        # Raspberry Pi / Linux playback
        subprocess.run(["aplay", output_file], check=True)
    else:
        print(f"Audio playback not supported on {system} yet.")

def speak_mimic(text: str, voice: str = "ap", save_to_file: bool = False):
    mimic_path = shutil.which("mimic")
    if mimic_path is None:
        print("[❌] Mimic 1 not found. Install with: sudo apt install mimic")
        return

    if save_to_file:
        outfile = "speech.wav"
        subprocess.run([mimic_path, "-voice", voice, "-t", text, "-o", outfile], check=True)
        print(f"[✔] Saved speech to {outfile}")
        subprocess.run(["aplay", outfile], check=True)
    else:
        subprocess.run([mimic_path, "-voice", voice, "-t", text], check=True)
        print("[✔] Mimic 1 spoke successfully!")


#----------------#
#----main--------#
#----------------#

# for windows
speak_espeak_ng(
    text="Hello! This is eSpeak NG speaking.",
    output_file="speech.wav",
    speed=160,
    pitch=55,
    voice="en-us"
)
# for rpi [linux os]:-
speak_mimic("Hello! This is mimic one speaking.")


