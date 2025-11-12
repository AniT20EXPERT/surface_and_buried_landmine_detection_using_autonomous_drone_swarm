import pyaudio

pa = pyaudio.PyAudio()

for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    print(f"Device {i}: {info['name']} (Input channels: {info['maxInputChannels']})")

pa.terminate()