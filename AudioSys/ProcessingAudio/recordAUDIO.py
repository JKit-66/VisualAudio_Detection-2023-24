
import sounddevice as sd
import scipy.io.wavfile as sp
import wavio as wv

choice = 2

if choice == 1:
    # Sampling frequency
    freq = 32000
     
    # Recording duration
    duration = 1
     
    # Start recorder with the given values 
    # of duration and sample frequency
    print("Starting...")
    recording = sd.rec(int(duration * freq), 
                       samplerate=freq, channels=1)


    # Record audio for the given number of seconds
    sd.wait()

    print("Finish...") 
    # This will convert the NumPy array to an audio
    # file with the given sampling frequency
    #write("recording0.wav", freq, recording)
     
    # Convert the NumPy array to audio file
    filename = "recording1.wav"

    wv.write( filename, recording, freq, sampwidth=2)

    sample_rate, audio_array = sp.read(filename)

    print("Sample rate:", sample_rate)
    print("Number of samples:", len(audio_array))
    print("Audio array:", audio_array)

elif choice == 2:
    # Sampling frequency
    freq = 32000
     
    # Recording duration
    duration = 1
     
    # Start recorder with the given values 
    # of duration and sample frequency

    idx = 0

    while True:
        print("Starting...")
        recording = sd.rec(int(duration * freq), 
                           samplerate=freq, channels=1)

        # Record audio for the given number of seconds
        sd.wait()
        print("Finish...") 
        # This will convert the NumPy array to an audio
        # file with the given sampling frequency
        #write("recording0.wav", freq, recording)
        idx += 1
        # Convert the NumPy array to audio file
        filename = f'virtualENV/virtualENV/onesec/stop/recording{idx}.wav'

        wv.write( filename, recording, freq, sampwidth=2)

        #sample_rate, audio_array = sp.read(filename)

        #print("Sample rate:", sample_rate)
        #print("Number of samples:", len(audio_array))
        #print("Audio array:", audio_array)


if __name__ == "__main__":
    choice = 2
