import numpy as np
from scipy.io import wavfile
import os
from pydub import AudioSegment


def split_audio(input_file, output_dir, cat, duration=1000):
    # Load the audio file
    audio = AudioSegment.from_wav(input_file)

    # Calculate the number of chunks
    num_chunks = len(audio) // int(duration)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Split the audio into one-second chunks
    for i in range(num_chunks):
        start_time = i * duration
        end_time = (i + 1) * duration
        chunk = audio[start_time:end_time]
        chunk.export(os.path.join(output_dir, f"chunk_{cat}{i}.wav"), format="wav")

def read_Audio_options(choice):
    if choice == 1:
        # Example usage

        #input_file = ["onehouraudio/1-45dogbarkDOWN.wav" ,  "onehouraudio/catmeow_1-45DOWN.wav"] # Path to the input WAV file
        #output_dir = ["upsample_mini_speech/dog-stereo", "upsample_mini_speech/cat-stereo"]      # Output directory for the audio clips
        input_file = ["upsample_mini_speech/try2validate/45-60_cat_meow.wav"] 
        output_dir = ["upsample_mini_speech/try2validate/cat"]
        catg = ['dog', 'cat', 'drone']
        for idx in range(len(input_file)):
            split_audio(input_file[idx], output_dir[idx], catg[0], duration=1000)
            print(f'done {idx}')


    elif choice == 2:
        cat = "upsample_mini_speech/try2validate"
        #output_dir = ["dog-stereo", "cat-stereo"]
        output_dir = ["dog"]
        num = 0
        to_be_removed = []

        for idx in range(len(output_dir)):
            cattt = output_dir[idx]
            iiddxx = os.path.join(cat, cattt)
            files = os.listdir(iiddxx)
            files2 = len(files)
            type_aud = "mono"

            for j in range(len(files)):
                #print('Yes', files[j])
                audioN = str(files[j])
                #print(type(audioN))
                pathN = os.path.join("upsample_mini_speech", "try2validate", cattt, audioN)
                dummy = 0
                sample_rate, audio_data = wavfile.read(str(pathN))
                print(pathN)
                
                if type_aud == "stereo":
                    mono_audio = np.mean(audio_data, axis=1)
                    mono_audio_reshaped = mono_audio.reshape(-1, 1)

                    for idx in mono_audio_reshaped:
                        if idx[0] > 0.01:
                            dummy += 1

                elif type_aud == "mono":
                    mono_audio_reshaped = audio_data

                    for idx in mono_audio_reshaped:
                        if idx > 0.01:
                            dummy += 1
                        
                target = int(mono_audio_reshaped.shape[0]*0.487)  #cat 0.3   #dog 0.47
                

                if dummy <= target:
                    num += 1
                    print ('To be Removed..', num, pathN)
                    to_be_removed.append(pathN)
                    
                else:
                    pass
        x = input('Prompt to Remove [y/n]:')

        if x =="y":
            for i in to_be_removed:
                os.remove(i)
                print('Removed...', i)
                #pass
            #print(to_be_removed)

        else:
            pass
                
    elif choice == 3:
        sample_rate, audio_data = wavfile.read('onehouraudio/emptyhad.wav')

        sample_rate2, audio_data2 = wavfile.read('onehouraudio/notemptyhad.wav')

        mono_audio = np.mean(audio_data, axis=1)
        mono_audio_reshaped = mono_audio.reshape(-1, 1)

        mono_audio2 = np.mean(audio_data2, axis=1)
        mono_audio_reshaped2 = mono_audio2.reshape(-1, 1)

        #print(mono_audio_reshaped)
        #print(mono_audio_reshaped2)

        target = int(mono_audio_reshaped.shape[0]*0.35)
        print(mono_audio_reshaped2.shape, target)
        dummy = 0

        for idx in range(len(mono_audio_reshaped)):
            if mono_audio_reshaped[idx][0] > 0.01:
                dummy += 1

        if dummy >= target:
            print('Congrats')


    elif choice == 4:
        dura = []
        choices = ["stop", "drone"]
        init_num = 0
        #get audio file time length
        directory = "mini_speech_commands"
        files = os.listdir(directory)
        num_cat = len(choices)
        
        for i in range(num_cat):
          #cat = next(os.walk(directory))[1][i]
          cat = choices[i]
          print(cat)
          
          ans = os.listdir(str(directory + '/' + cat))
          files2 = len(ans)
          init_num += files2
          
          for j in range(files2):
            directory2 = "mini_speech_commands/{}".format(cat)
            cat2 = next(os.walk(directory2))[2][j]
            audio_path  = os.listdir(str(directory+ '/' + cat))
            path = os.path.join('mini_speech_commands', cat, audio_path[j])
            pathOut = os.path.join('upsample_mini_speech', cat, audio_path[j])
            
            fileN = str(path)
            fileNOut = str(pathOut)
            
            output_file = fileNOut   # "compressed_output_file.wav"
            output_str = f"ffmpeg -i {fileN} -ac 1 -ar 32000 {fileNOut}"

            os.system(output_str)

    else:
        pass


if __name__ == "__main__":
    choices = 2
    read_Audio_options(choices)
        
    
