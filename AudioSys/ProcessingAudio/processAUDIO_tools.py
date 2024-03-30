import os
import tensorflow as tf
import numpy as np
import seaborn as sns
import pathlib
from IPython import display
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
import random
import wave
from scipy.io import wavfile 
from scipy.fftpack import fft
from collections import Counter
#import seaborn as sns
#from memory_profiler import profile
import time

imported = tf.saved_model.load("savedAgain8")

def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

# Convert waveform to spectrogram
def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    return spectrogram[..., tf.newaxis]

# Plot the spectrogram
def plot_spectrogram(spectrogram, label):
    spectrogram = np.squeeze(spectrogram, axis=-1)
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    plt.figure(figsize=(10, 3))
    plt.title(label)
    plt.imshow(log_spec, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time')
    plt.ylabel('Frequency')

# Creating spectrogram dataset from waveform or audio data
def get_spectrogram_dataset(dataset):
    dataset = dataset.map(
        lambda x, y: (get_spectrogram(x), y),
        num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

# Defining the model
def get_model(input_shape, num_labels):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        # Resizing the input to a square image of size 64 x 64 and normalizing it
        tf.keras.layers.Resizing(64, 64),
        tf.keras.layers.Normalization(),

        # Convolution layers followed by MaxPooling layer
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),

        # Dense layer
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        # Softmax layer to get the label prediction
        tf.keras.layers.Dense(num_labels, activation='softmax')
    ])
    # Printing model summary
    model.summary()
    return model

def obtain_res(input_data):
    #print(res['class_names'])

    predicted_res = list(input_data['predictions'][0]._numpy())
    #print(predicted_res)
    max_res = max(predicted_res)

    """result_book = {
        0:"down",
        1:"drone",
        2:"go",
        3:"left",
        4:"no",
        5:"right",
        6:"stop",
        7:"up",
        8:"yes"}"""

    result_book = {
        0:"cat",
        1:"dog",
        2:"drone",
        3:"stop"}
    
    if max_res >= 0.7:
      index = predicted_res.index(max_res)
      conf = max_res
      return result_book[index], conf

    else:
      res = 'ERROR'
      conf = 0
      return res, conf

def get_duration_wave(file_path):
   with wave.open(file_path, 'r') as audio_file:
      frame_rate = audio_file.getframerate()
      n_frames = audio_file.getnframes()
      duration = n_frames / float(frame_rate)
      return duration

    


fileN = 'drone-audio(1).wav'
#fig = make_subplots(rows=3, cols=1)
class audiof:
    def __init__(self, filename):
        
        self.min_val = 10000
        self.filename = filename
        self.rate, self.data = wavfile.read(filename)
        self.focus_size = int(0.15 * self.rate)
        self.length = self.data.shape[0] / self.rate
        self.data_size = len(self.data)


    def plot_fft(self):
        data2 = self.data
        print(self.rate)
        signal = data2
        
        fft_spectrum = np.fft.rfft(signal)
        freq = np.fft.rfftfreq(signal.size, d=1./self.rate)
        fft_spectrum_abs = np.abs(fft_spectrum)
        fig, ax = plt.subplots()
        ax.plot(freq, fft_spectrum_abs)
        plt.xlabel("Frequency, Hz")
        plt.ylabel("Amplitude (not in dB)")
        ax.grid(which="both")
        ax.minorticks_on()
        ax.tick_params(which = "minor", bottom = False, left = False)
        plt.show()

    def plot_fft_logscale(self,i, cat):
        data2 = self.data
        #print(data2)
        signal = data2
        
        fft_spectrum = np.fft.rfft(signal)
        freq = np.fft.rfftfreq(signal.size, d=1./self.rate)
        fft_spectrum_abs = np.abs(fft_spectrum)
        fig, ax = plt.subplots()
        ax.plot(freq, fft_spectrum_abs, label=f'chunk_{cat}({i}).wav')
        fig.legend()
        plt.xlabel("Frequency, Hz (log scale)")
        plt.ylabel("Amplitude (log scale)")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim([0.3, 2e4])
        plt.ylim([0.2, 3e7])

        #plt.xlim([0.3, 2e4])
        #plt.ylim([1, 2e7])
        
        plt.grid(True)
        ax.grid(which="both")
        ax.minorticks_on()
        ax.tick_params(which = "minor", bottom = False, left = False)
        #plt.savefig('upsample_mini_speech/results/fft_norm_log/{}/{}_fft_logscale.jpg'.format(cat, i))
        plt.savefig('virtualENV/{}-{}-fft_norm.jpg'.format(cat, i))
        #plt.savefig('upsample_mini_speech/testthediffbetweenupscaleandnoupscael/fft_logscale2.jpg'.format(cat, i))
        #plt.show()

    
    def plot_fft_norm(self, i, cat):
        data2 = self.data / (2.0**15)
        
        signal = data2
        
        fft_spectrum = np.fft.rfft(signal)
        freq = np.fft.rfftfreq(signal.size, d=1./self.rate)
        fft_spectrum_abs = np.abs(fft_spectrum)
        #plt.plot(freq, 10*np.log10(fft_spectrum_abs))
        fig, ax = plt.subplots()
        ax.plot(freq, 20*np.log10(fft_spectrum_abs), label=f'chunk_{cat}{i}.wav')
        fig.legend()
        plt.xlabel("Frequency, Hz")
        plt.ylabel("Amplitude (dB)")
        plt.ylim([-70, 65])
        #plt.grid(True)
        ax.grid(which="both")
        ax.minorticks_on()
        ax.tick_params(which = "minor", bottom = False, left = False)
        #plt.close()

        #print('Goin to save...')
        plt.savefig('upsample_mini_speech/results/fft_norm/{}/{}-fft_norm.jpg'.format(cat, i))
        #plt.savefig('virtualENV/{}-fft_norm.jpg'.format(cat))
        #plt.savefig('upsample_mini_speech/testthediffbetweenupscaleandnoupscael/fft_norm2.jpg'.format(cat, i))
        #print('Saved...')
        #plt.show()
        

    def plot_read(self,i):
        
        time = np.linspace(0., self.length, self.data.shape[0])
        fig, ax = plt.subplots()
        #time, self.data, label="Audio"
        ax.plot(time, self.data, label="Audio")
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        ax.grid(which="both")
        ax.minorticks_on()
        ax.tick_params(which = "minor", bottom = False, left = False)
        plt.savefig('results/read/{}-read.jpg'.format(i))
        #plt.show()        

    def calc_distances(self):

        focuses = []
        distances = []
        idx = 0
        
        while idx < len(self.data):
            if ((self.data[idx][0]) > self.min_val) and ((self.data[idx][1]) > self.min_val):
                mean_idx = idx + self.focus_size // 2
                focuses.append(float(mean_idx) / self.data_size)
                if len(focuses) > 1:
                    last_focus = focuses[-2]
                    actual_focus = focuses[-1]
                    distances.append(actual_focus - last_focus)
                idx += self.focus_size
            else:
                idx += 1
        return distances 

#@profile

def options_AUDIO(choice):
  if choice == 0:
    path = os.path.join('Recording(23).wav')
    Input = tf.io.read_file(str(path))
    x, sample_rate = tf.audio.decode_wav(Input, desired_channels=1, desired_samples=32000,)
    audio, labels = squeeze(x, 'yes')
    waveform = audio
    res = imported(waveform[tf.newaxis, :])
    print(obtain_res(res))
      
  if choice == 1:
    # upsample_mini_speech/try2validate
    a = b = 0
    dog = drone = cat = stop = 0
    dog_cat = drone_cat = cat_cat = stop_cat = 0
    start_time = time.time()
    cat_mist = []
    dog_mist = []
    drone_mist = []
    stop_mist = []

    short_res = []
    for i in range(1,8000):
      #category_path = random.choice(os.listdir('upsample_mini_speech/try2validate/'))
      #category_path = random.choice(["cat", "dog", "drone", "go", "stop", "yes"])
      category_path = random.choice(["cat", "dog", "drone", "stop"])
      audio_path  = random.choice(os.listdir('upsample_mini_speech/try2validate/{}'.format(category_path)))
      path = os.path.join('upsample_mini_speech/try2validate/', category_path, audio_path)
      Input = tf.io.read_file(str(path))
      x, sample_rate = tf.audio.decode_wav(Input, desired_channels=1, desired_samples=32000,)
      audio, labels = squeeze(x, 'yes')
      waveform = audio
      res = imported(waveform[tf.newaxis, :])
      #print(res['predictions'][0]._numpy())

      short_res.append(f'{category_path}:{obtain_res(res)}')
      
      if category_path == "cat":
        cat_cat += 1

      elif category_path == "dog":
        dog_cat += 1

      elif category_path == "drone":
        drone_cat += 1

      elif category_path == "stop":
        stop_cat += 1
        

      
      if category_path == obtain_res(res):
        #print("True")
        a+= 1
        
      else:
        #print("False", category_path, obtain_res(res))
        b+=1
        a,_ = obtain_res(res)

        if category_path == 'cat':
          cat +=1
          cat_mist.append(a)
          
        elif category_path == 'dog':
          dog +=1
          dog_mist.append(a)

        elif category_path == 'drone':
          drone +=1
          drone_mist.append(a)

        elif category_path == 'stop':
          stop +=1
          stop_mist.append(a)

    end_time = time.time()
    print ("Test: ", ["Cat:", cat_cat, "Dog:", dog_cat, "Drone:", drone_cat, "Stop", stop_cat])
    print (["True:",a,"False:",b])
    elapsed_time = end_time - start_time
    print(f' Simulaiton done within: {elapsed_time} time (s)')
    print ("Mistake: ", ["Cat:", cat, "Dog:", dog, "Drone:", drone, "Stop", stop])
    print (["Cat treated as:", Counter(cat_mist), "Dog treated as:", Counter(dog_mist), "Drone treated as:", Counter(drone_mist), "Stop treated as:", Counter(stop_mist)]) 
    print (Counter(short_res))
      
  elif choice == 2:
    # upsample_mini_speech/try2validate
    dura = []
    tasks = 2
    finale_ = []
    choices = ["cat", "dog", "drone", "stop"]
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
        directory2 = "upsample_mini_speech/try2validate/{}".format(cat)
        cat2 = next(os.walk(directory2))[2][j]
        audio_path  = os.listdir(str(directory+ '/' + cat))
        path = os.path.join('upsample_mini_speech/try2validate', cat, audio_path[j])
        #print(path)

        if tasks == 1:
          duration = get_duration_wave(path)
          dura.append(duration)
          
        elif tasks == 2:
          Input = tf.io.read_file(str(path))
          x, sample_rate = tf.audio.decode_wav(Input, desired_channels=1, desired_samples=32000,)
          audio, labels = squeeze(x, 'yes')
          waveform = audio
          res = imported(waveform[tf.newaxis, :])
          obtain, _ = obtain_res(res)
          #data_in = f'{obtain}:{obtain}'
          #finale_.append(obtain)
          
    #dura = np.array(dura)
    #print(dura)
    #print(init_num)  #8310
    #perform plot with np.histogram
    finale_ = np.array(finale_)  
    plt.hist(finale_)
    hist,bins = np.histogram(finale_)
    print(Counter(finale_))
    plt.title("Histogram")
    plt.show()

    
  elif choice == 3:
    dura = []
    choices = ["cat", "dog", "drone", "stop"]
    init_num = 0
    #get audio file time length
    directory = "upsample_mini_speech"
    files = os.listdir(directory)
    num_cat = len(choices)
    
    
    for i in range(num_cat):
      cat = choices[i]
      print(cat)
      upper_lim = 0
      ans = os.listdir(str(directory + '/' + cat))
      files2 = len(ans)
      init_num += files2
      
      for j in range(files2):
        directory2 = "upsample_mini_speech/{}".format(cat)
        cat2 = next(os.walk(directory2))[2][j]
        audio_path  = os.listdir(str(directory+ '/' + cat))
        path = os.path.join('upsample_mini_speech', cat, audio_path[j])
        upper_lim +=1

        if upper_lim <= 300:
          fileN = str(path)
          audioNow = audiof(fileN)
          #dis4 = audioNow.plot_fft_norm(j, cat)
          dis4 = audioNow.plot_fft_logscale(j,cat)

        else:
          break
        
        #plot_fft_norm, plot_fft_logscale

  elif choice == 4:
    # Plot confusion matrix
    cm = np.array([[1775, 169, 0, 36],
                  [28, 2006, 0, 34],
                  [448, 16, 1484, 17],  
                  [53, 92, 1, 1849]])
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=True)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    plt.show()


  elif choice == 5:
    path = 'mini_speech_commands/'
    Input = tf.io.read_file(str(path))
    x, sample_rate = tf.audio.decode_wav(Input, desired_channels=1, desired_samples=16000,)
    audio, labels = squeeze(x, 'yes')
    waveform = audio
    res = imported(waveform[tf.newaxis, :])
    

  elif choice == 6:
    #path = os.path.join('mini_speech_commands/', 'dog', 'dog_barking_97.wav' )
    path = os.path.join('mini_speech_commands/', 'cat', 'cat_(4).wav' )
    Input = tf.io.read_file(str(path))
    x, sample_rate = tf.audio.decode_wav(Input, desired_channels=1, desired_samples=16000,)
    audio, labels = squeeze(x, 'yes')
    waveform = audio
    res = imported(waveform[tf.newaxis, :])
    obj, conf = obtain_res(res)
    print(obj, conf)

  elif choice == 7:
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
        output_str = f"ffmpeg -i {fileN} -ac 1 -ar 44100 {fileNOut}"

        os.system(output_str)

  elif choice == 8:
    #fileN = 'upsample_mini_speech/dog/chunk_dog12.wav'
    fileN ='onehouraudio/catmeow_1-45.wav'
    #fileNOut = 'upsample_mini_speech/chunk_dog12DOWN.wav'
    fileNOut ='onehouraudio/catmeow_1-45DOWN.wav'
    output_str = f"ffmpeg -i {fileN} -ac 1 -ar 32000 {fileNOut}"

    os.system(output_str)
    
  else:
    pass


if __name__ == "__main__":
  audioN = audiof('virtualENV/chunk_drone13.wav')
  audioN.plot_fft_logscale(13,'drone')
  #audioN.plot_fft_norm(1,'cat')

  #choices = 0
  #options_AUDIO(choices)
  
    
