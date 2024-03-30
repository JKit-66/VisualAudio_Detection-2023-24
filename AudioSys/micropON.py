import sounddevice as sd
import scipy.io.wavfile as sp
import wavio as wv
import datetime
import uuid
import time
import numpy as np
import math
import tensorflow as tf
#from matplotlib import pyplot as plt
#from sklearn.metrics import classification_report

class sensorM():

    def __init__(self):
        # Sampling frequency
        self.freq = 16000
        # Recording duration
        self.duration = 2

        self.audio_model = ('savedAgain8')
        self.imported = tf.saved_model.load(self.audio_model)

        
        self.directory = 'forFinalDetections/'
        self.file2analyze = ''

        self.microP_res = {}

        self.audio_result_book = {'cat':0,
                             'dog':0,
                             'drone':0,
                             'phone':0,
                             'background':0
                             }
        
    def random_name_generator(self):
        now = datetime.datetime.now()
        tod = datetime.date.today()
        HH = now.hour
        mm = now.minute

        MM, DD = tod.month, tod.day
        unique = str(uuid.uuid4())
        unique = unique.split('-')[0]
        fileN = f'{MM}{DD}_{HH}{mm}_{unique}'
        return fileN

    def record_ON(self, name):
        #print("Starting...")
        recording = sd.rec(int(self.duration * self.freq), 
                           samplerate=self.freq, channels=1)

        sd.wait()
        filename = self.directory + f'{name}.wav'
        wv.write(filename, recording, self.freq, sampwidth=2)

        self.file2analyze = filename
        

    def record_for_process(self):
        fileN = self.random_name_generator()
        self.record_ON(fileN)
        print('Finish REC')

    def obtain_res(self,input_data):
        self.microP_res = {}
        predicted_res = list(input_data['predictions'][0]._numpy())
        #return input_data['predictions']
        predicted_res.append(0)
        max_res = max(predicted_res)

        result_book = {
            0:'cat',
            1:'dog',
            2:'drone',
            3:'phone',
            4:'background'}

        
        self.audio_result_book = {'cat':0,
                                  'dog':0,
                                  'drone':0,
                                  'phone':0,
                                  'background':0
                                  }
        
        
        if max_res >= 0.8:
          for value, key in enumerate(self.audio_result_book.items()):
            self.audio_result_book[key[0]] = predicted_res[value]

        else:
            self.audio_result_book['background'] = 1


        print(self.audio_result_book)
            
          


    def squeeze(self, audio, labels):
        audio = tf.squeeze(audio, axis=-1)
        return audio, labels

    def microP_READ(self):
        audio_INPUT = self.file2analyze

        #print(audio_INPUT)
        Input = tf.io.read_file(audio_INPUT)
        
        x, sample_rate = tf.audio.decode_wav(Input, desired_channels=1, desired_samples=32000,)
        #print(x)
        waveform, labels = self.squeeze(x, 'yes')
        res = self.imported(waveform[tf.newaxis, :])
        self.obtain_res(res)
        #print(self.microP_res)
        #print('::', self.audio_result_book)
        
        
    def process_detection_res(self, inputs):
        if (len(inputs) > 0): 
            max_conf = max(inputs, key=inputs.get)
            return max_conf, inputs[max_conf]

        else:
            return None, None
    
    def record_and_process(self):
        print(self.file2analyze)

if __name__ == "__main__":
    sens = sensorM()
    while True:
        sens.record_for_process()
        time.sleep(0.1)
        sens.microP_READ()
        #microP_class, microP_conf = sens.process_detection_res(sens.microP_res)
        #print(sens.microP_res)
        #print(microP_class, microP_conf)
    
