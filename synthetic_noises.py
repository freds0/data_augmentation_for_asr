# source: https://github.com/makermovement/3.5-Sensor2Phone/blob/master/generate_any_audio.py
import struct
import numpy as np
from scipy import signal as sg
from scipy.io.wavfile import write

sampling_rate = 44100                    ## Sampling Rate
freq = 10                               ## Frequency (in Hz)
freq = 4400                               ## Frequency (in Hz)
samples = 44100*10                         ## Number of samples 
x = np.arange(samples)

####### sine wave ###########
y1 = 100*np.sin(2 * np.pi * freq * x / sampling_rate)

####### square wave ##########
y2 = 100* sg.square(2 *np.pi * freq * x / sampling_rate )

####### square wave with Duty Cycle ##########
y3 = 100* sg.square(2 *np.pi * freq * x / sampling_rate , duty = 0.8)

####### Sawtooth wave ########
y4 = 100* sg.sawtooth(2 *np.pi * freq * x / sampling_rate )


write("test1.wav", sampling_rate, y1.astype(np.int16))

write("test2.wav", sampling_rate, y2.astype(np.int16))

write("test3.wav", sampling_rate, y3.astype(np.int16))

write("test4.wav", sampling_rate, y4.astype(np.int16))

'''
f = open('test.wav','wb')
## Instructions to play test.wav on computer
## 1. Open as Signed 8-bit on Audacity - Watch Video at https://bit.ly/2YwmN9q for instructions
## 2. Or using SoX: play -t raw -r 44.1k -e signed -b 8 -c 1 test.wav

for i in y:
	f.write(struct.pack('b',int(i)))
f.close()
'''