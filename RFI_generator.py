import numpy as np
import matplotlib.pyplot as plt

nSam = 4096
SNR = 10.    # in dB
dDriftRate = 0.1     # 0 for sinewave, if not value of the normalized frequency drift of the signal duration (0.1 = 10% drift of nSam samples)
BR = 10  # baud rate in samples (duration of 1 message in samples)

# building pseudo-random message
sym = np.zeros(nSam)
sym[0::BR] = 2.*np.round(np.random.uniform(0.,1.,len(sym[0::BR])))-1.

# you can choose your favorite window here (size = BR)
#win = np.hanning(BR)
#win = np.hamming(BR)
win = np.ones(BR)

sym = np.convolve(sym,win)
sym = sym[0:nSam]

# modulation
freq = np.random.uniform()  # sine wave frequency
SOI = sym*np.exp(1j*2.*np.pi*(freq*np.arange(nSam) + dDriftRate/nSam/2.*np.arange(nSam)**2 + np.random.uniform(0,1)))
noise = (np.random.normal(0.,1.,nSam) + 1j*np.random.normal(0.,1.,nSam)) / np.sqrt(2)
Data = 10**(SNR/20.)*SOI + noise

plt.figure()
plt.subplot(311)
plt.plot(Data.real)
plt.xlim((0.,nSam))
plt.xlabel('frequency')
plt.ylabel('spectrum [dB]')
plt.subplot(312)
plt.plot(np.arange(-1,1,1./nSam*2),np.fft.fftshift(10.*np.log10(abs(np.fft.fft(Data)**2))))
plt.xlim((-1,1))
plt.xlabel('frequency')
plt.ylabel('spectrum [dB]')
plt.subplot(313)
plt.specgram(Data, NFFT=256,noverlap=128)
plt.xlabel('time samples')
plt.ylabel('normalized frequency')
