import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.signal import butter, lfilter, lfilter_zi, welch

def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    zi = lfilter_zi(b, a)
    y, z = lfilter(b, a, data, zi=zi*data[0])
    return y

time = []
start = 0.0
signal = []
fileNamee = r"C:\1.txt"
noi = []
count = 0
# read file
with open(fileNamee, "r") as file:
    for line in file:
        line_split = line.split('\t')
        time.append(start)
        start += 0.01
        signal.append(float(line_split[1][:-1]))
        count += 1


sig_sum = []
sig_sum = signal

#sig_sum[20] = sig_sum[20]+noi[20]

#spektr_noise = np.abs(np.fft.fft(noi))
spektr = np.abs(np.fft.fft(signal))
spektr_sum = np.abs(np.fft.fft(sig_sum))

time_new = np.arange(0, time[-1], 0.01/1000)
new_signal = np.interp(time_new, time, sig_sum)
print(time_new[-1])
print(time[-1])
fd = 1/0.01
fd_new = 1/(0.01/1000)
#fd_new = fd
f0 = fd_new/5
w0 = f0
m = 0.5
sig_sum_am = []
ma = max(np.abs(new_signal))
sig_sin = []
for i in range(0, len(new_signal)):
    sig_sum_am.append((1+m*new_signal[i]/ma)*math.sin(w0*time_new[i]))
    sig_sin.append(sig_sum_am[i]*math.sin(w0*time_new[i]))
    noi.append(0)
print(fd_new)
noi[1000] = 100
sig_dem = butter_bandpass_filter(sig_sin, 8, 200, fd_new, order=3)
noi_fil = butter_bandpass_filter(new_signal, 10, 200, fd_new, order=3)
freqs = []
for i in np.arange(0, fd_new, fd_new/(len(sig_sin))):
    freqs.append(i)

spektr_new = abs(np.fft.fft(new_signal))
spektr_sin = abs(np.fft.fft(sig_sin))
spektr_dem = abs(np.fft.fft(sig_dem))
spektr_am = abs(np.fft.fft(sig_sum_am))
spektr_noi = abs(np.fft.fft(noi_fil))

plt.plot(signal)
plt.plot(sig_sum)
plt.plot(noi)
plt.suptitle('Сигналы: исходный, шум и суммарный')


plt.figure(2)
plt.plot(spektr)
plt.plot(spektr_sum)
#plt.plot(spektr_noise)
plt.suptitle('Спектры: исходный и суммарный')

plt.figure(3)
plt.plot(time, signal)
plt.plot(time_new, new_signal)
# plt.plot(time_new, sig_sin)
# plt.plot(time_new, sig_sum_am)
# plt.plot(time_new, sig_dem)
# plt.plot(time_new, sig_dem + 0.5)
# plt.figure(10)
# plt.plot(time_new, sig_dem)
plt.suptitle('Сигналы: исходный и интерполированный')

plt.figure(4)
plt.plot(freqs, spektr_new)
#plt.plot(freqs, spektr_sin)
# plt.plot(freqs, spektr_dem)
plt.suptitle('Спектр интерполированного сигнала')

plt.figure(5)
plt.plot(time_new, sig_sin)
plt.suptitle('Сигнал, умноженный на синус')

plt.figure(6)
plt.plot(time_new, sig_sum_am)
plt.suptitle('Промодулированный сигнал')

plt.figure(7)
plt.plot(time_new, sig_dem)
plt.suptitle('Демодулированный сигнал')

plt.figure(8)
plt.plot(freqs, spektr_dem)
plt.suptitle('Спектр демодулированного сигнала')

plt.figure(9)
plt.plot(freqs, spektr_sin)
plt.plot(freqs, spektr_noi)
plt.suptitle('Спектры сигналов, умноженного на синус и шума')
print(spektr_noi)

plt.figure(10)
plt.plot(freqs, spektr_am)
plt.suptitle('Спектр модулированного сигнала')
print(spektr_dem)






plt.show()