import matplotlib.pyplot as plt
import numpy
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
time_new = []
new_signal = []
spektr = []
spektr_new = []
fileNamee = r"C:\10.txt"
noi = []

# read file
with open(fileNamee, "r") as file:
    for line in file:
        line_split = line.split('\t')
        time.append(start)
        start += 0.065
        signal.append(float(line_split[1][:-1]))
        noi.append(0.0)

sig_sum = signal
noi[20] = 2
sig_sum[20] = sig_sum[20]+noi[20]

spektr = abs(numpy.fft.fft(noi))
spektr_sig = abs(numpy.fft.fft(signal))
spekt_sig_sum = abs(numpy.fft.fft(sig_sum))


fd = 1/0.065
f0 = fd*100
w0 = 2*math.pi*f0
m = 1
sig_sum_am = []
norm_new_signal = []
print(sig_sum)
ma = max(numpy.abs(sig_sum))
time_new = numpy.arange(0, 10.01, 0.0001)
sig_sin = []
sig_cos = []
for i in range(0, len(sig_sum) - 1):
    # print(i)
    sig_sum_am.append((1+m*sig_sum[i]/ma)*math.sin((w0*time_new[i])))
    sig_sin.append(sig_sum_am[i]*math.sin(w0*time_new[i]))
    sig_cos.append(sig_sum_am[i] * math.cos(w0 * time_new[i]))


y_sin = butter_bandpass_filter(sig_sin, 0.01, 3, fd, order=5)
y_cos = butter_bandpass_filter(sig_cos, 0.01, 3, fd, order=5)

sig_arctan = []
for i in range((len(norm_new_signal) - 1)):
    sig_arctan.append(math.atan2(y_sin[i], y_cos[i]))



spektr_am = abs(numpy.fft.fft(sig_sum_am))

spektr_mul_sin = abs(numpy.fft.fft(sig_sin))
spektr_sin = abs(numpy.fft.fft(y_sin))




plt.figure(1)
plt.plot(spektr)
plt.plot(spektr_sig)
plt.plot(spekt_sig_sum)

plt.figure(2)
plt.plot(sig_sum_am)
plt.plot(norm_new_signal)


plt.figure(3)
plt.plot(spektr_am)


plt.figure(4)
plt.plot(norm_new_signal)
plt.plot(sig_sin)


plt.figure(5)
plt.plot(spektr_sin)
plt.plot(spektr_mul_sin)

plt.figure(6)
plt.plot(sig_sum)
plt.plot(sig_arctan)
plt.show()



