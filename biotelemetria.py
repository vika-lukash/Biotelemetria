import matplotlib.pyplot as plt
import numpy as np
import math

time = []
start = 0.0
signal = []
time_new = []
new_signal = []
spektr = []
spektr_new = []
fileNamee = r"C:\10.txt"

# read file
with open(fileNamee, "r") as file:
    for line in file:
        line_split = line.split('\t')
        time.append(start)
        start += 0.065
        signal.append(float(line_split[1][:-1]))
# plt.figure(1)
# plt.plot(time, signal)
# plt.xlabel('time(s)')
# plt.ylabel('U(mV)')


# make spectr of our signal
spektr = abs(np.fft.fft(signal))
# plt.figure(2)
# plt.plot(spektr)
# plt.xlabel('F(Hz)')

# interpolation
time_new = np.arange(0, 10.01, 0.0001)
new_signal = np.interp(time_new, time, signal)
# plt.figure(3)
# plt.plot(time, signal)
# plt.plot(time_new, new_signal)
# plt.title('Signal+Interp')
# plt.xlabel('time(s)')
# plt.ylabel('U(mV)')

#spectк of interpolated signal
spektr_new = abs(np.fft.fft(new_signal))
# plt.figure(4)
# plt.plot(spektr_new)
# plt.plot(spektr)
# plt.xlabel('F(Hz)')
# plt.title('Spectr:orange-old, blue-new')\
fd = 1/0.065
freqs = []
for i in np.arange(0, fd, fd/(len(new_signal))):
    freqs.append(i)
plt.figure(4)
plt.plot(freqs)
plt.figure(5)
plt.plot(freqs, spektr_new)

# Amplitude modulation
fd = 1/0.065
f0 = fd*100
w0 = 2*math.pi*f0
m = 1.5
signal_am = []
norm_new_signal = []
ma = max(abs(new_signal))
for i in range(0, len(time_new)):
    # print(i)

    signal_am.append((1+m*new_signal[i]/ma)*math.sin((w0*time_new[i])))
    norm_new_signal.append(new_signal[i]/ma)


print(freqs)
    # print(signal_am[i])
# plt.figure(5)
# plt.plot(time_new, signal_am)
# plt.plot(time_new, norm_new_signal)
# plt.xlabel('time(s)')
# plt.ylabel('U(mV)')
# plt.title('Signal:orange-signal, blue-mod')

# Amplitude modulation spectr
spektr_mod = abs(np.fft.fft(signal_am))
# plt.figure(6)
# plt.plot(spektr_mod)
# plt.plot(spektr_new)
# plt.xlabel('F(Hz)')
# plt.title('Spectr:orange-old, blue-new')




sum_new = 0
#print(spektr_mod.shape)
sum_all = 0.95*np.sum(spektr_mod)
for i in range(np.shape(spektr_mod)[0]-1):
    sum_new += spektr_mod[i]
    if sum_new >= sum_all:
        f_sr = i
        break
f_stop = freqs[i]
#print(sum_new)
# print(f_sr)
print(np.shape(spektr_mod)[0]-1)


rect = []


def rect_generator(start,t,h,signal):
    rect = []
    for i in range(len(signal)):
        if (i >= start) and (i <= start+t):
            rect.append(h)
        else:
            rect.append(0)
    rect = np.asarray(rect)
    return rect



T = 6000
t = 1700
h = 1
start = 0
leng = (len(new_signal))
print(leng/T)
rect = rect_generator((start), t, h, new_signal)
start += T
while start <= leng:
    rect += rect_generator((start), t, h, new_signal)
    start += T
print(rect)
plt.figure(3)
plt.plot(time_new, rect)
plt.show()


# for i in range(0,len(new_signal)):
#     for i1 in range(T+1):
#         if i1 <= t:
#             rect.append(h)
#         else:
#             rect.append(0)
#             # i1 = 0

# print(rect)
# plt.figure(1)
# plt.plot(rect)

plt.show()



# Frequency modulation
#fd = 1/0.065
#f0 = fd*10
#w0 = 2*math.pi*f0
#dw = 40*math.pi
#signal_fm = []
#integral = 0
#for i in range(0, len(time_new)):
#    integral = integral + (new_signal[i]/max(abs(new_signal)))*0.065 # переходить от интеграла к сумме?
#    signal_fm.append(math.sin((w0*time_new[i])+dw*integral))
#    print(signal_fm[i])
#plt.figure(7)
#plt.plot(time_new, signal_fm)
#plt.plot(time_new, norm_new_signal)
#plt.xlabel('time(s)')
#plt.ylabel('U(mV)')
#plt.title('Signal:orange-signal, blue-mod')

# Frequency modulation spectr
#spektr_fm = abs(np.fft.fft(signal_fm))
#plt.figure(8)
#plt.plot(spektr_fm)
#plt.plot(spektr_new)
#plt.xlabel('F(Hz)')
#plt.title('Spectr:orange-old, blue-new')


 # Phase modulation
#fd = 1/0.065
#f0 = fd*10
#w0 = 2*math.pi*f0
#dfi = 50
#signal_fi = []
#for i in range(0, len(time_new)):
#    signal_fi.append(math.sin((w0*time_new[i])+dfi*new_signal[i]/max(abs(new_signal))))
#    print(signal_fi[i])
#plt.figure(9)
#plt.plot(time_new, signal_fi)
#plt.plot(time_new, norm_new_signal)
#plt.xlabel('time(s)')
#plt.ylabel('U(mV)')
#plt.title('Signal:orange-signal, blue-mod')

# Phase modulation spectr
#spektr_fi = abs(np.fft.fft(signal_fi))
#plt.figure(10)
#plt.plot(spektr_fi)
#plt.plot(spektr_new)
#plt.xlabel('F(Hz)')
#plt.title('Spectr:orange-old, blue-new')


