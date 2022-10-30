import numpy as np
import matplotlib.pyplot as plt

filename = "output/train/20221029-133319-twins_pcpvt_small_spike_v-224/summary.csv"
save_fig_name = filename[:-4] + '.png'
data = np.loadtxt(open(filename, "rb"),delimiter=",",skiprows=1)


plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(data[:,0], data[:,1], label='train_loss')
plt.plot(data[:,0], data[:,2], label='valid_loss')
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(data[:,0], data[:,3], label='eval_acc1')
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(data[:,0], data[:,-1], label='lr')
plt.legend()
plt.savefig(save_fig_name)