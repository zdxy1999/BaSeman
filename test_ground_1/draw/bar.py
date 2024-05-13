import numpy as np
from matplotlib import pyplot as plt
'''
柱形对比图
共有11组数据
'''
plt.figure(figsize=(5, 5), dpi=300)
size = 2
## 两组数据
l_count_last = [27.79,11.80]
l_count_new = [28.09, 20.79]
x = np.arange(size)/4
# 有a/b两种类型的数据，n设置为2
total_width, n = 0.05, 2
# 每种类型的柱状图宽度
width = total_width / n
plt.bar(x, l_count_last, width=width, label="without CCC", color='#AA2222')#
plt.bar(x + width, l_count_new, width=width, label="with CCC", color='#2222AA')
## x轴刻度名称
xticks = ['FS', 'WJ']
xline = [i/4+total_width/n for i in np.arange(2)]
plt.xticks(xline, xticks)


plt.legend(loc='upper right')
# plt.xlabel("Threshold",fontname="Times New Roman")
plt.xlabel("BBox iou")
plt.ylabel("Number of samples")
plt.title('model comparision')
# 显示柱状图
plt.show()
