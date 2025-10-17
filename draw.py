import json
import matplotlib.pyplot as plt
import numpy as np

# 加载任务trace数据
with open('trace/2017-9-25-Simulation-100jobs.json', 'r') as f:    # 替换为你的json文件路径
    data = json.load(f)

# 整理任务数据 (按起始时间排序)
jobs = []
for item in data:
    jobs.append((item['jobid'], item['start_time'], item['end_time'] - item['start_time']))

jobs.sort(key=lambda x: x[1])

fig, ax = plt.subplots(figsize=(14, 8))
y_ticks = []
y_labels = []

for i, (jobid, start, duration) in enumerate(jobs):
    # 将时间戳转换为小时（假设原始为秒）
    start_hours = start / 3600
    duration_hours = duration / 3600
    ax.broken_barh([(start_hours, duration_hours)], (i - 0.4, 0.8), facecolors='tab:blue')
    y_ticks.append(i)
    y_labels.append(str(jobid))

# 横坐标每2小时一个大刻度
max_time_hours = min(max([item['end_time'] for item in data]) / 3600, 24)
print(max_time_hours)
# 下取整到最近的2的倍数，上取整到超过max_time_hours最近的2的倍数
x_ticks = np.arange(0, int(max_time_hours) + 2, 2)
ax.set_xticks(x_ticks)
ax.set_xlim(0, x_ticks[-1])

# 纵坐标每200一个刻度
y_max = len(jobs)
y_major_ticks = np.arange(0, y_max, 200)
ax.set_yticks(y_major_ticks)

# 设置轴标签
ax.set_xlabel('Time (Hours)')
ax.set_ylabel('Job Serial Number')
ax.set_title('Task Trace Waterfall Chart')

# 添加网格
ax.grid(True, which='major', axis='x', alpha=0.5)
ax.grid(True, which='major', axis='y', alpha=0.4)

# 保存图片
plt.savefig('2017-9-25-Simulation.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()