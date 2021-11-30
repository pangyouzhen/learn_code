import numpy as np
import matplotlib.pyplot as plt

warmup_steps = 2500

init_lr = 0.1

max_steps = 15000
x = []
y = []
for train_steps in range(max_steps):
    if warmup_steps and train_steps < warmup_steps:
        warmup_percent_done = train_steps / warmup_steps
        warmup_learning_rate = init_lr * warmup_percent_done
        learning_rate = warmup_learning_rate
    else:
        learning_rate = learning_rate ** 1.0001  # 预热学习率结束后,学习率呈指数衰减(近似模拟指数衰减)

    if (train_steps + 1) % 100 == 0:
        x.append(train_steps)
        y.append(learning_rate)
        print("train_steps:%.3f--warmup_steps:%.3f--learning_rate:%.3f" % (
            train_steps + 1, warmup_steps, learning_rate))

plt.plot(x, y)
plt.show()
