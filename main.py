# Перцептрон, обучение "Не" ("NOT")
import matplotlib.pyplot as plt
import numpy as np

x1x2_green = np.array([[0.], [0.], [0.]])
x1x2_red = np.array([[1.], [1.], [1.]])

x1x2 = np.concatenate((x1x2_green, x1x2_red))
labels = np.concatenate((np.ones(x1x2_green.shape[0]), -np.ones(x1x2_red.shape[0])))

indices = np.array(range(x1x2.shape[0]))
np.random.shuffle(indices)
x1x2 = x1x2[indices]
labels = labels[indices]

# случайные начальные веса
w1 = 0.5
b = 0.5


# разделяющая гиперплоскость (граница решений)
def lr_line(x1):
    return w1 * x1 + b


def decision_unit(value):
    return -1 if value <= 0 else 1


lines = [[w1, b]]

for max_iter in range(100):
    mismatch_count = 0
    for i, (x1) in enumerate(x1x2):
        value = lr_line(x1)
        true_label = int(labels[i])
        pred_label = decision_unit(value)
        if (true_label != pred_label):
            w1 = w1 + x1 * true_label
            b = b + true_label
            mismatch_count += 1

            if (mismatch_count > 0):
                lines.append([w1, b])
            else:
                break

print(f'w1 = {w1[0]}\nb = {b}')
plt.plot(lines)
plt.show()
