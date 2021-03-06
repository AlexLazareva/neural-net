# Перцептрон, обучение "И" ("AND")
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# для воспроизводимости результатов
np.random.seed(17)

# генерируем диапазон зеленых точек, где 1
x1x2_green = np.array([[1., 1.], [1., 1.], [1., 1.]])

# генерируем диапазон красных точек, где 0
x1x2_red = np.array([[0, 0], [0, 0], [0, 0]])

# объединяем массив
x1x2 = np.concatenate((x1x2_green, x1x2_red))

# проставляем классы: зелёные +1, красные -1
labels = np.concatenate((np.ones(x1x2_green.shape[0]), -np.ones(x1x2_red.shape[0])))

# перемешиваем
indicies = np.array(range(x1x2.shape[0]))
np.random.shuffle(indicies)
x1x2 = x1x2[indicies]
labels = labels[indicies]

# выставляем случайные начальные веса
w1 = 0.5
w2 = 0.7
b = -1.8


# разделяющая гиперплоскость (граница решений)
def lr_line(x1, x2):
    return w1 * x1 + w2 * x2 + b


# ниже границы -1
# выше границы +1
def decision_unit(value):
    return -1 if value <= 0 else 1


# добавляем начальное разбиение в список
lines = [[w1, w2, b]]

for max_iter in range(100):
    # счётчик неверно классифицированных примеров
    # для ранней остановки
    mismatch_count = 0

    # по всем образцам
    for i, (x1, x2) in enumerate(x1x2):
        # считаем значение линейной комбинации на гиперплоскости
        value = lr_line(x1, x2)

        # класс из тренировочного набора (-1, +1)
        true_label = int(labels[i])

        # предсказанный класс
        pred_label = decision_unit(value)

        # если имеет место ошибка классификации
        if (true_label != pred_label):
            # корректируем веса в сторону верного класса, т.е.
            # идём по нормали — (x1, x2) — в случае класса +1
            # или против нормали — (-x1, -x2) — в случае класса -1
            # т.к. нормаль всегда указывает в сторону +1
            w1 = w1 + x1 * true_label
            w2 = w2 + x2 * true_label

            # смещение корректируется по схожему принципу
            b = b + true_label

            # считаем количество неверно классифицированных примеров
            mismatch_count += 1

    if (mismatch_count > 0):
        lines.append([w1, w2, b])
    else:
        break

# рисуем точки (по последней границе решений)
for i, (x1, x2) in enumerate(x1x2):
    pred_label = decision_unit(lr_line(x1, x2))

    if (pred_label < 0):
        plt.plot(x1, x2, 'ro', color='red')
    else:
        plt.plot(x1, x2, 'ro', color='green')

# выставляем равное пиксельное разрешение по осям
plt.gca().set_aspect('equal', adjustable='box')

# проставляем названия осей
plt.xlabel('x1')
plt.ylabel('x2')

# служебный диапазон для визуализации границы решений
x1_range = np.arange(-3, 3, 0.1)


# функционал, возвращающий границу решений в пригодном для отрисовки виде
# x2 = f(x1) = -(w1 * x1 + b) / w2
def f_lr_line(w1, w2, b):
    def lr_line(x1):
        return -(w1 * x1 + b) / w2

    return lr_line


# отрисовываем историю изменения границы решений
it = 0
for coeff in lines:
    lr_line = f_lr_line(coeff[0], coeff[1], coeff[2])
    plt.plot(x1_range, lr_line(x1_range), label='it: ' + str(it))
    it += 1

# зум
plt.axis([-2, 3, -2, 3])
plt.legend(loc='lower left')
# нарисовать график
plt.show()

# выводим найденные значения для коэффициентов w1, w2 и b
print(f'w1 = {w1}\nw2 = {w2}\nb = {b}')
