import numpy as np
# https://lms.yandex.ru/courses/1344/groups/46509/lessons/8929/tasks/60647/solutions/21436051
x = np.array(input().split()).astype(float)

y = np.array(input().split()).astype(float)


def r2_calc(x, y):
    ssr = 0
    sst = 0
    true_mean = np.mean(x)
    for n in range(len(x)):
        ssr += (x[n] - y[n]) ** 2
        sst += (true_mean - x[n]) ** 2

    r2 = 1 - (ssr / sst)

    print(f"R2: {r2:.2f}")


r2_calc(x, y)
