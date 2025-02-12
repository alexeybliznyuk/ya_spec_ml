import numpy as np

x = np.array(input().split()).astype(float)
y = np.array(input().split()).astype(float)


def errors_calc(x, y):
    ss = 0
    absum = 0
    for n in range(len(x)):
        ss += (x[n] - y[n]) ** 2
        absum += abs(x[n] - y[n])

    print(f"MSE: {(ss / len(x)):.2f}")
    print(f"MAE: {absum / len(x):.2f}")
    print(f"RMSE: {((ss / len(x)) ** 0.5):.2f}")


errors_calc(x, y)
