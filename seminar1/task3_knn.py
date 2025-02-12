import pandas as pd
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# https://lms.yandex.ru/courses/1344/groups/46509/lessons/8929/tasks/60651/solutions/21350883
# dataset : https://github.com/mwaskom/seaborn-data/blob/master/penguins.csv
df = pd.read_csv("penguins.csv")
df = df.dropna()
# print(df.head())
features = df[["bill_length_mm", "bill_depth_mm"]]
labels = df["species"]

train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.2, random_state=123
)
# print(features.head())

df = df.dropna()
features = df[["bill_length_mm", "bill_depth_mm"]]
labels = df["species"]

best_ac = -1
worst_ac = 2
for wt in ("uniform", "distance"):
    for k in range(1, 11):
        knn = KNeighborsClassifier(n_neighbors=k, weights=wt)
        knn.fit(train_features, train_labels)

        y_predicted = knn.predict(test_features)
        # print(wt, k)
        # print(
        #     "Accuracy score is {:.6f}".format(
        #         metrics.accuracy_score(test_labels, y_predicted)
        #     )
        # )
        acc = float(f"{metrics.accuracy_score(test_labels, y_predicted):.6f}")
        if acc > best_ac:
            best_ac = acc
        if acc < worst_ac:
            worst_ac = acc
        # print(f"{metrics.accuracy_score(test_labels, y_predicted):.2f}")


print(f"Best accuracy: {best_ac:.6f}")
print(f"Worst accuracy: {worst_ac:.6f}")
# print(features.head())
