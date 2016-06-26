import pickle
import matplotlib.pyplot as plt
from sklearn.mixture import GMM


path = '../files/features.pkl'
with open(path, "rb") as f:
    x = pickle.load(f)

min_bic = 100
min_id = -1
classifiers = []
classes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25]
for i, c in enumerate(classes):
    classifier = GMM(n_components=c)
    classifiers.append(classifier)
    classifier.fit(x)
    bic = classifier.bic(x)
    if bic < min_bic:
        min_bic = bic
        min_id = i

print(min_bic)
print(classes[min_id])
plt.figure(1)
plt.imshow(classifiers[min_id].means_)
print classifiers[min_id].weights_
plt.grid(True)
plt.show()
#
# for i in range(classes[min_id]):
#     mean = classifiers[min_id].means
#     weight = classifiers[min_id].weights_[i]
#     axarr[i].plot(mean)
#     print weight
#
# plt.show()
# y_hat = classifier.predict(x)
