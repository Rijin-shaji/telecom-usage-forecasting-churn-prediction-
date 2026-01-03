import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD

def plot_svm_decision_boundary(X, y):
    svd = TruncatedSVD(n_components=2)
    X_2d = svd.fit_transform(X)

    model = SVC(kernel="rbf")
    model.fit(X_2d, y)

    x_min, x_max = X_2d[:, 0].min()-1, X_2d[:, 0].max()+1
    y_min, y_max = X_2d[:, 1].min()-1, X_2d[:, 1].max()+1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X_2d[:,0], X_2d[:,1], c=y, edgecolor='k')
    plt.title("SVM Decision Boundary")
    plt.show()
