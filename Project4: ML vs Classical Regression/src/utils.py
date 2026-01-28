import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return mse, preds

def save_predicted_vs_actual(y_true, y_pred, title, path):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        linestyle="--"
    )
    plt.xlabel("Actual Execution Time (ms)")
    plt.ylabel("Predicted Execution Time (ms)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()