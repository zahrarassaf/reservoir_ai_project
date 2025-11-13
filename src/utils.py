import matplotlib.pyplot as plt

def plot_predictions(y_true, y_pred, title="Predictions"):
    plt.figure(figsize=(10,5))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
