# import joblib
# import numpy as np
# # Load the model or object
# model = joblib.load("svm_filter_rbf.joblib")

# # Now you can use the model, e.g., for prediction
# # Example for a scikit-learn model:
# X = np.zeros((2, 784), dtype=np.float32)
# prediction = model.predict(X)
# print(prediction)


from torchvision import datasets, transforms
import numpy as np
import pathlib

# Download MNIST test set
mnist_test = datasets.MNIST(
    root="data/mnist",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# Save as NumPy arrays
X_list, y_list = [], []
for img, y in mnist_test:
    X_list.append(img.view(-1).numpy().astype("float32"))  # flatten to (784,)
    y_list.append(int(y))
X_test = np.stack(X_list, axis=0)  # (10000, 784)
y_test = np.array(y_list, dtype=np.int64)  # digit labels (0–9)

# Make a folder
pathlib.Path("data/shared").mkdir(parents=True, exist_ok=True)
np.save("data/shared/mnist_test_X.npy", X_test)
np.save("data/shared/mnist_test_y.npy", y_test)
print("Saved data/shared/mnist_test_X.npy and mnist_test_y.npy")

# The above code will download the test MNIST data.
# Then run the below code which will load both the data and the classifiers and test
# against the data (Note: It may take some time, may be around 1 minute depending on your laptop).
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

adv_csv = Path("data/adversarial/mnist_fgsm/mnist_test.csv")
df = pd.read_csv(adv_csv)
X_adv = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
maxv = float(X_adv.max())
X_adv = X_adv / (25.0 if maxv <= 25.0 else 255.0)
X_adv = np.clip(X_adv, 0.0, 1.0).astype(np.float32)
print("Adversarial set:", X_adv.shape, "[min,max] =", float(X_adv.min()), float(X_adv.max()))

# Load filters
svm_filter = joblib.load("models/svm_filter_rbf.joblib")
rf_filter = joblib.load("models/rf_filter.joblib")
mlp_filter = joblib.load("models/mlp_filter.joblib")

def summarize(name, preds):
    vals, cnts = np.unique(preds, return_counts=True)
    pct = (cnts / cnts.sum() * 100).round(2)
    print(f"[{name}] adversarial → counts:", {int(v): int(c) for v, c in zip(vals, cnts)},
          "| %:", {int(v): float(p) for v, p in zip(vals, pct)})

# Predict (expect mostly/all 0 = adversarial)
summarize("SVM", svm_filter.predict(X_adv))
summarize("RF", rf_filter.predict(X_adv))
summarize("MLP", mlp_filter.predict(X_adv))

# Since the downloaded data is clean data it will give you 100% at the end.
