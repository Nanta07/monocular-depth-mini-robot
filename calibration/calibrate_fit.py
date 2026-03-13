import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

PAIRS_FILE = Path("calibration/samples/pairs.jsonl")
OUT_FILE = Path("calibration/calib_params.json")

# load data
distances = []
preds = []

with open(PAIRS_FILE) as f:
    for line in f:
        rec = json.loads(line)
        distances.append(rec["distance_m"])
        preds.append(rec["pred_stat"])

distances = np.array(distances)
preds = np.array(preds)

# reshape for sklearn
X = preds.reshape(-1,1)
y = distances

# linear regression
model = LinearRegression()
model.fit(X,y)

y_pred = model.predict(X)

r2 = r2_score(y,y_pred)

print("Linear model")
print("distance = a * pred + b")
print("a =",model.coef_[0])
print("b =",model.intercept_)
print("R2 =",r2)

# save parameters
params = {
    "model":"linear",
    "a":float(model.coef_[0]),
    "b":float(model.intercept_)
}

with open(OUT_FILE,"w") as f:
    json.dump(params,f,indent=4)

print("Saved calibration parameters:",OUT_FILE)

# plot result
plt.scatter(preds,distances,label="samples")
plt.plot(preds,y_pred,label="fit")
plt.xlabel("Predicted Depth (MiDaS)")
plt.ylabel("Distance (meters)")
plt.title("Calibration Result")
plt.legend()
plt.show()