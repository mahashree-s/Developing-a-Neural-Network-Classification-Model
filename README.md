# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="757" height="707" alt="image" src="https://github.com/user-attachments/assets/c695e237-bdde-4e5b-86e0-911474162207" />

## DESIGN STEPS
### STEP 1: Import the required Python libraries and load the customer segmentation dataset using Pandas.

### STEP 2: Remove unnecessary columns and handle missing values using appropriate techniques.

### STEP 3: Convert categorical data into numerical values using Label Encoding.

### STEP 4: Split the dataset into training and testing sets and perform feature scaling using StandardScaler.

### STEP 5: Build a neural network using PyTorch with two hidden layers containing 32 and 16 neurons. Train the model using ReLU activation, CrossEntropyLoss and Adam optimizer.

### STEP 6: Evaluate the trained model using a confusion matrix, classification report and accuracy score. Predict the customer segment for a new sample.



## PROGRAM

### Name:MAHASHREE S

### Register Number:212225230163

```python


import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from torch.utils.data import TensorDataset, DataLoader



```

```python
df=pd.read_csv("customers.csv")
df
```

```python
df.shape
```

```python
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

df
```


```python
df.fillna({
    "Work_Experience": 0,
    "Family_Size": df["Family_Size"].median()
}, inplace=True)
df
```


```python
print(df.columns)
```


```python
cat_cols = ["Gender", "Ever_Married", "Graduated", "Profession", "Spending_Score", "Var_1", "Segmentation"]
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])
df
```

```python
x = df.drop(columns=["Segmentation"])

y = df["Segmentation"].values

print(x.head())
print(y[:10])
```


```python
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", x_train.shape)
print("Testing Data:", x_test.shape)
```



```python
scaler1 = StandardScaler()

x_train = scaler1.fit_transform(x_train)

x_test = scaler1.transform(x_test)

print(x_train[:5])
```



```python
x_train = torch.FloatTensor(x_train)

x_test = torch.FloatTensor(x_test)

y_train = torch.LongTensor(y_train)

y_test = torch.LongTensor(y_test)
```




```python
trl = DataLoader(train, batch_size=16, shuffle=True)

ts1 = DataLoader(test, batch_size=16, shuffle=False)
```



```python
model = classification_model(input_size=x_train.shape[1])

criterion = nn.CrossEntropyLoss()

op = optim.Adam(model.parameters(), lr=0.001)

print(model)
```



```python
epochs = 200

loss_values = []

for i in range(epochs):

    model.train()

    total_loss = 0

    for a, b in trl:

        op.zero_grad()

        pred = model(a)

        loss = criterion(pred, b)

        loss.backward()

        op.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(trl)

    loss_values.append(avg_loss)

    if i % 20 == 0:
        print(f"Loss {i}/{epochs}: {avg_loss:.4f}")
```


```python
plt.figure(figsize=(8, 5))

plt.plot(loss_values)

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.title("Neural Network Training Loss")

plt.show()
```


```python
model.eval()

pre = []

act = []

with torch.no_grad():

    for a, b in ts1:

        output = model(a)

        _, predicted = torch.max(output, 1)

        pre.extend(predicted.numpy())

        act.extend(b.numpy())

print("Actual Values:", act[:10])

print("Predicted Values:", pre[:10])
```


```python
cm = confusion_matrix(act, pre, labels=[0, 1, 2, 3])

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["A", "B", "C", "D"],
    yticklabels=["A", "B", "C", "D"]
)

plt.xlabel("Predicted Labels")

plt.ylabel("Actual Labels")

plt.title("Confusion Matrix")

plt.show()
```


```python
print(classification_report(
    act,
    pre,
    labels=[0, 1, 2, 3],
    target_names=["A", "B", "C", "D"],
    zero_division=0
))
accuracy = accuracy_score(act, pre)

print("Model Accuracy:", accuracy * 100, "%")


name = "Mahashree S"
reg_no = "212225230163"

sample = x.iloc[[0]]

sample_scaled = scaler1.transform(sample)

sample_tensor = torch.FloatTensor(sample_scaled)

model.eval()

with torch.no_grad():
    output = model(sample_tensor)
    _, predicted = torch.max(output, 1)

segments = ["A", "B", "C", "D"]

predicted_class = segments[predicted.item()]

actual_class = segments[df["Segmentation"].iloc[0]]

print("New Sample Data Prediction\n")

print("Name:", name)
print("Register No:", reg_no)

print("Predicted class for sample input:", predicted_class)

print("Actual class for sample input:", actual_class)
```

### Dataset Information
<img width="842" height="396" alt="image" src="https://github.com/user-attachments/assets/d1b1cf9c-b070-4903-ac28-bda007a1eccf" />


### OUTPUT


## Confusion Matrix
<img width="646" height="442" alt="image" src="https://github.com/user-attachments/assets/7426b742-01b7-4aa4-a30d-cecbedaebda5" />

## Classification Report
<img width="730" height="306" alt="image" src="https://github.com/user-attachments/assets/1b5f56b2-d1b9-46a3-ade3-976a8b88d886" />

### New Sample Data Prediction
<img width="874" height="597" alt="image" src="https://github.com/user-attachments/assets/3c91d1a7-0398-4b3c-817e-fd741e951624" />

## RESULT
Thus, a neural network classification model was successfully developed using PyTorch to classify customers into four segments (A, B, C and D). The model was trained and evaluated using the customer segmentation dataset. The performance was analyzed using a confusion matrix, classification report and new sample data prediction.
