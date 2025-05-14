### ML SKILL ASSESMENT 1

#CODE
```
/*
Developed by :Vishal s
Register Number:212224040364
*/

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


df = pd.read_csv("/content/FuelConsumption.csv")
df.head()


# Q1: Scatter plot between cylinder vs Co2Emission (green color)
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green')
plt.xlabel('Cylinders')
plt.ylabel('CO2 Emissions')
plt.title('Cylinders vs CO2 Emissions')
plt.show()


# Q2: Compare Cylinders vs CO2Emissions and EngineSize vs CO2Emissions
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='blue', label='Cylinders')
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='red', label='Engine Size')
plt.xlabel('Feature Value')
plt.ylabel('CO2 Emissions')
plt.title('Comparison of Features vs CO2 Emissions')
plt.legend()
plt.show()


# Q3: Add FuelConsumption_comb to the scatter plot
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='blue', label='Cylinders')
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='red', label='Engine Size')
plt.scatter(df['FUELCONSUMPTION_COMB'], df['CO2EMISSIONS'], color='green', label='Fuel Consumption') # Corrected column name
plt.xlabel('Feature Value')
plt.ylabel('CO2 Emissions')
plt.title('Multiple Features vs CO2 Emissions')
plt.legend()
plt.show()


# Q4: Model - Cylinders -> CO2 Emissions
X_cyl = df[['CYLINDERS']]
y = df['CO2EMISSIONS'] 
X_train, X_test, y_train, y_test = train_test_split(X_cyl, y, test_size=0.2, random_state=42)
model_cyl = LinearRegression()
model_cyl.fit(X_train, y_train)
pred_cyl = model_cyl.predict(X_test)
print("Q4 - Model Accuracy (Cylinders):", r2_score(y_test, pred_cyl))

# Q5: Model - FuelConsumption_comb -> CO2 Emissions
X_fuel = df[['FUELCONSUMPTION_COMB']] # Corrected column name
y = df['CO2EMISSIONS']
X_train, X_test, y_train, y_test = train_test_split(X_fuel, y, test_size=0.2, random_state=42)
model_fuel = LinearRegression()
model_fuel.fit(X_train, y_train)
pred_fuel = model_fuel.predict(X_test)
print("Q5 - Model Accuracy (FuelConsumption_comb):", r2_score(y_test, pred_fuel))

#Q6: Different train-test ratios
ratios = [0.9,0.5,0.6,0.2,0.8]
print("\nQ6 - Accuracy for different train-test splits:")
for r in ratios:
    X_train, X_test, y_train, y_test = train_test_split(X_fuel, y, test_size=r, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = r2_score(y_test, pred)
    print(f"Test Size = {r:.1f} --> Accuracy = {score:.4f}")
```

#OUTPUT
![image](https://github.com/user-attachments/assets/e17c08d5-87f9-4eba-9237-950e55250fcc)

Q1
![image](https://github.com/user-attachments/assets/1dad5366-0486-48cc-b9fe-59fb395030ef)

Q2
![image](https://github.com/user-attachments/assets/cdc7ee18-0e74-438c-840e-ce56bf080cdc)

Q3
![image](https://github.com/user-attachments/assets/20b90e97-811c-4724-9362-8f848ed632b9)


Q4
![image](https://github.com/user-attachments/assets/82787065-f5e5-4d79-994f-7dae289eeb3f)

Q5
![image](https://github.com/user-attachments/assets/4c4d3197-c14f-4b02-a942-30c4a53325d4)

Q6
![image](https://github.com/user-attachments/assets/3007213f-947c-4d2b-8bb9-6eb6abd9451b)

#RESULT

