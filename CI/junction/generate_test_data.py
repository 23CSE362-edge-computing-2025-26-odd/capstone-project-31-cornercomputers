import pandas as pd
import numpy as np


num_days = 2
junctions = [1]  
start_date = "2025-01-01"


date_rng = pd.date_range(start=start_date, periods=num_days*24, freq="H")

data = []
for j in junctions:
    
    base = np.random.randint(50, 200)  
    
    for dt in date_rng:
        hour = dt.hour
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            vehicles = base + np.random.randint(50, 100)  
        else:
            vehicles = base + np.random.randint(-20, 20) 
        
        vehicles = max(vehicles, 0)
        data.append([dt, j, vehicles])


df_test = pd.DataFrame(data, columns=["DateTime", "Junction", "Vehicles"])

df_test["ID"] = range(1, len(df_test) + 1)

df_test.to_csv("traffic_test.csv", index=False)
print("Synthetic test data saved as traffic_test.csv")
print(df_test.head(10))
