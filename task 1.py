# import library
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file
df = pd.read_csv(r"D:\task\adult.csv")

# Display column names
print(df.columns)

# plot bar chart
plt.figure(figsize=(8,5))
plt.bar(df['Gender'], df['Income'], color = "red")
plt.title("Bar Chart")
plt.xlabel("Gender")
plt.ylabel("Income")
plt.show()

# plot histogram
plt.figure(figsize=(8,5))
plt.hist(df['Age'], bins=10)
plt.title("Histogram")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()