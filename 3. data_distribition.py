import numpy as np
import matplotlib.pyplot as plt
data_distribution = np.random.normal(loc=50, scale=10, size=1000)

plt.hist(data_distribution, bins=30, edgecolor="black")
plt.title('Histogram - Data Distribution')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()