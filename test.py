import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
import sklearn as sk

print(np.__version__)
print(pd.__version__)
print(sk.__version__)
print(scipy.__version__)

x=[1,2,3,4,5]
y=[1,3,5,7,9]

plt.scatter(x,y)
plt.plot(x,y)
plt.show()
