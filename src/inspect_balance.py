import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

y = np.load('data/processed/y_labels.npy')

adl = int((y==0).sum())
fall = int((y==1).sum())

print(f"ADL windows:  {adl}")
print(f"FALL windows: {fall}")
print(f"Ratio fall/ADL: {fall/max(adl,1):.4f}")

plt.figure()
plt.bar(['ADL (no fall)', 'Fall'], [adl, fall])
plt.title('Class Balance: Falls vs ADL (window-level)')
plt.xlabel('Class')
plt.ylabel('Number of windows')
plt.tight_layout()
plt.show()
