import matplotlib.pyplot as plt
import numpy as np


regions = ['Penumbra', 'Full Sunlight', 'Umbra']
power_levels = [0.3, 1.0, 0.0] 
x = np.arange(len(regions))
y = power_levels
plt.figure(figsize=(10, 6))
plt.step(x, y, where='mid', linewidth=3, color='#f1c40f')
plt.xticks(x, regions)
plt.ylim(-0.1, 1.2)
plt.ylabel('Solar Power Output (Normalized)')
plt.xlabel('Orbital Phase / Time')
plt.title('Solar Power Profile')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.fill_between(x, y, step="mid", alpha=0.2, color='gold')
plt.tight_layout()
plt.savefig('eclipse_visualizer.png')