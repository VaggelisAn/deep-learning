import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class NeuMF_Experiment:
    predictive_factors: int
    negative_sampling_factor: int
    pretrained: bool
    MLP_layer_count: int

    best_HR10:  Optional[float] = 1.0
    mean_HR10:  Optional[float] = 1.0
    std_HR10:   Optional[float] = 1.0

    best_NDCG:  Optional[float] = 1.0
    mean_NDCG:  Optional[float] = 1.0
    std_NDCG:   Optional[float] = 1.0 

    best_HR10s_array: Optional[np.ndarray] = np.array([1])
    best_NDCGs_array: Optional[np.ndarray] = np.array([1])

    plt_color: str = '#000000'

# iterate over number of layers
not_pretrained = []
pretrained = []

# Experiment with J=8, NOT pretrained
# best_HR10s[0]: 1 layer etc..
best_HR10s = [0.6500530222693531, 0.6553552492046659, 0.6617179215270413]
best_NDCGs = [0.37650572449776326, 0.3769742650204071, 0.38148240093108265]
plt_colors = ['#00ffa2', '#00ffff', '#0d00ff']

for i in range(3):
    plt_color = '#000000'
    best_HR10 = 1.0
    best_NDCG = 1.0

    not_pretrained.append(
        NeuMF_Experiment(
        predictive_factors = 8,
        negative_sampling_factor = 4,
        pretrained = False,
        MLP_layer_count = i,

        best_HR10=best_HR10s[i],
        best_NDCG=best_NDCGs[i],
        plt_color=plt_colors[i]
        )
    )

plt_colors = ['#f6ff00', '#ff8400', '#ff2f00']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
# Use wspace (width space) instead of hspace to add padding between the left and right plots
fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.15, wspace=0.2)
num_of_layers = np.array([1, 2, 3])

ax1.plot(best_HR10s, num_of_layers)
ax1.set_title('HR@10 for 1 to 3 MLP Layers')
ax2.plot(best_NDCGs, num_of_layers)
ax2.set_title('NDCGs for 1 to 3 MLP Layers')

plt.show()
plt.savefig('not_pretrained_j8.png', dpi=300)
# plt.close(fig) # Free up memory before rendering the next plot