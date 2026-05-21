import matplotlib.pyplot as plt
import numpy as np


# ===== Aggregate Best Metrics =====
# not pretrained, 1 MLP layer
best_hrs_no_pretrain_1_layer = np.array([0.6288441145281018, 0.6500530222693531, 0.6458112407211029, 0.6436903499469777, 0.6426299045599152, 0.6500530222693531, 0.6352067868504772, 0.6341463414634146, 0.6436903499469777, 0.6224814422057264])
best_ndcgs_no_pretrain_1_layer = np.array([0.36722925556411823, 0.37650572449776326, 0.3689247300215812, 0.3671036054245824, 0.36882252215131933, 0.3745536239640359, 0.367690411179328, 0.36267145160870984, 0.37320137316596946, 0.3617361349368296])
# Best HR mean: 0.639661, std: 0.008703
# Best NDCG mean: 0.368844, std: 0.004537


# not pretrained, 2 MLP layers
best_hrs_no_pretrain_2_layer = 
best_ndcgs_no_pretrain_2_layer = 

# not pretrained, 3 MLP layers
best_hrs_no_pretrain_3_layer = 
best_ndcgs_no_pretrain_3_layer = 

# pretrained, 1 MLP layer
best_hrs_pretrained_1_layer = 
best_ndcgs_pretrained_1_layer = 

# pretrained, 2 MLP layers
best_hrs_pretrained_2_layer = 
best_ndcgs_pretrained_2_layer = 

# pretrained, 3 MLP layers
best_hrs_pretrained_3_layer = 
best_ndcgs_pretrained_3_layer = 

fig, (ax1, ax2) = plt.subplots(2)

num_of_layers = np.array([1, 2, 3])

ax1.plot()
ax1.set_title('HR@10 for 1 to 3 MLP Layers')
ax2.plot()
ax2.set_title('NDCGs for 1 to 3 MLP Layers')


plt.show()