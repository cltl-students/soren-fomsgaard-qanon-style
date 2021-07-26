# NOTE This file is just for exploring pretrained Qanon embeddings.


# Can be loaded with grasp. Uses same format as GloVE (ordered dict).

#%%
from utils.grasp import Embedding


#%%


# %%

with open('../../../Data/qanon/qanon_embeddings_phrased.txt', mode='r', encoding='utf-8') as f:
    q = Embedding.load(f, 'txt')

# %%
q.similar('biden', 5)
# %%

#q.values()
# %%
#len(q.keys())
# %%
