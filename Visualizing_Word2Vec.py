import pandas as pd
from sklearn.manifold import TSNE

vocab = list(your_gensim_model.wv.vocab)
model_vocab = your_gensim_model[vocab]
tsne = TSNE(n_components=4)
_tsne = tsne.fit_transform(model_vocab)

df = pd.DataFrame(_tsne, index=vocab, columns=['x', 'y'])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(df['x'], df['y'])
for word, pos in df.iterrows():
    ax.annotate(word, pos)
ax.set_xlim(right = 5)
ax.set_ylim(top = 5)
plt.show()
