# Unsupervised-Tree-LSTM

Part of the code was adopted from <a href='https://github.com/jihunchoi/unsupervised-treelstm'>jihunchoi/unsupervised-treelstm</a>.

The tree is constructed in a iterative manner. In each iteration, all possible parent nodes are composed, and then one of them is chosen greedily based on some scoring metric. So the number of nodes is reduces by one in every iteration. Finally we will be left with just one node, the root of the tree.

<a href='https://github.com/jihunchoi/unsupervised-treelstm'>jihunchoi/unsupervised-treelstm</a> has many features (like intra-attention) in building the tree but does not provide different options for choosing the scoring metric. This code provides an interface for that.
