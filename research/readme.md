### Problems in the field:

1. Data representation: How to turn chemical data into something a model can operate on? 
  - Images and videos are swtraightforward to represent as tensorts of pixel values. Text requires a little more thought to put into tokens. But molecules are harder - they have a larger vocabulary of parts (number of atoms and bonds), as well as important stucture in 3d space. The representation problem is how to encode molecules so their important characteristics (whatever they are, we aren't sure!) are preserved.

2. Data scarcity
- There's a huge corpus of videos, images, and text on the web to use to train models. Chemical data is a lot more scarce. This means that models need to make the most of available data and generalize rapidly - like training a useful LLM with only a single paragraph of text.

### Papers & Notes
[Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212)

This paper was on karpathy's 30u30 list, so it's the first one I read.
Important ideas:

- Molecular data has lots of isomorphisms that should be taken advantage of in the encoding process - translations, permutations, rotations.
- The right graph traversal algorithms could be used to bring down the compute needs of training, veryu similar to how convolutions shrink images in image processing problems.
- Message functions: these are functions that "pass" messages along the structure of a molecule. Think of these as very similar to how a transformer "gathers" context from tokens earlier in a string.
  - While the MPNN maintains individual hidden states for each node, it does eventually produce a graph-level representation.
  - This happens in the readout phase, after T steps of message passing.
  - The readout function R takes the set of final node states {h^T_v | v ∈ G} and produces a single feature vector for the entire graph.

[Directed Message Passing Based on Attention for Prediction of Molecular Properties](https://arxiv.org/pdf/2305.14819)

This paper is best understood in comparison to the one before:

- Instead of treating molecules as undirected graphs, treat them instead as directed (with each bond being 2 directed edges). This can prevent signals from being passed in loops or repeated with a little bit of math. 
- Since we now have a sequence, we can get closer to using a self attention mechanism - letting molecules along a directed set of edges "pass information" to "future" molecules along that path. 
  - This attention mechanism can be trained with BERT style masking of random atoms. 
- A "supervirtual" node - a virtual node attached to all other nodes to facilitate passing messages across distant nodes. Sticking with the LLM metaphors, this would be like a "virtual word" that comes immediately before every other word in the text. 
- Performance of directed models is slower than undirecteds, but with better performance. 

[BiuoAct-Het: A Hererogeneous Siamese Neural Network for Bioactivity Prediction Using Novel Bioactivity Representation](https://paperswithcode.com/paper/bioact-het-a-heterogeneous-siamese-neural)