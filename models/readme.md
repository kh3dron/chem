# Models 

Replicating and tinkering with experiments from papers. Current task is to replicate a couple approaches on the Toxc21 dataset. From [Papers with code](https://paperswithcode.com/dataset/tox21-1): 

```
The Tox21 data set comprises 12,060 training samples and 647 test samples that represent chemical compounds. There are 801 "dense features" that represent chemical descriptors, such as molecular weight, solubility or surface area, and 272,776 "sparse features" that represent chemical substructures (ECFP10, DFS6, DFS8; stored in Matrix Market Format ). Machine learning methods can either use sparse or dense data or combine them. For each sample there are 12 binary labels that represent the outcome (active/inactive) of 12 different toxicological experiments. Note that the label matrix contains many missing values (NAs). The original data source and Tox21 challenge site is https://tripod.nih.gov/tox21/challenge/.
```

So, to be really clear about the shape of the data: 
- train dense: 12k rows
    - 801 columns, each column being 1 property of a molecule.
    - Each molecule has a name (the name column), which can be used as a lookup to get it's SMILES format, at a link like [this.](https://pubchem.ncbi.nlm.nih.gov/substance/170465670) This can be used to pull in other data not strictly in the tox training data. 
- Test data: 600 rows, 12 columns. 
    - 11 columns hold NaNs, the remaining one has 0 or 1. Each input (tested) molecule is only being evaluated for a single class.

`sampleCode.py`: Default random forest classifier that comes with the [Tox21 dataset](http://bioinf.jku.at/research/DeepTox/tox21.html). Added a comparison to the SOTA of the dataset, so you can see the random forest is pretty decent

`Attempt 1`: KNN classification