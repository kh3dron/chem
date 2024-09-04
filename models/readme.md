# Models 

Replicating and tinkering with experiments from papers. Current task is to replicate a couple approaches on the Toxc21 dataset. From [Papers with code](https://paperswithcode.com/dataset/tox21-1): 

```
The Tox21 data set comprises 12,060 training samples and 647 test samples that represent chemical compounds. There are 801 "dense features" that represent chemical descriptors, such as molecular weight, solubility or surface area, and 272,776 "sparse features" that represent chemical substructures (ECFP10, DFS6, DFS8; stored in Matrix Market Format ). Machine learning methods can either use sparse or dense data or combine them. For each sample there are 12 binary labels that represent the outcome (active/inactive) of 12 different toxicological experiments. Note that the label matrix contains many missing values (NAs). The original data source and Tox21 challenge site is https://tripod.nih.gov/tox21/challenge/.
```

- sampleCode.py, sampleCode.R
    - Default random forest classifier that comes with the [Tox21 dataset](http://bioinf.jku.at/research/DeepTox/tox21.html). Added a comparison to the SOTA of the dataset, so you can see the random forest is pretty decent
- tox21_explorer.py
    - scratch pad for examining tox21 data
- attempt1.py
    - WIP #1 for tox21 classification