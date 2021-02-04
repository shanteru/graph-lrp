# Graph Layer-wise Relevance Propagation (GLRP)
This is an implementation of Layer-wise Relevance Propagation (LRP) for [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375).
The code here is devoted to the paper "Explaining decisions of Graph Convolutional Neural Networks: patient specific molecular sub-networks responsible for metastasis prediction in breast cancer".
The implementation of LRP for Graph CNN is in *components* folder.
The folder *lib* contains modifed by me code from Michaël Defferrard's [Graph CNN](https://github.com/mdeff/cnn_graph) repository.
The visualization of the results can be found [on this website](http://mypathsem.bioinf.med.uni-goettingen.de/MetaRelSubNetVis).
## Breast Cancer Data
The preprocessed breast cancer data is under this [link](http://mypathsem.bioinf.med.uni-goettingen.de/resources/glrp). It contains three zip-archived csv files: 
Gene expression  *GEO_HG_PPI.csv.zip*
Adjacency matrix *HPRD_PPI.csv.zip*
Patient labels *labels_GEO_HG.csv.zip*
To run the code, download the csv files into graph-lrp/data/GE_PPI directory.


## License
The source code is released under the terms of the MIT license
