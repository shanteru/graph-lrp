# Graph Layer-wise Relevance Propagation (GLRP)
This is an implementation of the Layer-wise Relevance Propagation (LRP) method for [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375).
The code here is devoted to the paper [Explaining decisions of Graph Convolutional Neural Networks: patient specific molecular sub-networks responsible for metastasis prediction in breast cancer](https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-021-00845-7).
The version of the software that was published with the paper uses the Tensorflow 1.x and is under this [commit](https://gitlab.gwdg.de/UKEBpublic/graph-lrp/-/tree/2bf6cdf8ff15eb1498bc60a607515ea43b89f135).  

Current version of the code uses Tensorflow 2.x.
The implementation of LRP for Graph CNN is in the *components* folder.
The folder *lib* contains modifed code from Michaël Defferrard's [Graph CNN](https://github.com/mdeff/cnn_graph) repository.
The visualization of the results can be found [on this website](http://mypathsem.bioinf.med.uni-goettingen.de/MetaRelSubNetVis).

The file *run_glrp_ge_data_record_relevances.py* runs Graph Layer-wise Relevance Propagation (GLRP) to generate gene-wise relevances for individual breast cancer patients. 
The details are in the [paper](https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-021-00845-7).

The file *run_glrp_grid_mnist.py* executes training of GCNN on the MNIST data and applies GLRP to it.

# Different LRP variants

    
## Requirements
To run the software one needs tensorflow, pandas, scipy, sklearn and matplotlib installed. I have made a requirement.txt (updated as of 11/1/23) which you can just simple 
*pip install -r requirements.txt*

## Breast Cancer Data
The preprocessed breast cancer data is under this [link](http://mypathsem.bioinf.med.uni-goettingen.de/resources/glrp). It contains three zip-archived csv files:  
Gene expression  *GEO_HG_PPI.csv.zip*  
Adjacency matrix *HPRD_PPI.csv.zip*  
Patient labels *labels_GEO_HG.csv.zip*  
To run the code, download the csv files into *graph-lrp/data/GE_PPI* directory.


## License
The source code is released under the terms of the MIT license
