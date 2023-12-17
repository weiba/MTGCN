# MTGCN
A Multi-Task learning method, called MTGCN, based on the Graph Convolutional Network to identify cancer driver genes
Run the MTCGN should under the environment of python >=3.7 , pytorch >=1.6.0 and pytorch geometric >=1.7.0
1. invoke the deepwalk.py to obtain structure features. 
2. invoke the MTGCN.py

The final result return a list of predicted value of each unlabeled gene.

Cite：
Peng, Wei, Qi Tang, Wei Dai, and Tielin Chen. "Improving cancer driver gene identification using multi-task learning on graph convolutional network." Briefings in Bioinformatics 23, no. 1 (2022): bbab432.
