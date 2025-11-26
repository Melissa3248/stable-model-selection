# Stabilizing black-box model selection with the inflated argmax

_Melissa Adrian, Jake A. Soloff, Rebecca Willett_

Model selection is the process of choosing from a class of candidate models given data. For instance, methods such as the LASSO and sparse identification of nonlinear dynamics (SINDy) formulate model selection as finding a sparse solution to a linear system of equations determined by training data. However, absent strong assumptions, such methods are highly unstable: if a single data point is removed from the training set, a different model may be selected. In this paper, we present a new approach to stabilizing model selection with theoretical stability guarantees that leverages a combination of bagging and an ``inflated'' argmax operation. Our method selects a small collection of models that all fit the data, and it is stable in that, with high probability, the removal of any training point will result in a collection of selected models that overlaps with the original collection. We illustrate this method in (a) a simulation in which strongly correlated covariates make standard LASSO model selection highly unstable, (b) a Lotka–Volterra model selection problem focused on identifying how competition in an ecosystem influences species' abundances, (c) a graph subset selection problem using cell-signaling data from proteomics, and (d) unsupervised $\kappa$-means clustering. In these settings, the proposed method yields stable, compact, and accurate collections of selected models, outperforming a variety of benchmarks.

## Getting Started

All package dependencies are specified in the ```environment.yml``` file, and to activate the conda environment with these necessary dependencies, run the following lines. Note that the prefix section in ```environment.yml``` will need to be changed to point toward the directory with your conda environments.

```
conda env create -f environment.yml
conda activate stable
```

All code and visualizations needed to reproduce the results presented in the paper are in the ```experiments``` folder.

## Cite this work
```
@misc{adrian2025stabilizingblackboxmodelselection,
      title={Stabilizing black-box model selection with the inflated argmax}, 
      author={Melissa Adrian and Jake A. Soloff and Rebecca Willett},
      year={2025},
      eprint={2410.18268},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2410.18268}, 
}
```

## MIT License 

Copyright (c) 2025 Melissa Adrian, Jake A. Soloff, Rebecca Willett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.