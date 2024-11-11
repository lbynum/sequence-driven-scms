# sequence-driven-scms
Code for "Language Models as Causal Effect Generators" implementing sequence-driven structural causal models (SD-SCMs).

# benchmark

The `data` folder contains 2000 example SD-SCM-generated datasets for benchmarking treatment effect estimation algorithms (1000 from GPT-2, 1000 from Llama-3-8b). The notebook `benchmark.ipynb` replicates all estimation methods tested in the example benchmark (R and Python).

# files and usage 
- `confounder_collider.ipynb`: example usage of the functions in `sdscm.py` to generate two SD-SCMs over the same set of variables (one with a confounder, another with a collider)
- `bcancer_generation.ipynb`: example generation of a breast cancer SD-SCM using the config file `breast_cancer_config.json`
- `bcancer_plots.ipynb`: some plots of the generated breast cancer datasets
