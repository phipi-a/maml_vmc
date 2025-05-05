# MAML-VMC
## Requirements 
- Python ≥ 3.12.7
- [Weights & Biases](https://wandb.ai/)  (W&B) account and a running instance
## Installation 
```bash
pip install -e .
```
## Dataset Generation 
Prepare the datasets using the following commands:
### H3p System 
```bash
python maml_vmc/dataset/ds_h3p.py -i exp/ds/train_H3p.json -o exp/ds/train_H3p
python maml_vmc/dataset/ds_h3p.py -i exp/ds/test_H3p.json -o exp/ds/test_H3p
```
### Hydrogen Chains 
```bash
python maml_vmc/dataset/ds_h_chains.py -i exp/ds/train_h2_h4_chains.json -o exp/ds/train_h2_h4_chains
python maml_vmc/dataset/ds_h_chains.py -i exp/ds/test_h6_chain.json -o exp/ds/test_h6_chain
python maml_vmc/dataset/ds_h_chains.py -i exp/ds/test_h10_chains.json -o exp/ds/test_h10_chains
```
## Training 
`{i}` = Geometry index (e.g., 0, 1, 2...)

`{x}` = Number of training samples

`{z}` = Number of batches
### Meta-Training (H2–H4 Chains) 
```bash
python maml_vmc/main.py fit --config configs/hchain_2-4_meta.yaml
```
### Fine-Tuning / Optimization 
#### H6 Chain (With Meta-Training) 
```bash
python maml_vmc/main.py fit --config configs/hchain_6.yaml --trainer.trans_weights path/to/checkpoint.pkl
```
#### H10 Chain (With Meta-Training) 
```bash
python maml_vmc/main.py fit \
  --config configs/hchain_10.yaml \
  --trainer.trans_weights path/to/checkpoint.pkl \
  --trainer.data_sampler.file_name config_n10_el10_i{i}.pkl
```
#### H10 Chain (Without Meta-Training) 
```bash
python maml_vmc/main.py fit \
  --config configs/hchain_10.yaml \
  --trainer.data_sampler.file_name config_n10_el10_i{i}.pkl
```
### Gradient Sensitivity Analysis 
#### With Meta-Trained Model 
```bash
python maml_vmc/main.py fit \
  --config configs/hchain_10.yaml \
  --trainer.trans_weights path/to/checkpoint.pkl \
  --trainer.data_sampler.file_name config_n10_el10_i{i}.pkl \
  --trainer.data_sampler.num_samples {x} \
  --trainer.num_batches {z}
```
#### Without Meta-Trained Model 
```bash
python maml_vmc/main.py fit \
  --config configs/hchain_10.yaml \
  --trainer.data_sampler.file_name config_n10_el10_i{i}.pkl \
  --trainer.data_sampler.num_samples {x} \
  --trainer.num_batches {z}
```
## H3p In-Domain Experiments 
### Meta-Training 
#### With Random Reinitialization 
```bash
python maml_vmc/main.py fit --config configs/h3p_meta.yaml
```
#### Without Random Reinitialization 
```bash
python maml_vmc/main.py fit --config configs/h3p_meta.yaml --trainer.reinit_layers False
```
### Fine-Tuning on Geometries 
#### Independent Optimization 
```bash
python maml_vmc/main.py fit \
  --config configs/h3p.yaml \
  --trainer.data_sampler.file_name config_n3_el2_i{i}.pkl
```
#### Using Meta-Trained Model 
```bash
python maml_vmc/main.py fit \
  --config configs/h3p.yaml \
  --trainer.data_sampler.file_name config_n3_el2_i{i}.pkl \
  --trainer.trans_weights path/to/checkpoint.pkl
```
## Evaluation 
Run evaluation on a specific geometry:
```bash
python maml_vmc/main.py validate \
  --config path/to/config.yaml \
  --trainer.ckpt_path path/to/checkpoint.pkl
```
## Notes 
To enable 64-bit precision, set the following environment variable before running your script:
```bash
export JAX_ENABLE_X64=1
```
## References 
This implementation builds on the excellent [DeepErwin](https://github.com/mdsunivie/deeperwin)  project.
The reference values of the test dataset (`/maml_vmc/dataset/ds/test_*`) and weight-sharing results in `/results` are provided by:


**Scherbela, M., Reisenhofer, R., Gerard, L. et al.** 
*Solving the electronic Schrödinger equation for multiple nuclear geometries with weight-sharing deep neural networks*.\
Nature Computational Science , 2, 331–341 (2022).
[https://doi.org/10.1038/s43588-022-00228-x](https://doi.org/10.1038/s43588-022-00228-x)


