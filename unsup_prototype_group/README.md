# Testing different Model and Loss versions for unsupervised group Prototype Learning

See model_version_overview.txt for an overview over the notations and names.
Each model version has its own bash script in scripts/

### Note:
Place the .npy files of the data generation script in the data folder, if they are not present 
by default. 

### General remarks 

model_customae.py contains the base ae code that is being used in all other versions.

data.py contains the Dataset class. 

utils.py contains plotting functions for all model types.

autoencoder_helpers.py contain additional funcitons required for the Prototype AE. 