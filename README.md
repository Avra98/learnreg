# Learning Sparsifying Transforms (1D)

## One-time setup
1. get a working conda install
1. conda env create --file environment.yml
1. conda activate learnreg1d

## Updating
1. `git pull`
1. `conda env update -f environment.yml`

## Working with the code
* `conda activate learnreg1d`

## mongodb
* on digilink: `mongod --dbpath monogodb/`
* on local machine:
  * `ssh -L 9998:sai.dhcp.egr.msu.edu:22 -N mccann13@scully.egr.msu.edu &`
  * `ssh -L 27017:localhost:27017 -N -p 9998 localhost -l mccann13 &`
  * `omniboard`
* point browser to `localhost:9000`


## Rebuttal experiments
Refer to the rebuttal experiment folder for the experiments performed for ICASSP rebuttal. 
