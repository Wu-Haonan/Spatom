# Description

Spatom is a state-of-the-art software to predict protein-protein interaction sites (PPIS). This program is for users to predict PPIS for protein with multiple polypeptide chains in unbound state.
                                                                                
This software is free to use, modify, redistribute without any restrictions, except including the license provided with the distribution. 

Spatom can be used under both Linux and Windows environments. The Spatom is hosted on our webserver ([http://118.190.151.48/Spatom/server](http://118.190.151.48/Spatom/server)).

![Spatom_workflow](https://github.com/Wu-Haonan/Spatom/blob/main/IMG/Spatom_workflow.png)

# Requirements

Python 3.8.5

torch 1.10

torch-geometric 2.0.2

numpy 1.21.3

DSSP (user should install [DSSP](https://swift.cmbi.umcn.nl/gv/dssp/), the following command can be used to install DSSP)

```
$ conda install -c salilab dssp # for Linux
$ conda install -c speleo3 dssp # for Windows
```

freesasa 2.1.0

Biopython 1.7.9

sklean 0.23.2

If runing on GPU, Spatom needs

cuda 10.2

# Usage

Spatom needs users to provide the files as follows.

1) Providing PSSM

Please provide the PSSM files for all chains in PDB file, which should be named as proten_chain.pssm (e.g., 1A2K_A.pssm, 1A2K_B.pssm).

2) Providing PDB

Spatom allows users to provide a protein PDB file, or just the protein name because Spatom can automatically download the corresponding PDB file. 


## Predicting protein-protein interaction sites

usage: Spatom.py [-h] [--protein PROTEIN] [--PSSM PSSM] [--pdb PDB] [--output OUTPUT]

** Required **

  --protein PROTEIN, -p PROTEIN
                        protein name

  --PSSM PSSM, -m PSSM  path_to_PSSM_folder, save the PSSM files of corresponding chains into this folder

** Optional **

  -h, --help            show this help message and exit

  --pdb PDB, -b PDB     path_to_pdb/pdb_file

  --output OUTPUT, -o OUTPUT
                        path_to_output, defaut with path_to_Spatom-1.0/result

** Note **

Users must specify the '-p' item, i.e., protein's 4-letters PDB ID.

If the '-b' PDB item is empty in your command line, the corresponding PDB file will be downloaded automatically.

** Typical commands **

The following command is an example:

```
$ python ./Spatom-1.0/Spatom.py -p 1A2K -m ./Spatom-1.0/PSSM 
```

The predicted result will be stored in output_path/predict_1A2K.txt.

## Testing Spatom

Test data are provided with the software distribution in the ./sample_test directoriy.

a) change to the path_to_Spatom-1.0 directory

```
$ cd path_to_Spatom-1.0/
```

b) run Spatom

```
$ python Spatom.py -p 1A2K -m ./sample_test/PSSM -b ./sample_test/real_PDB/1A2K.pdb
```
# Notes

Please check your PDB file first. If the PDB file misses some important atoms, please use a software (e.g. , [PDBFixer](https://github.com/openmm/pdbfixer)) to fix it.

# Citation

If you use this code, please cite 

Haonan Wu, Jiyun Han, Shizhuo Zhang, Gaojia Xin, Chaozhou Mou, Juntao Liu, Spatom: a graph neural network for structure-based protein–protein interaction site prediction, Briefings in Bioinformatics, Volume 24, Issue 6, November 2023, bbad345, https://doi.org/10.1093/bib/bbad345

# Contact

Any questions, problems, bugs are welcome and should be dumped to Haonan Wu <hnw.bio@outlook.com>.

Created on Dec. 15, 2022.
