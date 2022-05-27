# Description

Spatom is a state-of-the-art software to predict protein-protein interaction sites(PPIS). This program is for users to predict PPIS.
                                                                                
This software is free to use, modify, redistribute without any restrictions, except including the license provided with the distribution. 

Spatom can be used under both Linux and Windows environments. The Spatom is hosted on our webserver ([http://liulab.top/Spatom/server](http://118.190.151.48/Spatom/server)).

<p align="center">
    <img src="/Spatom/IMG/workflow_Spatom.png" width="100%">
</p>

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

Please provide the PSSM files, which should be named as proten_chain.pssm (e.g., 3fju_A.pssm).

2) Providing CX and DPX

Please provide the CX and DPX files, which can be calculated by [PSAIA](http://complex.zesoi.fer.hr/index.php/en/10-category-en-gb/tools-en/19-psaia-en). 
Users can directly apply the table output from PSAIA to Spatom (e.g., 3fju_A_202202082126_unbound.tbl). Please check the boxes of "Depth Index" and "Protusion Index" and choose "Analyse by chain" using the structure analyser.

3) Providing PDB

Spatom allows users to provide a protein PDB file, or just the protein name because Spatom can automatically download the corresponding PDB file. 


## Predicting protein-protein interaction sites

usage: Spatom.py [-h] [--protein PROTEIN] [--chain CHAIN] [--PSSM PSSM] [--pdb PDB] [--CX_DPX CX_DPX]
                 [--output OUTPUT]

** Required **

  --protein PROTEIN, -p PROTEIN
                        protein name

  --chain CHAIN, -c CHAIN
                        chain name

  --PSSM PSSM, -m PSSM  path_to_PSSM/PSSM_file

  --CX_DPX CX_DPX, -d CX_DPX
                        path_to_CX_DPX/CX_DPX_file

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
$ python ./Spatom-1.0/Spatom.py -p 3fju -c a -m ./Spatom-1.0/PSSM/3FJU_A.pssm -d ./Spatom-1.0/CX_DPX/3fju_A_202202082126_unbound.tbl
```

The predicted result will be stored in output_path/predict_3fju_a.txt.

## Testing Spatom

Test data are provided with the software distribution in the ./sample_test directoriy.

a) change to the path_to_Spatom-1.0 directory

```
$ cd path_to_Spatom-1.0/
```

b) run Spatom

```
$ python Spatom.py -p 3fju -c a -m ./sample_test/PSSM/3FJU_A.pssm -d ./sample_test/CX_DPX/3fju_A_202202082126_unbound.tbl -b ./sample_test/real_PDB/pdb3fju.ent
```
# Notes

Please check your PDB file first. If the PDB file misses some important atoms, please use a software (e.g. , [PDBFixer](https://github.com/openmm/pdbfixer)) to fix it.

# Contact

Any questions, problems, bugs are welcome and should be dumped to Haonan Wu <hnw.bio@outlook.com>.

Created on Feb. 22, 2022.
