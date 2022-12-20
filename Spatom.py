import torch
from Bio.PDB import *
import argparse
import os
from Bio.PDB import PDBParser, PDBIO, Select
import numpy as np
from Bio.PDB.DSSP import DSSP
from Bio import PDB
from torch_geometric.data import Data
import torch.utils.data
import sys
import freesasa
import math
from model.model import Spatom
from torch_geometric.loader import DataLoader

BATCH = 1
LEARN_RATE = 0.001
HIDDEN_DIM = 1024
LAYERS = 7
DROPOUT = 0.1

def logo():
    print('\
*     ____              _                       *\n\
*    / ___| _ __   __ _| |_ ___  _ __ ___       *\n\
*    \___ \| \'_ \ / _` | __/ _ \| \'_ ` _ \      *\n\
*     ___) | |_) | (_| | || (_) | | | | | |     *\n\
*    |____/| .__/ \__,_|\__\___/|_| |_| |_|     *\n\
*          |_|                                  *')

def transpose(matrix):
    new_matrix = []
    for i in range(len(matrix[0])):
        matrix_raw = []
        for j in range(len(matrix)):
            matrix_raw.append(matrix[j][i])
        new_matrix.append(matrix_raw)
    return new_matrix

def PSSM_file_to_PSSM_dict(protein,chains,PSSM):
    Max_pssm = np.array([8., 9., 9., 9., 12., 9., 8., 8., 12., 9., 7., 8., 11.,
                    10., 9., 8., 8., 13., 10., 8.])

    Min_pssm = np.array([-10., -11., -12., -12., -11., -10., -11., -11., -11., -10., -11.,
                    -11., -10., -11., -12., -11., -10., -11., -10., -11.])
    PSSM_dict = {}
    PSSM_matrix = []
    for c in chains:
        PSSM_file = open(PSSM + '/' + protein.upper() + '_' + c.upper() + '.PSSM', 'r')
        for line in PSSM_file:
            if line != None and len(line.split()) > 40:
                PSSM_line = line.split()[2:22]
                PSSM_line = list(map(float, PSSM_line))
                PSSM_line = ((np.array(PSSM_line) - Min_pssm)/(Max_pssm - Min_pssm)).tolist()
                PSSM_matrix.append(PSSM_line)

    PSSM_dict[protein.lower()] = PSSM_matrix
    return PSSM_dict

def PDB_proressing(real_PDB,protein):
    class CA_Select(Select):
        def accept_residue(self, residue):
            return 1 if residue.id[0] == " " else 0

        def accept_atom(self, atom):
            if atom.get_name() == 'CA':
                return True
            else:
                return False
    class Residue_Select(Select):
        def accept_residue(self, residue):
            return 1 if residue.id[0] == " " else 0

    pdb = PDBParser(QUIET=1).get_structure(protein, real_PDB)
    io = PDBIO()
    io.set_structure(pdb)
    if not os.path.exists(sys.path[0]+"/CA_PDB"):
        os.makedirs(sys.path[0]+"/CA_PDB")
    io.save(sys.path[0]+'/CA_PDB/' + protein + ".pdb", CA_Select())
    if not os.path.exists(sys.path[0]+"/Residue_PDB"):
        os.makedirs(sys.path[0]+"/Residue_PDB")
    io.save(sys.path[0]+'/Residue_PDB/' + protein + ".pdb", Residue_Select())

def Dist_adj(protein,chains):
    def dictance(xyz, position):
        xyz = xyz - xyz[position]
        dictance = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2 + xyz[:, 2] ** 2).tolist()
        return dictance

    p = PDBParser(QUIET=1)
    structure = p.get_structure(protein, sys.path[0]+"/CA_PDB/" + protein + '.pdb')
    distance_list = []
    chain_list = []
    for c in chains:
        for residue in structure[0][c]:
            for atom in residue:
                chain_list.append(atom.get_vector().get_array())
    for i, center in enumerate(chain_list):
        distance_list.append(dictance(chain_list, i))
    distance_dict = {}
    distance_dict[protein.lower()] = np.array(distance_list)
    return distance_dict

def seq_and_one_hot(protein, chains):
    AA_index = {j: i for i, j in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    seq_dict = {}
    one_hot_dict = {}
    parser = PDBParser(QUIET=1)
    structure = parser.get_structure(protein, sys.path[0] + '/Residue_PDB/' + protein + ".pdb")
    ppb = PDB.PPBuilder()
    one_hot_list = []
    for c in chains:
        model = structure[0][c.upper()]
        seq = ''
        for cc in ppb.build_peptides(model):
            seq += cc.get_sequence()
        seq_dict[(protein + c).lower()] = seq
        for i in seq:
            zero = [0] * 20
            zero[AA_index[i]] = 1
            one_hot_list.append(zero)
    one_hot_dict[protein.lower()] = one_hot_list
    return seq_dict, one_hot_dict

class ChianSelect(Select):
    def __init__(self,chain_letter):
        self.chain_letter = chain_letter
    def accept_chain(self,chain):
        if chain.get_id()==self.chain_letter:
            return True
        else:
            return False

def get_RSA(protein,chains):
    RSA_dict = {}
    structure = freesasa.Structure(sys.path[0]+"/Residue_PDB/"+protein + '.pdb')
    result = freesasa.calc(structure,freesasa.Parameters({'algorithm' : freesasa.LeeRichards,'n-slices' : 100,'probe-radius' : 1.4}))
    residueAreas = result.residueAreas()
    RSA = []
    for c in chains:
        for r in residueAreas[c.upper()].keys():
            RSA_AA = []
            RSA_AA.append(min(1,residueAreas[c.upper()][r].relativeTotal))
            RSA_AA.append(min(1,residueAreas[c.upper()][r].relativePolar))
            RSA_AA.append(min(1,residueAreas[c.upper()][r].relativeApolar))
            RSA_AA.append(min(1,residueAreas[c.upper()][r].relativeMainChain))
            if math.isnan(residueAreas[c.upper()][r].relativeSideChain):
                RSA_AA.append(0)
            else:
                RSA_AA.append(min(1,residueAreas[c.upper()][r].relativeSideChain))
            RSA.append(RSA_AA)
    RSA_dict[protein.lower()] = RSA
    return RSA_dict

def dssp_feature(protein,chains,fasta_dict):
    SS_dict = {'H': 0, 'B': 1, 'E': 2, 'G': 3, 'I': 4, 'T': 5, 'S': 6, '-': 7}
    p = PDBParser(QUIET=1)
    pdb = p.get_structure(protein, sys.path[0]+'/Residue_PDB/' + protein + '.pdb')
    io = PDBIO()
    io.set_structure(pdb)
    # io.save(sys.path[0]+'/Residue_PDB/'+protein + '_' + chain.upper() + '.pdb', ChianSelect(chain.upper()))
    # structure = p.get_structure(protein,sys.path[0]+'/Residue_PDB/'+ protein + '_' + chain.upper() + '.pdb')
    model = pdb[0]
    dssp = DSSP(model, sys.path[0]+'/Residue_PDB/' + protein + '.pdb', dssp='mkdssp')
    dssp_matrix_complex = []
    for c in chains:
        dssp_matrix = []
        seq = ""
        ref_seq = fasta_dict[(protein + c).lower()]
        for i in dssp.keys():
            if i[0] == c:
                SS = dssp[i][2]
                AA = dssp[i][1]
                seq += AA
                phi = dssp[i][4]
                psi = dssp[i][5]
                raw = []
                raw.append(np.sin(phi * (np.pi / 180)))
                raw.append(np.sin(psi * (np.pi / 180)))
                raw.append(np.cos(phi * (np.pi / 180)))
                raw.append(np.cos(psi * (np.pi / 180)))
                ss_raw = [0] * 9
                ss_raw[SS_dict[SS]] = 1
                raw.extend(ss_raw)
                dssp_matrix.append(raw)
        pad = []
        pad.append(np.sin(360 * (np.pi / 180)))
        pad.append(np.sin(360 * (np.pi / 180)))
        pad.append(np.cos(360 * (np.pi / 180)))
        pad.append(np.cos(360 * (np.pi / 180)))
        ss_pad = [0] * 9
        ss_pad[-1] = 1
        pad.extend(ss_pad)
        pad_dssp_matrix = []
        p_ref = 0
        for i in range(len(seq)):
            while p_ref < len(ref_seq) and seq[i] != ref_seq[p_ref]:
                pad_dssp_matrix.append(pad)
                p_ref += 1
            if p_ref < len(ref_seq):  # aa matched
                pad_dssp_matrix.append(dssp_matrix[i])
                p_ref += 1
        if len(pad_dssp_matrix) != len(ref_seq):
            for i in range(len(ref_seq) - len(pad_dssp_matrix)):
                pad_dssp_matrix.append(pad)
        dssp_matrix_complex.extend(pad_dssp_matrix)
    dssp_dict = {}
    dssp_dict[protein.lower()] = dssp_matrix_complex
    return dssp_dict

def AA_property(protein,chains,seq_dict):
    Side_Chain_Atom_num = {'A': 5.0, 'C': 6.0, 'D': 8.0, 'E': 9.0, 'F': 11.0, 'G': 4.0, 'H': 10.0, 'I': 8.0, 'K': 9.0,
                           'L': 8.0, 'M': 8.0, 'N': 8.0, 'P': 7.0, 'Q': 9.0, 'R': 11.0, 'S': 6.0, 'T': 7.0, 'V': 7.0,
                           'W': 14.0, 'Y': 12.0}
    Side_Chain_Charge_num = {'A': 0.0, 'C': 0.0, 'D': -1.0, 'E': -1.0, 'F': 0.0, 'G': 0.0, 'H': 1.0, 'I': 0.0, 'K': 1.0,
                             'L': 0.0, 'M': 0.0, 'N': 0.0, 'P': 0.0, 'Q': 0.0, 'R': 1.0, 'S': 0.0, 'T': 0.0, 'V': 0.0,
                             'W': 0.0, 'Y': 0.0}
    Side_Chain_hydrogen_bond_num = {'A': 2.0, 'C': 2.0, 'D': 4.0, 'E': 4.0, 'F': 2.0, 'G': 2.0, 'H': 4.0, 'I': 2.0,
                                    'K': 2.0, 'L': 2.0, 'M': 2.0, 'N': 4.0, 'P': 2.0, 'Q': 4.0, 'R': 4.0, 'S': 4.0,
                                    'T': 4.0, 'V': 2.0, 'W': 3.0, 'Y': 3.0}
    Side_Chain_pKa = {'A': 7.0, 'C': 7.0, 'D': 3.65, 'E': 3.22, 'F': 7.0, 'G': 7.0, 'H': 6.0, 'I': 7.0, 'K': 10.53,
                      'L': 7.0, 'M': 7.0, 'N': 8.18, 'P': 7.0, 'Q': 7.0, 'R': 12.48, 'S': 7.0, 'T': 7.0, 'V': 7.0,
                      'W': 7.0, 'Y': 10.07}
    Hydrophobicity = {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': 3.9,
                      'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2,
                      'W': -0.9, 'Y': -1.3}

    AA_property_dict = {}
    AA_protein = []
    for c in chains:
        for AA in seq_dict[(protein+c).lower()]:
            AA_AA = []
            AA_AA.append(Side_Chain_Atom_num[AA])
            AA_AA.append(Side_Chain_Charge_num[AA])
            AA_AA.append(Side_Chain_hydrogen_bond_num[AA])
            AA_AA.append(Side_Chain_pKa[AA])
            AA_AA.append(Hydrophobicity[AA])
            AA_protein.append(AA_AA)
    AA_property_dict[protein.lower()] = AA_protein
    return AA_property_dict

def edge_weight(dist):
    matrix = dist.clone()
    softmax = torch.nn.Softmax(dim=0)
    dist = softmax(1./(torch.log(torch.log(dist+2))))
    dist[matrix>14] = 0
    return dist

def feature_extract(protein,chains,PSSM_file):
    PSSM_dict = PSSM_file_to_PSSM_dict(protein, chains, PSSM_file)
    Dist_dict = Dist_adj(protein, chains)
    fasta_dict, onehot_dict = seq_and_one_hot(protein, chains)
    DSSP_dict = dssp_feature(protein, chains, fasta_dict)
    RSA_dict = get_RSA(protein, chains)
    AA_property_dict = AA_property(protein, chains, fasta_dict)
    Datasets = []
    feature = []
    RSA = []
    for k in range(len(onehot_dict[protein.lower()])):
        AA = []
        AA.extend(onehot_dict[protein.lower()][k])
        AA.extend(PSSM_dict[protein.lower()][k])
        AA.extend(RSA_dict[protein.lower()][k])
        AA.extend(AA_property_dict[protein.lower()][k])
        AA.extend(DSSP_dict[protein.lower()][k])
        feature.append(AA)
        RSA.append(RSA_dict[protein.lower()][k][0])
    pos = np.where(np.array(RSA) >= 0.05)[0].tolist()
    Dist = edge_weight(torch.tensor(Dist_dict[protein.lower()]))[pos, :][:, pos]
    feature = torch.tensor(np.array(feature)[pos, :], dtype=torch.float)
    adj = torch.tensor(np.where(np.array(Dist_dict[protein.lower()]) < 14, 1, 0)[pos, :][:, pos])
    data = Data(x=feature)
    data.dist = Dist
    data.POS = pos
    length = len(onehot_dict[protein.lower()])
    data.length = length
    data.adj = adj
    Datasets.append(data)
    return Datasets

def predict(protein,chains,pdb_file,output_path,PSSM):
    def test(model,test_set,output_path):
        test_loader = DataLoader(test_set, batch_size=1)
        model.eval()
        all_pred = []
        with torch.no_grad():
            for step, data in enumerate(test_loader):
                feature = torch.autograd.Variable(data.x.to(DEVICE, dtype=torch.float))
                dist = torch.autograd.Variable(data.dist.to(DEVICE, dtype=torch.float))
                adj = torch.autograd.Variable(data.adj.to(DEVICE, dtype=torch.float))
                pos = data.POS[0]
                length = data.length.item()
                pred = model(feature, dist, adj)
                pred = pred.cpu().numpy().tolist()
                predict_protein = [0] * length
                for k, i in enumerate(pos):
                    predict_protein[i] = pred[k]
                all_pred.extend(predict_protein)
        result = np.where(np.array(all_pred) > 0.27, 1, 0).tolist()
        protein = args.protein

        i = 0
        f = open(output_path + 'predict_' + protein + '.txt', 'w')
        f.write('Protein: ' + protein + '\n')
        f.write('Number ' + '  Chain' + '  Amino_Acid ' + ' Predict ' + '    Score' + '\n')
        for c in chains:
            parser = PDBParser(QUIET=1)
            structure = parser.get_structure(protein, sys.path[0] + '/Residue_PDB/' + protein + '.pdb')
            residue_id_list = []
            for aa in structure[0][c.upper()].get_residues():
                residue_id_list.append(aa.get_id()[1])
            for k, p in enumerate(fasta_dict[(protein + c).lower()]):
                f.write('   ' + "{:>3d}".format(residue_id_list[k]) + '     ' + c + '        ' + p + '         ' + str(result[i])
                        + '         ' + "{:.3f}".format(all_pred[i]) + '\n')
                i += 1
        f.close()

    logo()
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ',DEVICE)
    best_model = Spatom().to(DEVICE)
    best_model.load_state_dict(torch.load(sys.path[0]+'/model/best_model.dat'))
    test_set = feature_extract(protein,chains, PSSM)
    fasta_dict, _ = seq_and_one_hot(protein, chains)
    test(best_model,test_set,output_path)
    print('Done!')
    return 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protein', '-p', help='protein name', type = str)
    parser.add_argument('--PSSM', '-m', help='path_to_PSSM_folder, save the PSSM files of corresponding chains into this folder', type=str)
    parser.add_argument('--pdb', '-b', help='path_to_pdb/pdb_file', type=str)
    parser.add_argument('--output', '-o', help='path_to_output',default=sys.path[0]+'/result/', type=str)
    args = parser.parse_args()

    if args.PSSM:
        fileList = os.listdir(args.PSSM)
        for file in fileList:
            os.rename(args.PSSM + '/' + file, args.PSSM + '/' + file.upper())

    if args.protein == None or args.PSSM == None:
        print('Please check your input!')
        sys.exit('error!')
    else:

        if args.pdb == None:
            print('Down load PDB for input protein name!')
            if not os.path.exists(sys.path[0]+'/real_PDB'):
                os.makedirs(sys.path[0]+'/real_PDB')
            pdb = PDBList()
            pdb.retrieve_pdb_file(args.protein, pdir=sys.path[0]+'/real_PDB', file_format='pdb')
            PDB_proressing(sys.path[0]+'/real_PDB/pdb'+args.protein.lower()+'.ent', args.protein)
            pre_pdb = PDBParser(QUIET=1).get_structure(args.protein, sys.path[0]+"/Residue_PDB/"+args.protein + '.pdb').get_chains()
            chains = []
            for _ in pre_pdb:
                chain = _.get_id()
                chains.append(chain)
                if not os.path.isfile(args.PSSM + '/' + args.protein.upper() + '_' + chain.upper() + '.PSSM'):
                    print('Please check the pssm file!')
                    sys.exit('error!')
            predict(args.protein,chains,sys.path[0]+'/real_PDB/pdb'+args.protein.lower()+'.ent',args.output,args.PSSM)
        else:
            print('Using given PDB files!')
            PDB_proressing(args.pdb, args.protein)
            pre_pdb = PDBParser(QUIET=1).get_structure(args.protein, sys.path[0]+"/Residue_PDB/"+args.protein + '.pdb').get_chains()
            chains = []
            for _ in pre_pdb:
                chain = _.get_id()
                chains.append(chain)
                print(chain)
                if not os.path.isfile(args.PSSM + '/' + args.protein.upper() + '_' + chain.upper() + '.PSSM'):
                    print('Please check the pssm file!')
                    sys.exit('error!')
            predict(args.protein,chains,args.pdb,args.output,args.PSSM)
