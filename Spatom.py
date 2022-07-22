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

def PSSM_file_to_PSSM_dict(protein,chain,PSSM):
    Max_pssm = np.array(
        [7., 9., 9., 9., 12., 9., 8., 8., 12., 8., 7., 9., 11., 10., 9., 8., 8., 13., 10., 8.])
    Min_pssm = np.array(
        [-10., - 12., - 12., - 12., - 11., - 11., - 12., - 12., - 12., - 12., - 12., - 12., - 11., - 11.,
         - 12., - 11., - 10., - 12., - 11., - 11.])
    PSSM_dict = {}
    PSSM_file = open(PSSM,'r')
    PSSM_matrix = []
    for line in PSSM_file:
        if line != None and len(line.split()) > 40:
            PSSM_line = line.split()[2:22]
            PSSM_line = list(map(float,PSSM_line))
            PSSM_line = ((np.array(PSSM_line) - Min_pssm)/(Max_pssm - Min_pssm)).tolist()
            PSSM_matrix.append(PSSM_line)
    key = str(PSSM).split('/')[-1].replace('.pssm','')
    key = key.split('_')[0] + key.split('_')[1]
    PSSM_dict[key.lower()] = PSSM_matrix
    protein_chain = protein + chain
    if protein_chain.lower() not in PSSM_dict.keys():
        print('The PSSM file name is inconsistent with command line arguments!')
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

def Dist_adj(protein,chain):
    def dictance(xyz, position):
        xyz = xyz - xyz[position]
        dictance = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2 + xyz[:, 2] ** 2).tolist()
        return dictance

    p = PDBParser(QUIET=1)
    chain = chain.upper()
    structure = p.get_structure(protein, sys.path[0]+"/CA_PDB/" + protein + '.pdb')

    distance_list = []
    chain_list = []
    for residue in structure[0][chain]:
        for atom in residue:
            chain_list.append(atom.get_vector().get_array())
    for i, center in enumerate(chain_list):
        distance_list.append(dictance(chain_list, i))
    distance_dict= {}
    distance_dict[(protein+chain).lower()] = np.array(distance_list)
    return distance_dict

def seq_and_one_hot(protein,chain):
    AA_index = {j: i for i, j in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    seq_dict = {}
    one_hot_dict = {}
    parser = PDBParser(QUIET=1)
    structure = parser.get_structure(protein, sys.path[0] + '/Residue_PDB/' + protein + ".pdb")
    ppb = PDB.PPBuilder()

    model = structure[0][chain.upper()]
    seq = ''
    for c in ppb.build_peptides(model):
        seq += c.get_sequence()
    seq_dict[(protein + chain).lower()] = seq
    one_hot_list = []
    for i in seq:
        zero = [0] * 20
        zero[AA_index[i]] = 1
        one_hot_list.append(zero)
    one_hot_dict[(protein + chain).lower()] = one_hot_list
    return seq_dict, one_hot_dict

class ChianSelect(Select):
    def __init__(self,chain_letter):
        self.chain_letter = chain_letter
    def accept_chain(self,chain):
        if chain.get_id()==self.chain_letter:
            return True
        else:
            return False

def get_RSA(protein,chain):
    RSA_dict = {}
    structure = freesasa.Structure(sys.path[0]+"/Residue_PDB/"+protein+'_'+chain.upper() +'.pdb')
    result = freesasa.calc(structure,freesasa.Parameters({'algorithm' : freesasa.LeeRichards,'n-slices' : 100,'probe-radius' : 1.4}))
    residueAreas = result.residueAreas()
    RSA = []
    for r in residueAreas[chain.upper()].keys():
        RSA_AA = []
        RSA_AA.append(min(1,residueAreas[chain.upper()][r].relativeTotal))
        RSA_AA.append(min(1,residueAreas[chain.upper()][r].relativePolar))
        RSA_AA.append(min(1,residueAreas[chain.upper()][r].relativeApolar))
        RSA_AA.append(min(1,residueAreas[chain.upper()][r].relativeMainChain))
        if math.isnan(residueAreas[chain.upper()][r].relativeSideChain):
            RSA_AA.append(0)
        else:
            RSA_AA.append(min(1,residueAreas[chain.upper()][r].relativeSideChain))
        RSA.append(RSA_AA)
    RSA_dict[(protein+chain).lower()] = RSA
    return RSA_dict

def CX_DPX(protein,chain,path):
    CX_DPX_dict = {}
    CX_DPX_protein = []
    f = open(path,'r')
    for line in f:
        if line.split() != []:
            if line.split()[0] == chain.upper():
                AA_CX_DPX = list(map(float,line.split()[3:]))
                CX_DPX_protein.append(AA_CX_DPX)
    min = np.min(np.array(CX_DPX_protein),axis=0)
    max = np.max(np.array(CX_DPX_protein),axis=0)
    max = np.where(max==0,1,max)
    CX_DPX_protein = ((np.array(CX_DPX_protein) - min)/max).tolist()
    CX_DPX_dict[(protein+chain).lower()] = CX_DPX_protein
    return CX_DPX_dict

def dssp_feature(protein,chain,ref_seq):
    SS_dict = {'H': 0, 'B': 1, 'E': 2, 'G': 3, 'I': 4, 'T': 5, 'S': 6, '-': 7}
    p = PDBParser(QUIET=1)
    pdb = p.get_structure(protein, sys.path[0]+'/Residue_PDB/' + protein + '.pdb')
    io = PDBIO()
    io.set_structure(pdb)
    io.save(sys.path[0]+'/Residue_PDB/'+protein + '_' + chain.upper() + '.pdb', ChianSelect(chain.upper()))
    structure = p.get_structure(protein,sys.path[0]+'/Residue_PDB/'+ protein + '_' + chain.upper() + '.pdb')
    model = structure[0]
    dssp = DSSP(model, sys.path[0]+'/Residue_PDB/' + protein + '_' + chain.upper()+'.pdb',dssp='mkdssp')

    key_list = []
    for i in dssp.keys():
        if i[0] == args.chain.upper():
            key_list.append(i)
    dssp_matrix = []
    seq = ""
    for i in key_list:
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
    dssp_dict = {}
    dssp_dict[(protein + chain).lower()] = pad_dssp_matrix
    return dssp_dict

def AA_property(protein,chain,seq_dict):
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
    for AA in seq_dict[(protein+chain).lower()]:
        AA_AA = []
        AA_AA.append(Side_Chain_Atom_num[AA])
        AA_AA.append(Side_Chain_Charge_num[AA])
        AA_AA.append(Side_Chain_hydrogen_bond_num[AA])
        AA_AA.append(Side_Chain_pKa[AA])
        AA_AA.append(Hydrophobicity[AA])
        AA_protein.append(AA_AA)
    AA_property_dict[(protein+chain).lower()] = AA_protein
    return AA_property_dict

def edge_weight(dist):
    matrix = dist.clone()
    softmax = torch.nn.Softmax(dim=0)
    dist = softmax(1./(torch.log(torch.log(dist+2))))
    dist[matrix>14] = 0
    return dist

def feature_extract(protein,chain,PSSM_file,CX_DPX_file):
    PSSM_dict = PSSM_file_to_PSSM_dict(protein, chain, PSSM_file)
    Dist_dict = Dist_adj(protein, chain)
    fasta_dict, onehot_dict = seq_and_one_hot(protein, chain)
    DSSP_dict = dssp_feature(protein, chain, fasta_dict[(protein + chain).lower()])
    RSA_dict = get_RSA(protein, chain)
    os.remove(sys.path[0] + '/Residue_PDB/' + protein + '_' + chain.upper() + '.pdb')
    AA_property_dict = AA_property(protein, chain, fasta_dict)
    CX_DPX_dict = CX_DPX(protein, chain, CX_DPX_file)
    Datasets = []
    feature = []
    RSA = []
    for k in range(len(onehot_dict[(protein + chain).lower()])):
        AA = []
        AA.extend(onehot_dict[(protein + chain).lower()][k])
        AA.extend(PSSM_dict[(protein + chain).lower()][k])
        AA.extend(RSA_dict[(protein + chain).lower()][k])
        AA.extend(AA_property_dict[(protein + chain).lower()][k])
        AA.extend(DSSP_dict[(protein + chain).lower()][k])
        AA.extend(np.array(CX_DPX_dict[(protein + chain).lower()][k])[[0, 2, 6, 8]].tolist())
        feature.append(AA)
        RSA.append(RSA_dict[(protein + chain).lower()][k][0])
    pos = np.where(np.array(RSA) >= 0.05)[0].tolist()
    Dist = edge_weight(torch.tensor(Dist_dict[(protein + chain).lower()]))[pos, :][:, pos]
    feature = torch.tensor(np.array(feature)[pos, :], dtype=torch.float)
    adj = torch.tensor(np.where(np.array(Dist_dict[(protein + chain).lower()]) < 14, 1, 0)[pos, :][:, pos])
    data = Data(x=feature)
    data.dist = Dist
    data.POS = pos
    length = len(fasta_dict[(protein + chain).lower()])
    data.length = length
    data.adj = adj
    Datasets.append(data)
    return Datasets

def predict(protein,chain,pdb_file,output_path,PSSM,CX_DPX_file):
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
        result = np.where(np.array(all_pred) > 0.29, 1, 0).tolist()
        protein = args.protein
        chain = args.chain
        f = open(output_path + 'predict_' + protein + '_' + chain + '.txt', 'w')
        f.write('Protein: ' + protein + '\n')
        f.write('Chain: ' + chain + '\n')
        f.write('Number ' + '  Amino_Acid ' + ' Predict ' + '     Score' + '\n')
        for k, p in enumerate(result):
            f.write('   ' + "{:>3d}".format(k + 1) + '        ' + fasta_dict[(protein + chain).lower()][
                k] + '         ' + str(p) + '         ' + "{:.3f}".format(all_pred[k]) + '\n')
        f.close()

    logo()
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    PDB_proressing(pdb_file,protein)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ',DEVICE)
    best_model = Spatom().to(DEVICE)
    best_model.load_state_dict(torch.load(sys.path[0]+'/model/best_model.dat'))
    test_set = feature_extract(protein, chain, PSSM, CX_DPX_file)
    fasta_dict, _ = seq_and_one_hot(protein, chain)
    test(best_model,test_set,output_path)
    print('Done!')
    return 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protein', '-p', help='protein name', type = str)
    parser.add_argument('--chain', '-c', help='chain name', type = str)
    parser.add_argument('--PSSM', '-m', help='path_to_PSSM/PSSM_file', type=str)
    parser.add_argument('--pdb', '-b', help='path_to_pdb/pdb_file', type=str)
    parser.add_argument('--CX_DPX', '-d', help='path_to_CX_DPX/CX_DPX_file', type=str)
    parser.add_argument('--output', '-o', help='path_to_output',default=sys.path[0]+'/result/', type=str)
    args = parser.parse_args()

    if args.protein == None or args.chain == None or args.PSSM == None:
        print('Please check your input!')
    else:
        if args.pdb == None:
            print('Down load PDB for input protein name!')
            if not os.path.exists(sys.path[0]+'/real_PDB'):
                os.makedirs(sys.path[0]+'/real_PDB')
            pdb = PDBList()
            pdb.retrieve_pdb_file(args.protein, pdir=sys.path[0]+'/real_PDB', file_format='pdb')
            predict(args.protein,args.chain,sys.path[0]+'/real_PDB/pdb'+args.protein+'.ent',args.output,args.PSSM,args.CX_DPX)
        else:
            print('Using given PDB files!')
            predict(args.protein,args.chain,args.pdb,args.output,args.PSSM,args.CX_DPX)
