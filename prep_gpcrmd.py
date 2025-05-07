import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='splits/atlas.csv')
parser.add_argument('--gpcrmd_dir', type=str, required=True)
parser.add_argument('--outdir', type=str, default='./data_atlas')
parser.add_argument('--num_workers', type=int, default=1)
args = parser.parse_args()

import mdtraj as md
import os, tempfile, tqdm
from alphaflow.utils import protein
from openfold.data.data_pipeline import make_protein_features
import pandas as pd 
from multiprocessing import Pool
import numpy as np

os.makedirs(args.outdir, exist_ok=True)

df = pd.read_csv(args.split, index_col='dyn_id')

def main():
    jobs = []
    for dyn in df.index:
        #if os.path.exists(f'{args.outdir}/{dyn}.npz'): continue
        jobs.append(dyn)

    if args.num_workers > 1:
        p = Pool(args.num_workers)
        p.__enter__()
        __map__ = p.imap
    else:
        __map__ = map
    for _ in tqdm.tqdm(__map__(do_job, jobs), total=len(jobs)):
        pass
    if args.num_workers > 1:
        p.__exit__(None, None, None)

def do_job(dyn):

    pdb = md.load(f'{args.gpcrmd_dir}/{dyn}/AAonly/{dyn}_AAonly.pdb') 

    trajs = []
    for xtc in os.listdir(f'{args.gpcrmd_dir}/{dyn}/AAonly/'):
        if xtc.endswith('.xtc'):
            
            traj = md.load(os.path.join(f'{args.gpcrmd_dir}/{dyn}/AAonly/', xtc), top=pdb)
            unitcell_vectors = traj.unitcell_vectors
            trajs.append(traj)
        
    if pdb.n_frames != len(unitcell_vectors):
        unitcell_vectors = np.tile(unitcell_vectors[0], (pdb.n_frames, 1, 1))
        pdb.unitcell_vectors = unitcell_vectors
    
    dyn_traj = pdb + trajs

    f, temp_path = tempfile.mkstemp(); os.close(f)
    positions_stacked = []
    for i in tqdm.trange(0, len(dyn_traj), 100):
        dyn_traj[i].save_pdb(temp_path)
    
        with open(temp_path) as f:
            prot = protein.from_pdb_string(f.read())
            pdb_feats = make_protein_features(prot, dyn)
            positions_stacked.append(pdb_feats['all_atom_positions'])
            
    
    pdb_feats['all_atom_positions'] = np.stack(positions_stacked)
    print({key: pdb_feats[key].shape for key in pdb_feats})
    np.savez(f"{args.outdir}/{dyn}.npz", **pdb_feats)
    os.unlink(temp_path)

main()
