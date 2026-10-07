#!/usr/bin/env python3

"""Configurable cheminformatics similarity -> clustering -> profiling -> diversity selection pipeline."""

from __future__ import annotations
import argparse, json, logging, math, os, random, sys, time
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors, QED, MACCSkeys, Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import rdFingerprintGenerator
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

RDLogger.DisableLog('rdApp.warning')
LOG = logging.getLogger('chemdiv')

DEFAULTS = {
 'io': {'query':'query.smi','parent':'parent.smi','outdir':'chemdiv_out','smiles_column':'SMILES','id_column':'ID','delimiter':None,'chunksize':50000},
 'standardize': {'cleanup':True,'largest_fragment':True,'uncharger':True,'canonical_tautomer':False,'remove_stereo':False},
 'similarity': {'fingerprint':'morgan','fp_mode':'bit','n_bits':2048,'radius':3,'min_path':1,'max_path':7,'use_chirality':False,'metric':'tanimoto','threshold':0.60,'tversky_alpha':0.9,'tversky_beta':0.1,'aggregate':'max','top_n_per_query':0},
 'clustering': {'enabled':True,'sample_size':200000,'n_clusters':50000,'batch_size':4096,'max_iter_stage1':100,'max_iter_stage2':100,'n_init_stage1':3,'random_state':42,'representative':'nearest_centroid'},
 'profiling': {'enabled':True,'generate_3d':False,'max_3d_molecules':50000,'num_confs':10,'prune_rms_thresh':0.5,'optimize':'MMFF','pmi_sample':50000},
 'selection': {'enabled':True,'n_select':1000,'qed_min':0.35,'qed_weight':0.45,'diversity_weight':0.35,'scaffold_weight':0.20,'max_per_scaffold':1,'pool_size':10000,'selection_fp':'morgan','selection_radius':3,'selection_bits':2048},
 'plots': {'enabled':True,'dpi':250,'max_points':50000,'umap':True,'tsne':False,'top_scaffolds':30}
}

def deep_update(a,b):
    for k,v in b.items():
        if isinstance(v,dict) and isinstance(a.get(k),dict): deep_update(a[k],v)
        else: a[k]=v
    return a

def load_cfg(path):
    cfg=json.loads(json.dumps(DEFAULTS))
    if path:
        with open(path) as f: deep_update(cfg,json.load(f))
    return cfg

def read_table(path, smiles_col='SMILES', id_col='ID', delimiter=None):
    p=Path(path); sep=delimiter
    if p.suffix.lower() in ['.smi','.smiles','.txt']:
        if sep is None: sep=r'\s+'
        df=pd.read_csv(p,sep=sep,engine='python',comment='#',header=None)
        if df.shape[1]==1: df.columns=[smiles_col]; df[id_col]=[f'{p.stem}_{i}' for i in range(len(df))]
        else:
            cols=[smiles_col,id_col]+[f'extra_{i}' for i in range(df.shape[1]-2)]; df.columns=cols
    else:
        if sep is None: sep='\t' if p.suffix.lower() in ['.tsv','.tab'] else ','
        df=pd.read_csv(p,sep=sep)
        if id_col not in df: df[id_col]=[f'{p.stem}_{i}' for i in range(len(df))]
    return df

def standardize_mol(smi,cfg):
    try:
        mol=Chem.MolFromSmiles(str(smi),sanitize=True)
        if mol is None: return None,None
        if cfg['cleanup']: mol=rdMolStandardize.Cleanup(mol)
        if cfg['largest_fragment']: mol=rdMolStandardize.LargestFragmentChooser().choose(mol)
        if cfg['uncharger']: mol=rdMolStandardize.Uncharger().uncharge(mol)
        if cfg['canonical_tautomer']: mol=rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
        if cfg['remove_stereo']: Chem.RemoveStereochemistry(mol)
        Chem.SanitizeMol(mol)
        return mol,Chem.MolToSmiles(mol,canonical=True,isomericSmiles=not cfg['remove_stereo'])
    except Exception: return None,None

def standardize_df(df, smi_col, cfg):
    mols=[]; std=[]; ok=[]
    for s in df[smi_col].astype(str):
        m,c=standardize_mol(s,cfg); mols.append(m); std.append(c); ok.append(m is not None)
    out=df.copy(); out['Original_SMILES']=out[smi_col].astype(str); out['SMILES_std']=std; out['_mol']=mols
    bad=out.loc[[not x for x in ok]].drop(columns=['_mol']); out=out.loc[ok].reset_index(drop=True)
    out=out.drop_duplicates('SMILES_std').reset_index(drop=True)
    return out,bad

def fp_generator(name,cfg):
    n=int(cfg['n_bits']); r=int(cfg['radius']); ch=bool(cfg['use_chirality'])
    if name=='morgan': return rdFingerprintGenerator.GetMorganGenerator(radius=r,fpSize=n,includeChirality=ch)
    if name in ('featmorgan','fcfp'): return rdFingerprintGenerator.GetMorganGenerator(radius=r,fpSize=n,includeChirality=ch,atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
    if name=='rdkit': return rdFingerprintGenerator.GetRDKitFPGenerator(minPath=int(cfg['min_path']),maxPath=int(cfg['max_path']),fpSize=n)
    if name=='atompair': return rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n,includeChirality=ch)
    if name in ('torsion','topologicaltorsion'): return rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=n,includeChirality=ch)
    return None

def make_fp(mol,cfg):
    name=cfg['fingerprint'].lower(); mode=cfg['fp_mode'].lower(); n=int(cfg['n_bits'])
    gen=fp_generator(name,cfg)
    if gen:
        if mode=='count': return gen.GetCountFingerprint(mol)
        if mode=='sparse_count': return gen.GetSparseCountFingerprint(mol)
        return gen.GetFingerprint(mol)
    if name=='maccs': return MACCSkeys.GenMACCSKeys(mol)
    if name=='pattern': return Chem.PatternFingerprint(mol,fpSize=n)
    if name=='layered': return Chem.LayeredFingerprint(mol,fpSize=n,minPath=int(cfg['min_path']),maxPath=int(cfg['max_path']))
    raise ValueError(f'Unknown fingerprint {name}')

def similarity(q,p,cfg):
    m=cfg['metric'].lower()
    if m=='tanimoto': return DataStructs.TanimotoSimilarity(q,p)
    if m=='tversky': return DataStructs.TverskySimilarity(q,p,float(cfg['tversky_alpha']),float(cfg['tversky_beta']))
    if m=='dice': return DataStructs.DiceSimilarity(q,p)
    if m=='cosine': return DataStructs.CosineSimilarity(q,p)
    raise ValueError(m)

def similarity_search(qdf,pdf,cfg,idcol):
    LOG.info('Fingerprinting %d query and %d parent molecules',len(qdf),len(pdf))
    qfps=[make_fp(m,cfg) for m in qdf['_mol']]; pfps=[make_fp(m,cfg) for m in pdf['_mol']]
    rows=[]; keep=np.zeros(len(pdf),dtype=bool); best=np.full(len(pdf),-1.0); bestq=np.full(len(pdf),-1,dtype=int)
    threshold=float(cfg['threshold'])
    for qi,qfp in enumerate(qfps):
        for j,pfp in enumerate(pfps):
            s=similarity(qfp,pfp,cfg)
            if s>best[j]: best[j]=s; bestq[j]=qi
    keep=best>=threshold
    hits=pdf.loc[keep].copy(); idx=np.where(keep)[0]
    hits['Similarity']=best[idx]; hits['BestQueryIndex']=bestq[idx]
    hits['BestQueryID']=[qdf.iloc[i][idcol] for i in bestq[idx]]
    hits['BestQuerySMILES']=[qdf.iloc[i]['SMILES_std'] for i in bestq[idx]]
    return hits.sort_values('Similarity',ascending=False).reset_index(drop=True)

def fp_numpy(mols,cfg):
    X=np.zeros((len(mols),int(cfg['n_bits'])),dtype=np.uint8)
    for i,m in enumerate(mols):
        fp=make_fp(m,cfg); arr=np.zeros((int(cfg['n_bits']),),dtype=np.int32); DataStructs.ConvertToNumpyArray(fp,arr); X[i]=arr>0
    return X

def two_stage_cluster(df,cfg,fp_cfg):
    n=len(df); k=min(int(cfg['n_clusters']),n); rng=np.random.default_rng(int(cfg['random_state']))
    X=fp_numpy(df['_mol'].tolist(),fp_cfg).astype(np.float32)
    ns=min(int(cfg['sample_size']),n); si=rng.choice(n,ns,replace=False); Xs=X[si]
    LOG.info('Stage 1 MiniBatchKMeans: sample=%d clusters=%d',ns,k)
    km1=MiniBatchKMeans(n_clusters=k,batch_size=int(cfg['batch_size']),max_iter=int(cfg['max_iter_stage1']),n_init=int(cfg['n_init_stage1']),random_state=int(cfg['random_state']),reassignment_ratio=0.01)
    km1.fit(Xs)
    LOG.info('Stage 2 MiniBatchKMeans over %d molecules initialized from stage-1 centroids',n)
    km2=MiniBatchKMeans(n_clusters=k,init=km1.cluster_centers_,n_init=1,batch_size=int(cfg['batch_size']),max_iter=int(cfg['max_iter_stage2']),random_state=int(cfg['random_state']),reassignment_ratio=0.0)
    labels=km2.fit_predict(X); df=df.copy(); df['Cluster']=labels
    centers=km2.cluster_centers_; reps=[]
    for c in range(k):
        inds=np.where(labels==c)[0]
        if not len(inds): continue
        # squared Euclidean distance to binary-FP centroid
        d=((X[inds]-centers[c])**2).sum(axis=1); reps.append(inds[int(np.argmin(d))])
    reps_df=df.iloc[reps].copy().reset_index(drop=True); reps_df['ClusterSize']=df.groupby('Cluster').size().reindex(reps_df['Cluster']).values
    return df,reps_df

def descriptors(m):
    scaf=MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=True)
    return {'MW':Descriptors.MolWt(m),'HeavyAtoms':m.GetNumHeavyAtoms(),'cLogP':Crippen.MolLogP(m),'TPSA':rdMolDescriptors.CalcTPSA(m),'HBD':Lipinski.NumHDonors(m),'HBA':Lipinski.NumHAcceptors(m),'RotB':Lipinski.NumRotatableBonds(m),'Rings':rdMolDescriptors.CalcNumRings(m),'AromaticRings':rdMolDescriptors.CalcNumAromaticRings(m),'FractionCSP3':rdMolDescriptors.CalcFractionCSP3(m),'FormalCharge':Chem.GetFormalCharge(m),'QED':QED.qed(m),'BertzCT':Descriptors.BertzCT(m),'MolMR':Crippen.MolMR(m),'Scaffold':scaf or '[acyclic]'}

def add_descriptors(df):
    vals=pd.DataFrame([descriptors(m) for m in df['_mol']]); return pd.concat([df.reset_index(drop=True),vals],axis=1)

def add_3d_shape(df,cfg):
    cols=['PMI1','PMI2','PMI3','NPR1','NPR2','RadiusGyration','Asphericity','Eccentricity','InertialShapeFactor','SpherocityIndex']
    for c in cols: df[c]=np.nan
    lim=min(len(df),int(cfg['max_3d_molecules']))
    for i in range(lim):
        m=Chem.AddHs(Chem.Mol(df.iloc[i]['_mol']))
        try:
            ps=AllChem.ETKDGv3(); ps.randomSeed=0xf00d; ps.pruneRmsThresh=float(cfg['prune_rms_thresh'])
            ids=AllChem.EmbedMultipleConfs(m,numConfs=int(cfg['num_confs']),params=ps)
            if not ids: continue
            energies=[]
            for cid in ids:
                try:
                    if cfg['optimize'].upper()=='MMFF' and AllChem.MMFFHasAllMoleculeParams(m):
                        props=AllChem.MMFFGetMoleculeProperties(m); ff=AllChem.MMFFGetMoleculeForceField(m,props,confId=cid)
                    else: ff=AllChem.UFFGetMoleculeForceField(m,confId=cid)
                    ff.Minimize(maxIts=300); energies.append((ff.CalcEnergy(),cid))
                except Exception: pass
            cid=min(energies)[1] if energies else ids[0]
            vals=[rdMolDescriptors.CalcPMI1(m,cid),rdMolDescriptors.CalcPMI2(m,cid),rdMolDescriptors.CalcPMI3(m,cid),rdMolDescriptors.CalcNPR1(m,cid),rdMolDescriptors.CalcNPR2(m,cid),rdMolDescriptors.CalcRadiusOfGyration(m,cid),rdMolDescriptors.CalcAsphericity(m,cid),rdMolDescriptors.CalcEccentricity(m,cid),rdMolDescriptors.CalcInertialShapeFactor(m,cid),rdMolDescriptors.CalcSpherocityIndex(m,cid)]
            df.loc[df.index[i],cols]=vals
        except Exception: continue
    return df

def greedy_select(df,cfg):
    pool=df[df['QED']>=float(cfg['qed_min'])].copy()
    if len(pool)==0: return pool
    sf=pool['Scaffold'].value_counts(); pool['ScaffoldFreq']=pool['Scaffold'].map(sf)
    pool['ScaffoldRarity']=1.0/pool['ScaffoldFreq']
    pool=pool.sort_values(['QED','ScaffoldRarity'],ascending=False).head(int(cfg['pool_size'])).reset_index(drop=True)
    fpcfg={'fingerprint':cfg['selection_fp'],'fp_mode':'bit','n_bits':cfg['selection_bits'],'radius':cfg['selection_radius'],'min_path':1,'max_path':7,'use_chirality':False}
    fps=[make_fp(m,fpcfg) for m in pool['_mol']]; selected=[]; scount={}; maxsc=int(cfg['max_per_scaffold']); nsel=min(int(cfg['n_select']),len(pool))
    q=(pool['QED'].to_numpy()-pool['QED'].min())/(pool['QED'].max()-pool['QED'].min()+1e-12); rarity=pool['ScaffoldRarity'].to_numpy(); rarity=(rarity-rarity.min())/(rarity.max()-rarity.min()+1e-12)
    for _ in range(nsel):
        best_i=None; best_score=-1e9
        for i in range(len(pool)):
            if i in selected: continue
            sc=pool.iloc[i]['Scaffold']
            if scount.get(sc,0)>=maxsc: continue
            div=1.0 if not selected else 1.0-max(DataStructs.TanimotoSimilarity(fps[i],fps[j]) for j in selected)
            score=float(cfg['qed_weight'])*q[i]+float(cfg['diversity_weight'])*div+float(cfg['scaffold_weight'])*rarity[i]
            if score>best_score: best_score=score; best_i=i
        if best_i is None: break
        selected.append(best_i); sc=pool.iloc[best_i]['Scaffold']; scount[sc]=scount.get(sc,0)+1
    out=pool.iloc[selected].copy(); out['SelectionRank']=np.arange(1,len(out)+1); return out

def save_plots(df,out,cfg):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); dpi=int(cfg['dpi']); maxp=min(len(df),int(cfg['max_points'])); d=df.sample(maxp,random_state=42) if len(df)>maxp else df
    numeric=['MW','cLogP','TPSA','HBD','HBA','RotB','Rings','AromaticRings','FractionCSP3','QED','BertzCT']
    for c in numeric:
        if c not in d: continue
        plt.figure(figsize=(6,4)); plt.hist(d[c].dropna(),bins=50); plt.xlabel(c); plt.ylabel('Count'); plt.tight_layout(); plt.savefig(out/f'hist_{c}.png',dpi=dpi); plt.close()
    cols=[c for c in numeric if c in d]; corr=d[cols].corr(); plt.figure(figsize=(9,8)); plt.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm'); plt.xticks(range(len(cols)),cols,rotation=90); plt.yticks(range(len(cols)),cols); plt.colorbar(label='Pearson r'); plt.tight_layout(); plt.savefig(out/'descriptor_correlation.png',dpi=dpi); plt.close()
    sc=df['Scaffold'].value_counts().head(int(cfg['top_scaffolds'])); plt.figure(figsize=(8,6)); plt.barh(range(len(sc)),sc.values); plt.yticks(range(len(sc)),[str(x)[:45] for x in sc.index]); plt.xlabel('Frequency'); plt.tight_layout(); plt.savefig(out/'top_scaffolds.png',dpi=dpi); plt.close()
    if {'NPR1','NPR2'}.issubset(df.columns) and df['NPR1'].notna().any():
        z=df.dropna(subset=['NPR1','NPR2']); plt.figure(figsize=(6,6)); plt.scatter(z['NPR1'],z['NPR2'],s=5,alpha=.35); plt.xlabel('NPR1 = PMI1/PMI3'); plt.ylabel('NPR2 = PMI2/PMI3'); plt.xlim(0,1); plt.ylim(0,1); plt.tight_layout(); plt.savefig(out/'pmi_shape_triangle.png',dpi=dpi); plt.close()
    # PCA of standardized physicochemical space
    Z=d[cols].dropna();
    if len(Z)>2:
        P=PCA(n_components=2).fit_transform(StandardScaler().fit_transform(Z)); plt.figure(figsize=(6,5)); plt.scatter(P[:,0],P[:,1],s=5,alpha=.35); plt.xlabel('PC1'); plt.ylabel('PC2'); plt.tight_layout(); plt.savefig(out/'physchem_PCA.png',dpi=dpi); plt.close()

def clean_save(df,path):
    df.drop(columns=['_mol'],errors='ignore').to_csv(path,index=False)

def main():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--config'); ap.add_argument('--query'); ap.add_argument('--parent'); ap.add_argument('--outdir'); ap.add_argument('--write-default-config',action='store_true')
    args=ap.parse_args(); cfg=load_cfg(args.config)
    if args.write_default_config:
        print(json.dumps(DEFAULTS,indent=2)); return
    for k in ['query','parent','outdir']:
        v=getattr(args,k,None)
        if v: cfg['io'][k]=v
    out=Path(cfg['io']['outdir']); out.mkdir(parents=True,exist_ok=True); (out/'plots').mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(out/'pipeline.log')])
    with open(out/'config_used.json','w') as f: json.dump(cfg,f,indent=2)
    smi=cfg['io']['smiles_column']; idc=cfg['io']['id_column']
    q=read_table(cfg['io']['query'],smi,idc,cfg['io']['delimiter']); p=read_table(cfg['io']['parent'],smi,idc,cfg['io']['delimiter'])
    q,qbad=standardize_df(q,smi,cfg['standardize']); p,pbad=standardize_df(p,smi,cfg['standardize'])
    clean_save(q,out/'query_standardized.csv'); clean_save(pbad,out/'parent_invalid.csv'); clean_save(qbad,out/'query_invalid.csv')
    hits=similarity_search(q,p,cfg['similarity'],idc); clean_save(hits,out/'similarity_hits.csv'); LOG.info('Similarity hits: %d',len(hits))
    work=hits
    if cfg['clustering']['enabled'] and len(work):
        assigned,reps=two_stage_cluster(work,cfg['clustering'],cfg['similarity']); clean_save(assigned,out/'cluster_assignments.csv'); clean_save(reps,out/'cluster_representatives.csv'); work=reps
    if cfg['profiling']['enabled'] and len(work):
        work=add_descriptors(work)
        if cfg['profiling']['generate_3d']: work=add_3d_shape(work,cfg['profiling'])
        clean_save(work,out/'library_properties.csv')
        work['Scaffold'].value_counts().rename_axis('Scaffold').reset_index(name='Frequency').to_csv(out/'scaffold_frequencies.csv',index=False)
        work.describe(include='all').to_csv(out/'property_summary.csv')
        if cfg['plots']['enabled']: save_plots(work,out/'plots',cfg['plots'])
    if cfg['selection']['enabled'] and len(work):
        sel=greedy_select(work,cfg['selection']); clean_save(sel,out/'selected_diverse_highQED.csv'); LOG.info('Selected: %d',len(sel))
    LOG.info('Done: %s',out)
if __name__=='__main__': main()
