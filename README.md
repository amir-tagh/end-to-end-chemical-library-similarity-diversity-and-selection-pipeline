# end-to-end-chemical-library-similarity-diversity-and-selection-pipeline
ChemDiverse: A scalable cheminformatics pipeline for molecular similarity searching, chemical-space clustering, physicochemical and shape analysis, scaffold characterization, and diversity-aware compound library selection
# ChemDiv pipeline

End-to-end RDKit/scikit-learn workflow:
1. read query + parent SMILES
2. sanitize/standardize/deduplicate
3. fingerprint similarity search (Tanimoto/Tversky; optional Dice/Cosine)
4. two-stage MiniBatchKMeans
5. nearest-centroid cluster representatives
6. physicochemical/QED/scaffold profiling
7. optional ETKDGv3 3D conformers + PMI/NPR shape descriptors
8. scaffold/QED/diversity-aware greedy final selection
9. CSV checkpoints + figures

## Install
```bash
conda create -n chemdiv -c conda-forge python=3.12 rdkit scikit-learn pandas numpy matplotlib -y
conda activate chemdiv
```

## Run
```bash
python chemdiv_pipeline.py --config config.example.json
```
or override paths:
```bash
python chemdiv_pipeline.py --config config.example.json --query queries.smi --parent parent.smi --outdir run01
```

## Fingerprint choices
`morgan`, `featmorgan`/`fcfp`, `rdkit`, `atompair`, `torsion`, `maccs`, `pattern`, `layered`.
For generator fingerprints use `fp_mode`: `bit`, `count`, or `sparse_count` (clustering is binarized in this version).

## Important scaling note
The similarity implementation in this reference script is exact but pairwise. For many queries x millions of parent molecules, use chunking/indexing/parallelism or a database/vector-search backend. The clustering matrix is also dense, so 1M x 2048 float32 is ~8.2 GB before estimator overhead. A production version should use packed/sparse fingerprints or dimensionality reduction.

## Outputs
`query_standardized.csv`, `parent_invalid.csv`, `similarity_hits.csv`, `cluster_assignments.csv`, `cluster_representatives.csv`, `library_properties.csv`, `scaffold_frequencies.csv`, `property_summary.csv`, `selected_diverse_highQED.csv`, `plots/*.png`, `config_used.json`, `pipeline.log`.
