from pegasus import calc_kBET
import pickle 
import anndata as ad

def calculate_kbet_from_path(emb_path, data_path):
    f = open(emb_path,'rb')
    emb = pickle.load(f)
    adata = ad.read_h5ad(data_path)
    adata.obsm['X_latent']=emb
    kbet = calc_kBET(adata, 'batch_indices',
                 rep='latent',
                 K=25,
                 alpha=0.05)
    print('kBET acceptance rate: ', kbet[2])
    return kbet
  
def calculate_kbet_from_adata(adata):
  '''adata object needs to have .obsm['X_latent'] and .obs['batch_indices']'''
    kbet = calc_kBET(adata, 'batch_indices',
                 rep='latent',
                 K=25,
                 alpha=0.05)
    print('kBET acceptance rate: ', kbet[2])
    return kbet
