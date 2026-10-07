import os,gc,json,math,warnings,io,zipfile
from pathlib import Path
import numpy as np,pandas as pd,torch,torch.nn as nn,torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
import anndata as ad
import requests
from scipy.sparse import issparse
from scipy.stats import pearsonr,spearmanr,fisher_exact
from scipy.spatial.distance import cosine
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score
from collections import defaultdict
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt,matplotlib.gridspec as gridspec,seaborn as sns
warnings.filterwarnings("ignore")

PATH   = Path("/content/drive/MyDrive/K562_essential_normalized_singlecell_01.h5ad")
OUTDIR = Path("dsrp_v19_outputs");OUTDIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS  = [0]

CORUM_PATH  = "/content/drive/MyDrive/niwiad/corum_humanComplexes (1).txt"
ALIAS_PATH  = "/content/drive/MyDrive/niwiad/9606.protein.aliases.v12.0.txt"
STRING_PATH = "/content/drive/MyDrive/niwiad/9606.protein.physical.links.v12.0 2.txt"

N_GENES    = 5000
CTRL_LABEL = "non-targeting"
PERT_COL   = "gene"
N_MIN      = 30

K        = 128
G        = 64
M_AND    = 12
M_XOR    = 6
M_NAND   = 6
K_ACTIVE = 6

STAGE1_END = 30
STAGE2_END = 70
EPOCHS     = 100
BS         = 2048
LR         = 2e-4
WD         = 1e-5
GRAD_CLIP  = 1.0

LAM_RECON   = 1.0
LAM_CONSIST = 0.5
LAM_CORUM   = 0.8
LAM_ENT     = 0.30
LAM_SPA     = 0.20
LAM_DIV     = 0.10
LAM_MD      = 0.15
LAM_UNIQ    = 0.05
LAM_DEC     = 0.05

print(f"Device:{DEVICE} K={K} G={G} AND={M_AND} XOR={M_XOR} NAND={M_NAND}")

print("\n--- Loading CORUM complexes from local file ---")

def fetch_corum(cache=OUTDIR/"corum_complexes.tsv"):
    if cache.exists():
        df = pd.read_csv(cache, sep="\t")
        print(f"  Loaded cached CORUM: {len(df)} complexes")
        return df

    print(f"  Reading: {CORUM_PATH}")
    raw = pd.read_csv(CORUM_PATH, sep="\t")
    rows = []
    for _, row in raw.iterrows():
        genes = [g.strip() for g in str(row["subunits_gene_name"]).split(";")
                 if g.strip() and g.strip() != "nan"]
        name  = str(row["complex_name"])
        if len(genes) >= 2:
            rows.append({"complex_name": name, "genes": ";".join(sorted(genes))})

    df = pd.DataFrame(rows)
    df.to_csv(cache, sep="\t", index=False)
    print(f"  CORUM: {len(df)} complexes parsed and cached")
    return df

df_corum = fetch_corum()

corum_complexes = {}
for i, row in df_corum.iterrows():
    genes = frozenset(g.strip() for g in str(row["genes"]).split(";") if g.strip())
    if len(genes) >= 3:
        corum_complexes[i] = {"name": row["complex_name"], "genes": genes}
print(f"  Complexes with ≥3 subunits: {len(corum_complexes)}")

gene_to_complexes = defaultdict(set)
for cid, info in corum_complexes.items():
    for g in info["genes"]:
        gene_to_complexes[g].add(cid)

print("\n--- Loading STRING interactions from local files ---")

def fetch_string_interactions(genes, score_threshold=700,
                               cache=OUTDIR/"string_interactions.tsv"):
    if cache.exists():
        df = pd.read_csv(cache, sep="\t")
        print(f"  Loaded cached STRING: {len(df)} interactions")
        return df

    print(f"  Building alias map from: {ALIAS_PATH}")
    print("  (This may take ~30s — 3.8M rows)")

    GENE_SYM_SOURCES = {
        "Ensembl_HGNC",
        "Ensembl_HGNC_symbol",
        "BLAST_UniProt_GN_symbol",
        "BLAST_UniProt_GN",
        "Ensembl_UniProt_GN",
    }

    ensp2gene   = {}
    ensp2gene_fb= {}

    chunk_iter = pd.read_csv(
        ALIAS_PATH, sep="\t", header=None,
        names=["string_id", "alias", "source"],
        chunksize=200_000)

    for chunk in chunk_iter:
        for _, row in chunk.iterrows():
            pid = row["string_id"]
            src = str(row["source"])
            alias = str(row["alias"]).strip()
            if not alias or alias == "nan":
                continue
            if src in GENE_SYM_SOURCES:
                if pid not in ensp2gene:
                    ensp2gene[pid] = alias
            elif alias.isupper() and 2 <= len(alias) <= 12 and pid not in ensp2gene:
                ensp2gene_fb[pid] = alias

    for pid, alias in ensp2gene_fb.items():
        if pid not in ensp2gene:
            ensp2gene[pid] = alias

    print(f"  Alias map: {len(ensp2gene):,} proteins mapped to gene symbols")

    print(f"  Loading STRING links from: {STRING_PATH}")
    print("  (This may take ~20s — 1.4M rows)")

    gene_set = set(genes)
    kept_rows = []

    chunk_iter2 = pd.read_csv(
        STRING_PATH, sep=" ",
        chunksize=100_000)

    for chunk in chunk_iter2:
        chunk = chunk[chunk["combined_score"] >= score_threshold]
        if len(chunk) == 0:
            continue
        chunk["gene_a"] = chunk["protein1"].map(ensp2gene)
        chunk["gene_b"] = chunk["protein2"].map(ensp2gene)
        chunk = chunk.dropna(subset=["gene_a", "gene_b"])
        mask = chunk["gene_a"].isin(gene_set) & chunk["gene_b"].isin(gene_set)
        chunk = chunk[mask]
        if len(chunk) > 0:
            kept_rows.append(chunk[["gene_a", "gene_b", "combined_score"]])

    if not kept_rows:
        print("  WARNING: No STRING interactions found among perturbation genes.")
        print("  This may mean gene symbol mapping failed.")
        print(f"  Sample alias map keys: {list(ensp2gene.items())[:5]}")
        df = pd.DataFrame(columns=["gene_a", "gene_b", "combined_score"])
        df.to_csv(cache, sep="\t", index=False)
        return df

    df = pd.concat(kept_rows, ignore_index=True)

    df["pair"] = df.apply(
        lambda r: tuple(sorted([r["gene_a"], r["gene_b"]])), axis=1)
    df = df.drop_duplicates("pair").drop("pair", axis=1).reset_index(drop=True)

    df.to_csv(cache, sep="\t", index=False)
    print(f"  STRING: {len(df)} physical interactions among perturbation genes (cached)")
    return df

print("\n--- Loading expression data ---")
adata = ad.read_h5ad(PATH)
X = (adata.X.toarray() if issparse(adata.X) else adata.X).astype(np.float32)

vc   = adata.obs[PERT_COL].value_counts()
keep = vc[vc >= N_MIN].index
mask = adata.obs[PERT_COL].isin(keep).values
adata = adata[mask].copy(); X = X[mask]

gene_var = X.var(0)
top_idx  = np.argsort(gene_var)[-N_GENES:]
X        = X[:, top_idx]
gene_names = np.array([adata.var["gene_name"].iloc[i] for i in top_idx])

ctrl_mask  = adata.obs[PERT_COL].values == CTRL_LABEL
ctrl_mean  = X[ctrl_mask].mean(0).astype(np.float32)
X_delta    = (X - ctrl_mean).astype(np.float32)

le_p        = LabelEncoder()
pert_labels = le_p.fit_transform(adata.obs[PERT_COL].values).astype(np.int64)
N_CLASSES   = len(le_p.classes_)
pert_names  = le_p.classes_

print(f"  Cells:{X.shape[0]:,} Genes:{N_GENES} Perturbations:{N_CLASSES}")
print(f"  Control cells:{ctrl_mask.sum():,}")

string_df = fetch_string_interactions(
    list(pert_names), cache=OUTDIR/"string_interactions.tsv")

string_scores = {}
for _,row in string_df.iterrows():
    pair = (row["gene_a"], row["gene_b"])
    string_scores[pair] = row["combined_score"]
    string_scores[(row["gene_b"], row["gene_a"])] = row["combined_score"]
print(f"  STRING pairs in lookup: {len(string_scores)//2}")

print("\n--- Computing pseudo-bulk perturbation profiles ---")
pseudo_bulk   = np.zeros((N_CLASSES, N_GENES), dtype=np.float32)
pseudo_bulk_n = np.zeros(N_CLASSES, dtype=int)
for p in range(N_CLASSES):
    msk = pert_labels == p
    if msk.sum() > 0:
        pseudo_bulk[p]   = X_delta[msk].mean(0)
        pseudo_bulk_n[p] = msk.sum()

print(f"  Mean cells per perturbation: {pseudo_bulk_n.mean():.1f}")
print(f"  Pseudo-bulk range: [{pseudo_bulk.min():.3f}, {pseudo_bulk.max():.3f}]")

print("\n--- Building CORUM concept supervision matrix ---")
pert_gene_set = set(pert_names)
valid_complexes = {
    cid: info for cid,info in corum_complexes.items()
    if len(info["genes"] & pert_gene_set) >= 2
}
print(f"  CORUM complexes with ≥2 perturbed genes: {len(valid_complexes)}")

sorted_complexes = sorted(
    valid_complexes.items(),
    key=lambda x: len(x[1]["genes"] & pert_gene_set),
    reverse=True)

N_CORUM_CONCEPTS = min(K // 2, len(sorted_complexes))
anchored_complexes = sorted_complexes[:N_CORUM_CONCEPTS]
print(f"  Anchoring {N_CORUM_CONCEPTS} concepts to CORUM complexes")
for cid,info in anchored_complexes[:5]:
    overlap = info["genes"] & pert_gene_set
    print(f"    {info['name'][:50]}: {len(overlap)} genes ({sorted(overlap)[:4]}...)")

Y_corum = np.zeros((N_CLASSES, N_CORUM_CONCEPTS), dtype=np.float32)
for k, (cid, info) in enumerate(anchored_complexes):
    for pi, pname in enumerate(pert_names):
        if pname in info["genes"]:
            Y_corum[pi, k] = 1.0

n_supervised = (Y_corum.sum(1) > 0).sum()
print(f"  Perturbations with ≥1 complex label: {n_supervised}/{N_CLASSES}")
print(f"  Average complexes per perturbation: {Y_corum.sum(1).mean():.2f}")

print("\n--- Splitting perturbations ---")
np.random.seed(42)
n_test = int(N_CLASSES * 0.20)
n_val  = int(N_CLASSES * 0.10)
perm        = np.random.permutation(N_CLASSES)
test_perts  = set(perm[:n_test])
val_perts   = set(perm[n_test:n_test+n_val])
train_perts = set(perm[n_test+n_val:])

train_cell_mask = np.array([pert_labels[i] in train_perts for i in range(len(pert_labels))])
val_cell_mask   = np.array([pert_labels[i] in val_perts   for i in range(len(pert_labels))])
test_cell_mask  = np.array([pert_labels[i] in test_perts  for i in range(len(pert_labels))])

idx_train = np.where(train_cell_mask)[0]
idx_val   = np.where(val_cell_mask)[0]
idx_test  = np.where(test_cell_mask)[0]

test_pert_list  = sorted(test_perts)
val_pert_list   = sorted(val_perts)
train_pert_list = sorted(train_perts)

print(f"  Train perts:{len(train_perts)} Val:{len(val_perts)} Test:{len(test_perts)}")
print(f"  Train cells:{idx_train.sum():,} Val:{idx_val.sum():,} Test:{idx_test.sum():,}")

X_t      = torch.tensor(X, dtype=torch.float32)
Xd_t     = torch.tensor(X_delta, dtype=torch.float32)
y_t      = torch.tensor(pert_labels, dtype=torch.long)
Ycorum_t = torch.tensor(Y_corum, dtype=torch.float32)
pb_t     = torch.tensor(pseudo_bulk, dtype=torch.float32)
del X, X_delta; gc.collect()

def make_loader(idx, shuffle=True, seed=0):
    ds = TensorDataset(
        X_t[idx],
        Xd_t[idx],
        y_t[idx],
        Ycorum_t[y_t[idx]],
        torch.tensor(idx, dtype=torch.long))
    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(ds, batch_size=BS, shuffle=shuffle,
                      generator=g if shuffle else None,
                      num_workers=2, pin_memory=torch.cuda.is_available(),
                      persistent_workers=True)

def gumbel_binary(logits, tau=1.0, hard=False, eps=1e-10):
    g1 = -torch.log(-torch.log(torch.rand_like(logits)+eps)+eps)
    g2 = -torch.log(-torch.log(torch.rand_like(logits)+eps)+eps)
    y  = torch.sigmoid((logits+g1-g2)/tau)
    if hard: yh=(y>0.5).float(); return yh-y.detach()+y
    return y

class GatedEncoder(nn.Module):
    def __init__(self, P, K):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(P, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU())
        self.B  = nn.Parameter(torch.randn(256,256)*0.01)
        with torch.no_grad(): self.B.fill_diagonal_(0.)
        self.vh = nn.Linear(256, K)
        self.gh = nn.Linear(256, K)
    def forward(self, x_delta):
        h = self.net(x_delta.float())
        h = h + torch.tanh(h @ self.B.T)
        return self.vh(h), self.gh(h)
    def zero_diag(self):
        with torch.no_grad(): self.B.fill_diagonal_(0.)

class SymbolicLayer(nn.Module):
    def __init__(self, in_dim, M_and, M_xor, M_nand, G, tau=0.3, n_init=6, k_active=6):
        super().__init__()
        self.Ma=M_and; self.Mx=M_xor; self.Mn=M_nand
        self.Mt=M_and+M_xor+M_nand; self.G=G; self.k=k_active; self.tau=tau
        def si(s):
            w=torch.zeros(*s)
            for d in range(s[0]):
                for m in range(s[1]):
                    w[d,m,torch.randperm(s[2])[:n_init]]=torch.randn(n_init)*0.5
            return w
        self.wa  = nn.Parameter(si((G, M_and,  in_dim)))
        self.wxa = nn.Parameter(si((G, M_xor,  in_dim)))
        self.wxb = nn.Parameter(si((G, M_xor,  in_dim)))
        self.wn  = nn.Parameter(si((G, M_nand, in_dim)))
        self.lt  = nn.Parameter(torch.zeros(G, self.Mt))
        self.cs  = nn.Parameter(torch.zeros(G, self.Mt))
        self.b   = nn.Parameter(torch.zeros(G))

    def _and(self, c):
        c_=c.unsqueeze(1).unsqueeze(1)
        m_=torch.sigmoid(self.wa.abs()/0.3).unsqueeze(0)
        return torch.sigmoid((m_*(torch.log(
            torch.sigmoid(self.wa.unsqueeze(0)*c_).clamp(1e-6))-math.log(0.5))).sum(-1))

    def _xor(self, c):
        c_=c.unsqueeze(1).unsqueeze(1)
        pa=torch.sigmoid((self.wxa.unsqueeze(0)*c_).sum(-1))
        pb=torch.sigmoid((self.wxb.unsqueeze(0)*c_).sum(-1))
        return pa+pb-2.*pa*pb

    def _nand(self, c):
        c_=c.unsqueeze(1).unsqueeze(1)
        m_=torch.sigmoid(self.wn.abs()/0.3).unsqueeze(0)
        return torch.sigmoid(-(m_*(torch.log(
            torch.sigmoid(self.wn.unsqueeze(0)*c_).clamp(1e-6))-math.log(0.5))).sum(-1))

    def _ste(self):
        h=torch.zeros_like(self.cs)
        h.scatter_(1,self.cs.topk(self.k,-1).indices,1.)
        s=torch.sigmoid(self.cs); return h-s.detach()+s

    def forward(self, c):
        cl    = torch.cat([self._and(c),self._xor(c),self._nand(c)],-1)
        t     = F.softplus(self.lt).unsqueeze(0)
        sharp = cl.pow(1./t.clamp(0.1))
        active= sharp*self._ste().unsqueeze(0)
        wm    = active.mean(-1, keepdim=True)
        return self.tau*torch.logsumexp((active-wm)/self.tau,-1)+self.b

    def spa_loss(self):
        return (torch.relu(self.wa.abs() -0.1).mean()+
                torch.relu(self.wxa.abs()-0.1).mean()+
                torch.relu(self.wxb.abs()-0.1).mean()+
                torch.relu(self.wn.abs() -0.1).mean())

    def uniq_loss(self):
        S=torch.sigmoid(self.cs); Sn=S/(S.sum(1,keepdim=True)+1e-8)
        ov=Sn@Sn.T; mask=1-torch.eye(self.G,device=S.device)
        return (ov*mask).sum()/max(self.G*(self.G-1),1)

    def programs(self, names, thr=0.30):
        wa=self.wa.detach().cpu().numpy()
        wxa=self.wxa.detach().cpu().numpy()
        wxb=self.wxb.detach().cpu().numpy()
        wn=self.wn.detach().cpu().numpy()
        cs=self.cs.detach().cpu().numpy(); K2=wa.shape[-1]//2
        def terms(w):
            act=np.where(np.abs(w)>thr)[0]; seen,out=[],[]
            for i in sorted(act,key=lambda i:-abs(w[i])):
                kc=i if i<K2 else i-K2
                if kc in seen: continue
                seen.append(kc); nm=names[i] if i<len(names) else f"F{i}"
                out.append(nm if w[i]>0 else f"NOT({nm})")
            return out
        progs={}
        for d in range(self.G):
            tidx=np.argsort(-cs[d])[:self.k]; cls=[]
            for idx in tidx:
                if idx<self.Ma:
                    t=terms(wa[d,idx])
                    if t: cls.append({"type":"AND","rule":" AND ".join(t),
                                       "score":float(cs[d,idx]),"n":len(t)})
                elif idx<self.Ma+self.Mx:
                    m=idx-self.Ma; ta=terms(wxa[d,m]); tb=terms(wxb[d,m])
                    if ta or tb:
                        aa="("+" AND ".join(ta)+")" if ta else "∅"
                        bb="("+" AND ".join(tb)+")" if tb else "∅"
                        cls.append({"type":"XOR","rule":f"{aa} XOR {bb}",
                                    "score":float(cs[d,idx]),"n":len(ta)+len(tb)})
                else:
                    m=idx-self.Ma-self.Mx; t=terms(wn[d,m])
                    if t: cls.append({"type":"NAND","rule":"NAND("+",".join(t)+")",
                                       "score":float(cs[d,idx]),"n":len(t)})
            progs[d]=cls
        return progs

class PerturbationDSRP(nn.Module):
    def __init__(self, P, K, G, M_and, M_xor, M_nand, N_corum, k_active=6):
        super().__init__()
        self.K=K; self.G=G; self.N_corum=N_corum
        self.enc = GatedEncoder(P, K)
        self.sym = SymbolicLayer(2*K, M_and, M_xor, M_nand, G, k_active=k_active)
        self.decode_head = nn.Sequential(
            nn.Linear(G, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, P))
        self.corum_head = nn.Linear(K, N_corum)

    def forward(self, x_delta, tau=1.0, hard=False):
        vl, gl    = self.enc(x_delta)
        c_soft    = torch.sigmoid(vl/tau)*torch.sigmoid(gl/tau)
        c_hard    = gumbel_binary(vl,tau,hard)*gumbel_binary(gl,tau,hard)
        c_exp     = torch.cat([c_soft, 1.-c_soft], -1)
        mlog      = self.sym(c_exp); m_soft = torch.sigmoid(mlog)
        x_pred    = self.decode_head(m_soft)
        corum_logits = self.corum_head(c_soft)
        return {"c_soft":c_soft,"c_hard":c_hard,"m_soft":m_soft,
                "x_pred":x_pred,"corum_logits":corum_logits}

    def get_programs(self, concept_names=None):
        if concept_names and len(concept_names)==self.K:
            names = [f"{n}_ON" for n in concept_names]+[f"{n}_OFF" for n in concept_names]
        else:
            names = [f"C{k}_ON" for k in range(self.K)]+[f"C{k}_OFF" for k in range(self.K)]
        return self.sym.programs(names)

def ent_loss(c):
    c=c.clamp(1e-6,1-1e-6); return -(c*c.log()+(1-c)*(1-c).log()).mean()

def div_loss(c):
    cn=F.normalize(c,dim=0); cov=cn.T@cn/c.shape[0]
    return ((cov*(1-torch.eye(c.shape[1],device=c.device))).pow(2)).sum()

def dec_loss(c):
    cc=c-c.mean(0,keepdim=True); cov=cc.T@cc/max(c.shape[0]-1,1)
    mask=1-torch.eye(c.shape[1],device=c.device)
    return (cov*mask).pow(2).sum()/max(c.shape[1]*(c.shape[1]-1),1)

def md_loss(m):
    mn=F.normalize(m,dim=0); cov=mn.T@mn/m.shape[0]
    mask=1-torch.eye(m.shape[1],device=m.device)
    return (cov*mask).pow(2).sum()/max(m.shape[1]*(m.shape[1]-1),1)

def concept_consistency_loss(c_soft, y, min_cells=2):
    ups=y.unique(); wv=torch.tensor(0.,device=c_soft.device); n=0
    for p in ups:
        msk=y==p
        if msk.sum()<min_cells: continue
        wv=wv+c_soft[msk].var(0).mean(); n+=1
    return wv/max(n,1)

def pseudo_bulk_recon_loss(x_pred, y, pb_tensor):
    targets = pb_tensor[y]
    xp = x_pred - x_pred.mean(1, keepdim=True)
    xt = targets - targets.mean(1, keepdim=True)
    num = (xp * xt).sum(1)
    den = (xp.pow(2).sum(1) * xt.pow(2).sum(1)).sqrt().clamp(1e-8)
    r   = num / den
    return (1. - r).mean()

@torch.no_grad()
def collect_concepts(model, loader, tau=0.3, hard=True):
    model.eval(); cs,ms,preds,ys=[],[],[],[]
    for xd_,xd2_,yc_,_,_ in loader:
        xd_ = xd_.to(DEVICE)
        out = model(xd_, tau=tau, hard=hard)
        cs.append(out["c_soft"].cpu())
        ms.append(out["m_soft"].cpu())
        preds.append(out["x_pred"].cpu())
        ys.append(yc_)
    return {"C":torch.cat(cs).numpy(),"M":torch.cat(ms).numpy(),
            "x_pred":torch.cat(preds).numpy(),"y":torch.cat(ys).numpy()}

def pearson_delta(pred_pb, true_pb, eps=1e-8):
    r_list = []
    for i in range(len(pred_pb)):
        p=pred_pb[i]; t=true_pb[i]
        pc=p-p.mean(); tc=t-t.mean()
        denom=math.sqrt(float((pc**2).sum()*(tc**2).sum()))
        if denom > eps: r_list.append(float((pc*tc).sum()/denom))
    return float(np.mean(r_list)) if r_list else 0.

def pearson_delta_top20de(pred_pb, true_pb, ctrl_mean_np, eps=1e-8):
    r_list = []
    for i in range(len(pred_pb)):
        t=true_pb[i]
        de_idx=np.argsort(np.abs(t))[-20:]
        p_de=pred_pb[i][de_idx]; t_de=t[de_idx]
        pc=p_de-p_de.mean(); tc=t_de-t_de.mean()
        denom=math.sqrt(float((pc**2).sum()*(tc**2).sum()))
        if denom > eps: r_list.append(float((pc*tc).sum()/denom))
    return float(np.mean(r_list)) if r_list else 0.

def rank_score(pred_pb, true_pb, eps=1e-8):
    N = len(pred_pb)
    ranks = []
    for i in range(N):
        t = true_pb[i]; t_n = t/(np.linalg.norm(t)+eps)
        sims = []
        for j in range(N):
            p = pred_pb[j]; p_n = p/(np.linalg.norm(p)+eps)
            sims.append(float(np.dot(p_n,t_n)))
        sims = np.array(sims)
        sorted_idx = np.argsort(-sims)
        rank = float(np.where(sorted_idx == i)[0][0]) / N
        ranks.append(rank)
    return float(np.mean(ranks))

def direction_match(pred_pb, true_pb, n_top=20):
    matches = []
    for i in range(len(pred_pb)):
        t=true_pb[i]; p=pred_pb[i]
        top_de=np.argsort(np.abs(t))[-n_top:]
        sign_match=(np.sign(p[top_de])==np.sign(t[top_de])).mean()
        matches.append(float(sign_match))
    return float(np.mean(matches))

def compute_pred_pseudo_bulk(collected, pert_list):
    pred_pb  = np.zeros((len(pert_list), N_GENES), dtype=np.float32)
    for i,p in enumerate(pert_list):
        msk = collected["y"] == p
        if msk.sum() > 0: pred_pb[i] = collected["x_pred"][msk].mean(0)
    return pred_pb

def corum_retrieval_map(concept_matrix, pert_list, corum_complexes, pert_gene_set):
    N = len(pert_list)
    C = concept_matrix
    norms = np.linalg.norm(C, axis=1, keepdims=True) + 1e-8
    C_n   = C / norms
    ap_list = []
    for i,p in enumerate(pert_list):
        pname = pert_names[p]
        p_complexes = gene_to_complexes.get(pname, set()) & set(valid_complexes.keys())
        if len(p_complexes) == 0: continue
        sims = C_n @ C_n[i]
        sims[i] = -1
        ranked = np.argsort(-sims)
        relevant = []
        for j in ranked:
            qname = pert_names[pert_list[j]]
            q_complexes = gene_to_complexes.get(qname, set()) & set(valid_complexes.keys())
            relevant.append(1 if (p_complexes & q_complexes) else 0)
        if sum(relevant) == 0: continue
        precision_at_k = []
        hits = 0
        for rank, rel in enumerate(relevant[:50], 1):
            if rel: hits+=1; precision_at_k.append(hits/rank)
        if precision_at_k: ap_list.append(np.mean(precision_at_k))
    return float(np.mean(ap_list)) if ap_list else 0.

def string_score_correlation(concept_matrix, pert_list, string_scores_dict):
    norms = np.linalg.norm(concept_matrix,axis=1,keepdims=True)+1e-8
    C_n   = concept_matrix/norms
    cos_sims=[]; str_scores=[]
    for i,p in enumerate(pert_list):
        for j,q in enumerate(pert_list):
            if j<=i: continue
            pname=pert_names[p]; qname=pert_names[q]
            pair=(pname,qname)
            if pair in string_scores_dict:
                cos_sims.append(float(np.dot(C_n[i],C_n[j])))
                str_scores.append(float(string_scores_dict[pair]))
    if len(cos_sims)<10: return 0.,0
    r,p=spearmanr(cos_sims,str_scores)
    return float(r),len(cos_sims)

def train_one(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    tr = make_loader(idx_train, seed=seed)
    vl = make_loader(idx_val,   shuffle=False)
    te = make_loader(idx_test,  shuffle=False)

    model = PerturbationDSRP(
        N_GENES, K, G, M_AND, M_XOR, M_NAND, N_CORUM_CONCEPTS, K_ACTIVE
    ).to(DEVICE)

    with torch.no_grad():
        model.decode_head[0].bias.zero_()

    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*65}")
    print(f"  DSRP v19 | seed={seed} | params={n_par:,}")
    print(f"  Task: X_delta → {K}binary concepts → {G}AND/XOR/NAND → pseudo-bulk recon")
    print(f"  CORUM anchors: {N_CORUM_CONCEPTS} concepts | Free: {K-N_CORUM_CONCEPTS}")
    print(f"{'='*65}")

    pb_dev = pb_t.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=60, T_mult=2, eta_min=LR*0.02)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    best_score=-np.inf; best_state=None; pat=0; pat_max=60; min_ep=80
    hist={k:[] for k in ["Lrec","Lcorum","Lcons","vscore"]}

    for ep in range(1, EPOCHS+1):
        if ep <= STAGE1_END:
            stage=1; tau=1.0; hard=False; prog=ep/STAGE1_END
            lr_=LAM_RECON; lc=LAM_CORUM*prog; lcons=LAM_CONSIST*prog
            le=ls=ld=lmd=lu=ldec=0.
        elif ep <= STAGE2_END:
            stage=2; hard=True
            prog=(ep-STAGE1_END)/(STAGE2_END-STAGE1_END)
            tau=max(0.4,1.0-0.6*prog)
            lr_=LAM_RECON; lc=LAM_CORUM; lcons=LAM_CONSIST
            le=LAM_ENT*prog; ls=LAM_SPA*prog; ld=LAM_DIV*prog
            lmd=LAM_MD*prog; lu=LAM_UNIQ*prog; ldec=LAM_DEC*prog
        else:
            stage=3; tau=0.4; hard=True
            lr_=LAM_RECON; lc=LAM_CORUM; lcons=LAM_CONSIST
            le=LAM_ENT; ls=LAM_SPA; ld=LAM_DIV
            lmd=LAM_MD; lu=LAM_UNIQ; ldec=LAM_DEC

        model.train(); trec=tcor=tcons=0.; nb=0

        for xa,xd,yc,y_corum,_ in tr:
            xd=xd.to(DEVICE,non_blocking=True)
            yc=yc.to(DEVICE,non_blocking=True)
            y_corum=y_corum.to(DEVICE,non_blocking=True)

            def fwd():
                out = model(xd, tau=tau, hard=hard)
                Lr = pseudo_bulk_recon_loss(out["x_pred"], yc, pb_dev)
                Lc = F.binary_cross_entropy_with_logits(out["corum_logits"], y_corum)
                Lcons = concept_consistency_loss(out["c_soft"], yc)
                cd=out["c_soft"].detach(); md=out["m_soft"].detach()
                Le  =ent_loss(out["c_soft"])   if le>0  else torch.tensor(0.,device=DEVICE)
                Ls  =model.sym.spa_loss()       if ls>0  else torch.tensor(0.,device=DEVICE)
                Ld  =div_loss(cd)              if ld>0  else torch.tensor(0.,device=DEVICE)
                Lu  =model.sym.uniq_loss()     if lu>0  else torch.tensor(0.,device=DEVICE)
                Ldec=dec_loss(cd)              if ldec>0 else torch.tensor(0.,device=DEVICE)
                Lmd =md_loss(md)               if lmd>0 else torch.tensor(0.,device=DEVICE)
                L = (lr_*Lr + lc*Lc + lcons*Lcons
                     + le*Le + ls*Ls + ld*Ld + lu*Lu + ldec*Ldec + lmd*Lmd)
                return L, Lr, Lc, Lcons

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    L,Lr,Lc,Lcons = fwd()
                opt.zero_grad(set_to_none=True)
                scaler.scale(L).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt); scaler.update()
            else:
                L,Lr,Lc,Lcons = fwd()
                opt.zero_grad(set_to_none=True)
                L.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()

            m_ = model._orig_mod if hasattr(model,'_orig_mod') else model
            m_.enc.zero_diag()
            sch.step(ep-1+nb/len(tr))
            trec+=Lr.item(); tcor+=Lc.item(); tcons+=Lcons.item(); nb+=1

        vp = collect_concepts(model, vl)
        val_pred_pb = compute_pred_pseudo_bulk(vp, val_pert_list)
        val_true_pb = pseudo_bulk[val_pert_list]
        v_pd  = pearson_delta(val_pred_pb, val_true_pb)
        v_pd20= pearson_delta_top20de(val_pred_pb, val_true_pb, ctrl_mean)
        v_score = 0.6*v_pd + 0.4*v_pd20

        for k0,v in [("Lrec",trec/nb),("Lcorum",tcor/nb),
                      ("Lcons",tcons/nb),("vscore",v_score)]:
            hist[k0].append(v)

        if v_score > best_score:
            best_score=v_score; pat=0
            m_=model._orig_mod if hasattr(model,'_orig_mod') else model
            best_state={k:v.detach().cpu().clone() for k,v in m_.state_dict().items()}
        else:
            pat+=1

        if ep%10==0 or ep==1:
            print(f"[S{stage}] Ep{ep:03d} rec={trec/nb:.4f} corum={tcor/nb:.4f} "
                  f"cons={tcons/nb:.4f} τ={tau:.2f} | "
                  f"PD={v_pd:.4f} PD@20={v_pd20:.4f} score={v_score:.4f}")

        if pat>=pat_max and ep>=min_ep:
            print(f"  Early stop ep={ep}"); break

    m_=model._orig_mod if hasattr(model,'_orig_mod') else model
    m_.load_state_dict({k:v.to(DEVICE) for k,v in best_state.items()})
    model=m_

    print(f"\n  === Seed {seed} Test Metrics ===")
    tp = collect_concepts(model, te)
    test_pred_pb = compute_pred_pseudo_bulk(tp, test_pert_list)
    test_true_pb = pseudo_bulk[test_pert_list]

    pd_all  = pearson_delta(test_pred_pb, test_true_pb)
    pd_top20= pearson_delta_top20de(test_pred_pb, test_true_pb, ctrl_mean)
    rank    = rank_score(test_pred_pb, test_true_pb)
    dirc    = direction_match(test_pred_pb, test_true_pb)

    test_concept_mx = np.zeros((len(test_pert_list), K), dtype=np.float32)
    for i,p in enumerate(test_pert_list):
        msk=tp["y"]==p
        if msk.sum()>0: test_concept_mx[i]=tp["C"][msk].mean(0)

    corum_map = corum_retrieval_map(test_concept_mx, test_pert_list,
                                     corum_complexes, pert_gene_set)
    str_corr, n_str_pairs = string_score_correlation(
        test_concept_mx, test_pert_list, string_scores)

    print(f"  Pearson Delta (all genes)      : {pd_all:.4f}")
    print(f"  Pearson Delta (top20 DE)       : {pd_top20:.4f}")
    print(f"  Rank score (↓ better, 0.5=rand): {rank:.4f}")
    print(f"  Direction match (top20 DE)     : {dirc:.4f}")
    print(f"  CORUM retrieval MAP@50         : {corum_map:.4f}")
    print(f"  STRING score correlation (ρ)   : {str_corr:.4f} (n={n_str_pairs})")

    metrics = {"pd_all":pd_all,"pd_top20":pd_top20,"rank":rank,
               "dir_match":dirc,"corum_map":corum_map,"string_rho":str_corr}

    return {"seed":seed,"model":model,"history":hist,
            "tp":tp,"metrics":metrics,
            "test_pred_pb":test_pred_pb,"test_true_pb":test_true_pb,
            "test_concept_mx":test_concept_mx,"n_par":n_par}

results = [train_one(s) for s in SEEDS]
best_run = max(results, key=lambda r: r["metrics"]["pd_top20"])
model    = best_run["model"]

pd.DataFrame([{"seed":r["seed"],**r["metrics"]} for r in results])\
  .to_csv(OUTDIR/"test_metrics.csv", index=False)

print("\n--- Full cohort concept extraction ---")
full_loader = make_loader(np.arange(len(y_t)), shuffle=False)
full        = collect_concepts(model, full_loader)
C_all = full["C"]; y_all = full["y"]; C_bin = (C_all>0.5).astype(float)

pert_concept_mx = np.zeros((N_CLASSES, K), dtype=np.float32)
for p in range(N_CLASSES):
    msk=y_all==p
    if msk.sum()>0: pert_concept_mx[p]=C_bin[msk].mean(0)
pd.DataFrame(pert_concept_mx, index=pert_names,
             columns=[f"C{k}" for k in range(K)]).to_csv(
    OUTDIR/"pert_concept_matrix.csv")

print("--- Naming concepts ---")
Xd_full = Xd_t.numpy()
Xd_c    = Xd_full - Xd_full.mean(0,keepdims=True)
Xd_s    = Xd_c / (Xd_full.std(0,keepdims=True)+1e-8)
Cb_c    = C_bin - C_bin.mean(0,keepdims=True)
Cb_s    = Cb_c / (C_bin.std(0,keepdims=True)+1e-8)
CHUNK=32; gene_concept_r=np.zeros((K,N_GENES),dtype=np.float32)
for k0 in range(0,K,CHUNK):
    k1=min(k0+CHUNK,K)
    gene_concept_r[k0:k1] = (Cb_s[:,k0:k1].T @ Xd_s / len(y_all))
print(f"  Gene-concept |r| mean:{np.abs(gene_concept_r).mean():.4f}")

concept_names = []
for k in range(K):
    if k < N_CORUM_CONCEPTS:
        cid, info = anchored_complexes[k]
        name = info["name"][:20].replace(" ","_").replace("/","_")
        concept_names.append(f"CORUM_{name}")
    else:
        top_p = np.argsort(pert_concept_mx[:,k])[-1]
        gene  = pert_names[top_p]
        concept_names.append(gene if gene!=CTRL_LABEL else f"C{k}")

lc_=defaultdict(int); dedup=[""]*(K)
for k in range(K):
    base=concept_names[k]; cnt=lc_[base]; lc_[base]+=1
    dedup[k]=base if cnt==0 else f"{base}_{cnt+1}"
concept_names=dedup

pert_jsd=np.zeros(K)
for k in range(K):
    probs=[[max(C_bin[y_all==p,k].mean(),1e-9),
            max(1-C_bin[y_all==p,k].mean(),1e-9)]
           for p in range(N_CLASSES) if (y_all==p).sum()>0]
    p_arr=np.array(probs)+1e-9; p_arr=p_arr/p_arr.sum(1,keepdims=True)
    m_=p_arr.mean(0)
    pert_jsd[k]=float(sum((p*(np.log(p)-np.log(m_))).sum() for p in p_arr)/len(probs))
print(f"  JSD mean:{pert_jsd.mean():.4f} >0.10:{(pert_jsd>0.10).sum()}/{K}")

all_C={}
for r in results:
    m_=r["model"]; m_.eval()
    with torch.no_grad():
        cs=[]
        for _,xd,_,_,_ in make_loader(np.arange(len(y_t)),shuffle=False):
            vl,gl=m_.enc(xd.to(DEVICE))
            cs.append((torch.sigmoid(vl)*torch.sigmoid(gl)).cpu().numpy())
    all_C[r["seed"]]=np.concatenate(cs)
pairs=[(s1,s2) for i,s1 in enumerate(SEEDS) for s2 in SEEDS[i+1:]
       if np.abs(all_C[s1]-all_C[s2]).max()>1e-6]
if pairs:
    stab_r=np.zeros((K,len(pairs)))
    for pi,(s1,s2) in enumerate(pairs):
        C1=all_C[s1]; C2=all_C[s2]
        for k in range(K):
            c1=C1[:,k]-C1[:,k].mean(); c2=C2[:,k]-C2[:,k].mean()
            d=math.sqrt(float((c1**2).sum()*(c2**2).sum()))
            if d>1e-8: stab_r[k,pi]=float((c1*c2).sum()/d)
    mean_stab=stab_r.mean(1)
    print(f"  Stability r mean:{mean_stab.mean():.4f} >0.80:{(mean_stab>0.80).sum()}/{K}")
else:
    mean_stab=np.zeros(K)

mean_stab_s=np.where(np.isnan(mean_stab),0.,mean_stab)
pd.DataFrame({"concept":[f"C{k}" for k in range(K)],"name":concept_names,
              "jsd":pert_jsd,"stability":mean_stab_s,
              "anchored":[k<N_CORUM_CONCEPTS for k in range(K)],
              "max_gene_r":gene_concept_r.max(1),
              }).to_csv(OUTDIR/"concept_summary.csv",index=False)

print("\n--- Extracting symbolic programs ---")
programs = model.get_programs(concept_names)
prog_rows=[]
for g,clauses in programs.items():
    for j,cl in enumerate(clauses):
        prog_rows.append({"module":f"MOD{g}","clause":j,"type":cl["type"],
                           "rule":cl["rule"],"score":cl["score"],"n_terms":cl["n"]})
pd.DataFrame(prog_rows).to_csv(OUTDIR/"symbolic_programs.csv",index=False)
tc_={"AND":0,"XOR":0,"NAND":0}
for g,cl in programs.items():
    for c in cl: tc_[c["type"]]+=1
print(f"  Gates: AND={tc_['AND']} XOR={tc_['XOR']} NAND={tc_['NAND']}")

with open(OUTDIR/"symbolic_programs_readable.txt","w") as f:
    f.write("DSRP v19 Symbolic Programs\n"+"="*60+"\n")
    f.write(f"K={K} binary concepts | G={G} modules | AND/XOR/NAND\n\n")
    for g,clauses in programs.items():
        f.write(f"\nMOD{g}:\n")
        for cl in clauses:
            f.write(f"  [{cl['type']:4s}] score={cl['score']:.3f} n={cl['n']}\n")
            f.write(f"    {cl['rule']}\n")

print("\n--- XOR epistasis validation (held-out expression space) ---")
import re as _re

def parse_arm(arm_str, cnames, Cb):
    n2k={n:k for k,n in enumerate(cnames)}
    for k,n in enumerate(cnames):
        for f in [f"{n}_ON",f"{n}_OFF",f"C{k}",f"C{k}_ON",f"C{k}_OFF"]: n2k[f]=k
    acts=np.zeros(Cb.shape[0]); cidxs=[]; clbls=[]
    for tok in _re.split(r"\s+AND\s+", arm_str.strip("() ")):
        tok=tok.strip()
        if not tok or tok=="∅": continue
        neg=tok.startswith("NOT("); inner=tok.lstrip("NOT(").rstrip(")")
        ki=n2k.get(inner,n2k.get(inner.replace("_ON","").replace("_OFF",""),-1))
        if 0<=ki<Cb.shape[1]:
            acts=np.maximum(acts, 1-Cb[:,ki] if neg else Cb[:,ki])
            cidxs.append(ki); clbls.append(cnames[ki])
    return (acts>0.5).astype(int),cidxs,clbls

test_C_bin   = (best_run["test_concept_mx"] > 0.5).astype(float)
test_pred_pb = best_run["test_pred_pb"]
test_true_pb = best_run["test_true_pb"]
test_pert_arr= np.array(test_pert_list)

def cosine(a,b):
    na=np.linalg.norm(a); nb=np.linalg.norm(b)
    return float(np.dot(a,b)/(na*nb)) if na>1e-12 and nb>1e-12 else 0.

def perm_test_cos(profA, profB, n_perm=500, rng=None):
    if rng is None: rng=np.random.default_rng(42)
    all_p=np.concatenate([profA,profB]); nA=len(profA)
    obs=np.mean([cosine(a,b) for a in profA for b in profB])
    null=[np.mean([cosine(a,b) for a in all_p[pm[:nA]] for b in all_p[pm[nA:]]])
          for pm in [rng.permutation(len(all_p)) for _ in range(n_perm)]]
    return obs, float(np.mean(np.array(null)<=obs))

RNG=np.random.default_rng(0)
xor_rows=[]
for g,clauses in programs.items():
    for cl in clauses:
        if cl["type"]!="XOR": continue
        parts=cl["rule"].split(" XOR ")
        if len(parts)!=2: continue
        A,cia,cla=parse_arm(parts[0],concept_names,test_C_bin)
        B,cib,clb=parse_arm(parts[1],concept_names,test_C_bin)
        if A.sum()==0 and B.sum()==0: continue

        armA=[]; armB=[]
        for i,p in enumerate(test_pert_list):
            sA=best_run["test_concept_mx"][i][cia].mean() if cia else 0.
            sB=best_run["test_concept_mx"][i][cib].mean() if cib else 0.
            if sA>sB+0.05: armA.append(i)
            elif sB>sA+0.05: armB.append(i)

        if len(armA)<2 or len(armB)<2: continue

        profA=test_true_pb[armA]; profB=test_true_pb[armB]
        obs_cos, perm_p = perm_test_cos(profA, profB, n_perm=300, rng=RNG)

        meanA=profA.mean(0); meanB=profB.mean(0)
        nA_=np.linalg.norm(meanA); nB_=np.linalg.norm(meanB)
        if nA_>1e-12 and nB_>1e-12:
            projA=test_true_pb@meanA/nA_; projB=test_true_pb@meanB/nB_
            cellA=(projA>projB).astype(int); cellB=(projB>projA).astype(int)
            n11=int((cellA&cellB).sum()); n10=int((cellA&~cellB.astype(bool)).sum())
            n01=int((~cellA.astype(bool)&cellB).sum())
            n00=int((~cellA.astype(bool)&~cellB.astype(bool)).sum())
            dp=math.sqrt((n11+n10)*(n01+n00)*(n11+n01)*(n10+n00))
            expr_phi=(n11*n00-n10*n01)/dp if dp>0 else 0.
        else: expr_phi=0.; n11=n10=n01=n00=0

        armA_genes={pert_names[test_pert_list[i]] for i in armA}
        armB_genes={pert_names[test_pert_list[i]] for i in armB}
        armA_cxs={cid for g_ in armA_genes for cid in gene_to_complexes.get(g_,set())}
        armB_cxs={cid for g_ in armB_genes for cid in gene_to_complexes.get(g_,set())}
        shared_cx=armA_cxs & armB_cxs
        inter_scores=[string_scores.get((a,b),string_scores.get((b,a),np.nan))
                      for a in armA_genes for b in armB_genes]
        inter_scores=[s for s in inter_scores if not np.isnan(s)]
        mean_inter=float(np.mean(inter_scores)) if inter_scores else np.nan

        xor_rows.append({
            "module":f"MOD{g}","rule":cl["rule"],
            "arm_A_concepts":"+".join(cla) if cla else "∅",
            "arm_B_concepts":"+".join(clb) if clb else "∅",
            "arm_A_genes":";".join(sorted(armA_genes)[:5]),
            "arm_B_genes":";".join(sorted(armB_genes)[:5]),
            "n_perts_A":len(armA),"n_perts_B":len(armB),
            "obs_cos_sim":round(obs_cos,4),"perm_p":round(perm_p,4),
            "expr_phi":round(expr_phi,4),
            "shared_corum_complexes":len(shared_cx),
            "mean_string_inter":round(mean_inter,1) if not np.isnan(mean_inter) else "NA",
            "xor_score":round(-obs_cos,4),
        })

df_xor=pd.DataFrame(xor_rows)
if len(df_xor)>0:
    pv=df_xor["perm_p"].values; nt=len(pv); ord_=np.argsort(pv)
    rks=np.empty(nt,int); rks[ord_]=np.arange(1,nt+1)
    qv=np.minimum(1.,pv*nt/rks)
    for i in range(nt-2,-1,-1): qv[ord_[i]]=min(qv[ord_[i]],qv[ord_[i+1]])
    df_xor["perm_q"]=qv
    df_xor["verified"]=(df_xor["perm_q"]<0.05)&(df_xor["expr_phi"]<-0.10)
    df_xor["corum_separation"]=(df_xor["shared_corum_complexes"]==0)
    df_xor=df_xor.sort_values(["verified","xor_score"],ascending=[False,False])

df_xor.to_csv(OUTDIR/"xor_validation.csv",index=False)
n_ver=int(df_xor["verified"].sum()) if len(df_xor)>0 else 0
n_sep=int((df_xor["verified"]&df_xor["corum_separation"]).sum()) if len(df_xor)>0 else 0
print(f"  XOR clauses:{len(df_xor)} verified:{n_ver} CORUM-separated:{n_sep}")

print(f"\n{'='*65}")
print("DSRP v19 — FINAL RESULTS")
print(f"{'='*65}")
print(f"  Task: perturbation effect organization via symbolic programs")
print(f"  {len(y_t):,} cells | {N_CLASSES} perturbations | K={K} G={G}")
print(f"  Train:{len(train_perts)} Val:{len(val_perts)} Test:{len(test_perts)} perts (held-out)")
print()
print(f"  EXPRESSION PREDICTION (standard benchmark metrics):")
for r in results:
    m=r["metrics"]
    print(f"  Seed{r['seed']}: PD={m['pd_all']:.4f} PD@20={m['pd_top20']:.4f} "
          f"rank={m['rank']:.4f} dir={m['dir_match']:.4f}")
print(f"  Mean PD    : {np.mean([r['metrics']['pd_all']  for r in results]):.4f}±"
      f"{np.std([r['metrics']['pd_all']  for r in results]):.4f}")
print(f"  Mean PD@20 : {np.mean([r['metrics']['pd_top20'] for r in results]):.4f}±"
      f"{np.std([r['metrics']['pd_top20'] for r in results]):.4f}")
print(f"  Mean rank  : {np.mean([r['metrics']['rank']    for r in results]):.4f}  (0.5=random,0=perfect)")
print(f"  Mean dir   : {np.mean([r['metrics']['dir_match'] for r in results]):.4f}")
print()
print(f"  INTERPRETABILITY (concept-space validation):")
print(f"  CORUM retrieval MAP@50 : {np.mean([r['metrics']['corum_map']   for r in results]):.4f}")
print(f"  STRING correlation ρ   : {np.mean([r['metrics']['string_rho']  for r in results]):.4f}")
print(f"  JSD>0.10 : {(pert_jsd>0.10).sum()}/{K}  Stab>0.80:{(mean_stab_s>0.80).sum()}/{K}")
print(f"  XOR verified (held-out expression): {n_ver}  CORUM-separated: {n_sep}")
print(f"  CORUM anchors: {N_CORUM_CONCEPTS}  Free concepts: {K-N_CORUM_CONCEPTS}")
print()
print(f"  BASELINES (from literature on K562 Replogle):")
print(f"  Train Mean PD   : ~0.373  (Csendes et al. 2025 BMC Genomics)")
print(f"  RF+GO features  : ~0.480")
print(f"  scGPT PD        : ~0.327")
print(f"  scFoundation PD : ~0.269")
print(f"  GEARS PD        : ~0.35 (varies by split)")
print(f"{'='*65}")

print("\n--- Visualization ---")
hist=best_run["history"]; bm=best_run["metrics"]

fig=plt.figure(figsize=(24,48))
gs_=gridspec.GridSpec(6,3,figure=fig,hspace=0.5,wspace=0.35)

ax=fig.add_subplot(gs_[0,0])
ep_x=range(1,len(hist["Lrec"])+1)
ax.plot(ep_x,hist["Lrec"],color="steelblue",lw=1.5,label="Recon (1-Pearson r)")
ax.plot(ep_x,hist["Lcorum"],color="orange",lw=1.,linestyle="--",label="CORUM supervision")
ax.plot(ep_x,hist["Lcons"],color="purple",lw=1.,linestyle=":",label="Consistency")
ax2=ax.twinx()
ax2.plot(ep_x,hist["vscore"],color="tomato",lw=2.,label="Val PD score")
ax.legend(fontsize=6,loc="upper left"); ax2.legend(fontsize=6,loc="upper right")
ax.set_title("A  Training Curves",fontweight="bold")

ax=fig.add_subplot(gs_[0,1])
test_r=[pearsonr(best_run["test_pred_pb"][i],
                  best_run["test_true_pb"][i])[0]
         for i in range(len(test_pert_list))]
ax.hist(test_r,bins=30,color="steelblue",alpha=0.8)
ax.axvline(np.mean(test_r),color="black",lw=1.5,label=f"Mean={np.mean(test_r):.3f}")
ax.axvline(0.,color="gray",lw=1.,linestyle="--")
ax.set_xlabel("Pearson r (pred vs true X_delta)")
ax.legend(fontsize=8); ax.set_title("B  Per-Perturbation Pearson Δ (Test)",fontweight="bold")

ax=fig.add_subplot(gs_[0,2])
ranks=[]; norms_pb=np.linalg.norm(best_run["test_true_pb"],axis=1,keepdims=True)+1e-8
true_n=best_run["test_true_pb"]/norms_pb
pred_n_arr=best_run["test_pred_pb"]/(np.linalg.norm(best_run["test_pred_pb"],axis=1,keepdims=True)+1e-8)
for i in range(len(test_pert_list)):
    sims=true_n@pred_n_arr[i]
    ranks.append(float(np.where(np.argsort(-sims)==i)[0][0])/len(test_pert_list))
ax.hist(ranks,bins=30,color="teal",alpha=0.8)
ax.axvline(0.5,color="gray",lw=1.,linestyle="--",label="Random=0.5")
ax.axvline(np.mean(ranks),color="black",lw=1.5,label=f"Mean={np.mean(ranks):.3f}")
ax.legend(fontsize=8); ax.set_title("C  Rank Score Distribution (↓ better)",fontweight="bold")

ax=fig.add_subplot(gs_[1,0])
show_n=min(20,N_CORUM_CONCEPTS)
corum_act=np.zeros((show_n,show_n))
for i,(cid,info) in enumerate(anchored_complexes[:show_n]):
    for j,(cid2,info2) in enumerate(anchored_complexes[:show_n]):
        genes_j_in_data=[pi for pi,pn in enumerate(pert_names) if pn in info2["genes"]]
        if genes_j_in_data:
            corum_act[i,j]=pert_concept_mx[genes_j_in_data,i].mean()
sns.heatmap(corum_act,ax=ax,cmap="YlOrRd",
            xticklabels=[anchored_complexes[i][1]["name"][:10] for i in range(show_n)],
            yticklabels=[f"C{i}" for i in range(show_n)])
ax.tick_params(labelsize=5)
ax.set_title("D  CORUM-Anchored Concept Specificity",fontweight="bold")

ax=fig.add_subplot(gs_[1,1])
ax.hist(pert_jsd[:N_CORUM_CONCEPTS],bins=20,alpha=0.7,color="orange",label=f"CORUM-anchored (n={N_CORUM_CONCEPTS})")
ax.hist(pert_jsd[N_CORUM_CONCEPTS:],bins=20,alpha=0.7,color="steelblue",label=f"Free (n={K-N_CORUM_CONCEPTS})")
ax.axvline(0.10,color="red",lw=1.,linestyle="--")
ax.legend(fontsize=7); ax.set_title("E  Perturbation Specificity (JSD)\nAnchored vs Free Concepts",fontweight="bold")

ax=fig.add_subplot(gs_[1,2])
test_concept_mx_n=best_run["test_concept_mx"]
norms_c=np.linalg.norm(test_concept_mx_n,axis=1,keepdims=True)+1e-8
Cn=test_concept_mx_n/norms_c
cos_sims_plt=[]; str_scores_plt=[]
for i,p in enumerate(test_pert_list):
    for j,q in enumerate(test_pert_list):
        if j<=i: continue
        pair=(pert_names[p],pert_names[q])
        if pair in string_scores or (pair[1],pair[0]) in string_scores:
            s=string_scores.get(pair,string_scores.get((pair[1],pair[0]),None))
            if s: cos_sims_plt.append(float(np.dot(Cn[i],Cn[j]))); str_scores_plt.append(float(s))
if cos_sims_plt:
    ax.scatter(cos_sims_plt,str_scores_plt,alpha=0.3,s=5,color="steelblue")
    ax.set_xlabel("Concept cosine similarity"); ax.set_ylabel("STRING score")
    rho,_=spearmanr(cos_sims_plt,str_scores_plt)
    ax.set_title(f"F  STRING Score vs Concept Similarity\nSpearman ρ={rho:.3f}",fontweight="bold")
else:
    ax.text(0.5,0.5,"No STRING pairs found\nin test set",
            ha="center",va="center",transform=ax.transAxes)
    ax.set_title("F  STRING Score vs Concept Similarity",fontweight="bold")

ax=fig.add_subplot(gs_[2,0])
if len(df_xor)>0:
    top_x=df_xor.head(min(20,len(df_xor)))
    cols=["mediumseagreen" if v else "lightgray" for v in top_x["verified"]]
    ax.barh(range(len(top_x)),top_x["xor_score"].values,color=cols)
    ax.axvline(0.10,color="red",lw=1.,linestyle="--")
    ax.set_yticks(range(len(top_x)))
    ax.set_yticklabels([f"{r['arm_A_concepts'][:12]} XOR {r['arm_B_concepts'][:12]}"
                        for _,r in top_x.iterrows()],fontsize=4)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="mediumseagreen",label="Verified"),
                        Patch(color="lightgray",label="Not verified")],fontsize=7)
ax.set_title("G  XOR Epistasis (held-out expression space)",fontweight="bold")

ax=fig.add_subplot(gs_[2,1])
cs_np=model.sym.cs.detach().cpu().numpy()
xlbls=(["A"]*M_AND+["X"]*M_XOR+["N"]*M_NAND)
sns.heatmap(cs_np,ax=ax,cmap="YlGn",
            xticklabels=xlbls,yticklabels=[f"M{g}" for g in range(G)],linewidths=0.1)
ax.tick_params(labelsize=4)
ax.axvline(M_AND,color="red",lw=1.5,linestyle="--")
ax.axvline(M_AND+M_XOR,color="blue",lw=1.5,linestyle="--")
ax.set_title("H  Clause Scores (AND|XOR|NAND)",fontweight="bold")

ax=fig.add_subplot(gs_[2,2])
colors_g={"AND":"#2196F3","XOR":"#FF9800","NAND":"#9C27B0"}
ax.bar(tc_.keys(),[tc_[t] for t in tc_],color=[colors_g[t] for t in tc_],alpha=0.85)
for i,(t,c) in enumerate(tc_.items()):
    ax.text(i,c+0.3,f"{c}",ha="center",fontsize=10,fontweight="bold")
ax.set_title("I  Gate Distribution",fontweight="bold")

ax=fig.add_subplot(gs_[3,0])
ax.hist(mean_stab_s[:N_CORUM_CONCEPTS],bins=20,alpha=0.7,color="orange",label="CORUM-anchored")
ax.hist(mean_stab_s[N_CORUM_CONCEPTS:],bins=20,alpha=0.7,color="steelblue",label="Free")
ax.axvline(0.8,color="red",lw=1.,linestyle=":",label="r=0.80")
ax.legend(fontsize=7); ax.set_title("J  Concept Stability (3-seed ICC proxy)",fontweight="bold")

ax=fig.add_subplot(gs_[3,1])
softness=(C_all*(1-C_all)).mean(0)
ax.bar(range(K),softness,color=["orange" if k<N_CORUM_CONCEPTS else
                                  ("tomato" if softness[k]>0.2 else "steelblue")
                                  for k in range(K)])
ax.axhline(softness.mean(),color="black",lw=1.5,label=f"Mean={softness.mean():.3f}")
ax.legend(fontsize=7); ax.set_title("K  Concept Binariness c(1-c)",fontweight="bold")

ax=fig.add_subplot(gs_[3,2]); ax.axis("off")
prog_txt=["Top Symbolic Programs","─"*40]
for g in range(min(8,G)):
    cls=programs[g]
    for cl in cls[:2]:
        prog_txt.append(f"MOD{g}[{cl['type']}] {cl['rule'][:45]}")
ax.text(0.02,0.98,"\n".join(prog_txt),transform=ax.transAxes,
        va="top",ha="left",fontsize=6,fontfamily="monospace",
        bbox=dict(boxstyle="round",facecolor="lightyellow",alpha=0.9))
ax.set_title("L  Sample Programs",fontweight="bold")

ax=fig.add_subplot(gs_[4,0])
baselines={"Train Mean":0.373,"RF+GO":0.480,"scGPT":0.327,"scFoundation":0.269,"GEARS":0.35}
our_pd=np.mean([r["metrics"]["pd_all"] for r in results])
methods=list(baselines.keys())+["DSRP v19\n(ours)"]
values=list(baselines.values())+[our_pd]
colors_b=["lightgray"]*len(baselines)+["steelblue"]
ax.bar(methods,values,color=colors_b,alpha=0.85)
ax.axhline(our_pd,color="steelblue",lw=1.5,linestyle="--")
ax.set_ylabel("Pearson Delta")
ax.set_title("M  Benchmark Comparison\n(Replogle K562, Pearson Δ)",fontweight="bold")
ax.tick_params(labelsize=7)

ax=fig.add_subplot(gs_[4,1])
our_map=np.mean([r["metrics"]["corum_map"] for r in results])
n_with_cx=sum(1 for p in test_pert_list if gene_to_complexes.get(pert_names[p],set())&set(valid_complexes.keys()))
ax.bar(["Random\nbaseline","DSRP v19\nconcept space"],
        [0.05,our_map],color=["lightgray","steelblue"],alpha=0.85)
ax.set_ylabel("MAP@50 (CORUM retrieval)")
ax.set_title("N  CORUM Complex Retrieval\n(concept space vs random)",fontweight="bold")

ax=fig.add_subplot(gs_[4,2]); ax.axis("off")
smry=["DSRP v19 — Summary","─"*38,
      f"K562 | {len(y_t):,} cells | {N_CLASSES} perts",
      f"K={K} | G={G} | AND/XOR/NAND","─"*38,
      "EXPRESSION PREDICTION:",
      f"  PD (all)    : {bm['pd_all']:.4f}",
      f"  PD (top20)  : {bm['pd_top20']:.4f}",
      f"  Rank        : {bm['rank']:.4f}  (↓ better)",
      f"  Dir match   : {bm['dir_match']:.4f}","─"*38,
      "INTERPRETABILITY:",
      f"  CORUM MAP@50: {bm['corum_map']:.4f}",
      f"  STRING ρ    : {bm['string_rho']:.4f}",
      f"  XOR verified: {n_ver}",
      f"  CORUM-sep   : {n_sep}","─"*38,
      f"BASELINES (K562 Replogle):",
      f"  Train Mean  : 0.373",
      f"  RF+GO       : 0.480",
      f"  scGPT       : 0.327",
      f"  scFoundation: 0.269"]
ax.text(0.02,0.98,"\n".join(smry),transform=ax.transAxes,va="top",ha="left",
        fontsize=7.5,fontfamily="monospace",
        bbox=dict(boxstyle="round",facecolor="lightyellow",alpha=0.9))
ax.set_title("O  Summary",fontweight="bold")

ax=fig.add_subplot(gs_[5,0])
mkeys=["pd_all","pd_top20","corum_map","string_rho"]
x_=np.arange(len(SEEDS)); w=0.2
for i,(mk,col) in enumerate(zip(mkeys,["steelblue","teal","orange","tomato"])):
    ax.bar(x_+i*w,[r["metrics"][mk] for r in results],w,label=mk,color=col,alpha=0.85)
ax.set_xticks(x_+w); ax.set_xticklabels([f"Seed{s}" for s in SEEDS])
ax.legend(fontsize=6); ax.set_title("P  Seed Comparison",fontweight="bold")

ax=fig.add_subplot(gs_[5,1])
n_and_=[sum(1 for c in programs[g] if c["type"]=="AND") for g in range(G)]
n_xor_=[sum(1 for c in programs[g] if c["type"]=="XOR") for g in range(G)]
n_nand_=[sum(1 for c in programs[g] if c["type"]=="NAND") for g in range(G)]
x_=np.arange(G)
ax.bar(x_,n_and_,label="AND",color="#2196F3",alpha=0.85)
ax.bar(x_,n_xor_,bottom=n_and_,label="XOR",color="#FF9800",alpha=0.85)
ax.bar(x_,n_nand_,[n_and_[i]+n_xor_[i] for i in range(G)],label="NAND",color="#9C27B0",alpha=0.85)
ax.legend(fontsize=6); ax.set_title("Q  Clauses per Module",fontweight="bold")

plt.suptitle(
    f"DSRP v19 | K562 Perturbation-Seq | Neurosymbolic Regulatory Programs\n"
    f"K={K} CORUM-anchored+free binary concepts | G={G} AND/XOR/NAND modules\n"
    f"PD={bm['pd_all']:.4f} PD@20={bm['pd_top20']:.4f} "
    f"Rank={bm['rank']:.4f} CORUM-MAP={bm['corum_map']:.4f} STRING-ρ={bm['string_rho']:.4f}",
    fontsize=9,fontweight="bold",y=1.002)

plt.savefig(OUTDIR/"dsrp_v19_panel.png",dpi=150,bbox_inches="tight")
plt.close(); print(f"Saved: {OUTDIR/'dsrp_v19_panel.png'}")
print(f"All outputs: {OUTDIR}")
