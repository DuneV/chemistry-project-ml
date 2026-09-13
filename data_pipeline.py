"""
Pipeline de datos para el VAE de SELFIES.
==========================================
Descarga ZINC250k, convierte SMILES -> SELFIES, construye el vocabulario
de tokens y produce tensores listos para entrenar.

Uso:
    python data_pipeline.py --n_sample 2000   # subconjunto pequeno (smoke test en CPU)
    python data_pipeline.py                    # dataset completo (correr en GPU)
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
import selfies as sf
import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")  # silenciar warnings de RDKit durante la limpieza

ZINC_URL = ("https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/"
            "master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv")

PAD, SOS, EOS, UNK = "[pad]", "[sos]", "[eos]", "[unk]"
SPECIAL_TOKENS = [PAD, SOS, EOS, UNK]


def load_raw_smiles(csv_path: str) -> list:
    """Lee el CSV de ZINC y devuelve una lista de SMILES limpios."""
    df = pd.read_csv(csv_path)
    smiles = df["smiles"].astype(str).str.replace("\n", "", regex=False).str.strip().tolist()
    return smiles


def smiles_to_selfies(smiles_list: list) -> list:
    """Convierte SMILES validos a SELFIES. Descarta silenciosamente lo que RDKit no puede parsear.
    Esto NO es opcional: SELFIES garantiza que el decoder solo pueda generar
    quimica valida por construccion (a diferencia de SMILES crudo)."""
    out = []
    n_failed = 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_failed += 1
            continue
        canon = Chem.MolToSmiles(mol)
        try:
            out.append(sf.encoder(canon))
        except Exception:
            n_failed += 1
    print(f"[data_pipeline] SELFIES validos: {len(out)} | descartados: {n_failed}")
    return out


def build_vocab(selfies_list: list) -> dict:
    """Construye el vocabulario token -> indice a partir del alfabeto SELFIES observado."""
    alphabet = sf.get_alphabet_from_selfies(selfies_list)
    alphabet = sorted(alphabet)
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS + alphabet)}
    return vocab


def encode_selfies(selfies_list: list, vocab: dict, max_len: int) -> torch.Tensor:
    """Tokeniza cada SELFIES a una secuencia de indices de longitud fija (con padding)."""
    pad_idx, sos_idx, eos_idx, unk_idx = (vocab[PAD], vocab[SOS], vocab[EOS], vocab[UNK])
    encoded = torch.full((len(selfies_list), max_len), pad_idx, dtype=torch.long)
    for i, s in enumerate(selfies_list):
        toks = list(sf.split_selfies(s))[: max_len - 2]  # deja espacio para sos/eos
        ids = [sos_idx] + [vocab.get(t, unk_idx) for t in toks] + [eos_idx]
        encoded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return encoded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="zinc250k_raw.csv")
    ap.add_argument("--n_sample", type=int, default=None,
                     help="Si se especifica, usa solo N moleculas (para smoke test en CPU)")
    ap.add_argument("--max_len", type=int, default=72,
                     help="Longitud maxima de secuencia SELFIES (en tokens)")
    ap.add_argument("--out_dir", default="processed")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    smiles = load_raw_smiles(args.csv)
    if args.n_sample is not None:
        idx = rng.choice(len(smiles), size=min(args.n_sample, len(smiles)), replace=False)
        smiles = [smiles[i] for i in idx]
    print(f"[data_pipeline] SMILES cargados: {len(smiles)}")

    selfies_list = smiles_to_selfies(smiles)
    vocab = build_vocab(selfies_list)
    print(f"[data_pipeline] Tamano de vocabulario: {len(vocab)}")

    encoded = encode_selfies(selfies_list, vocab, args.max_len)

    n = encoded.shape[0]
    n_val = max(1, int(0.05 * n))
    perm = torch.randperm(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    torch.save(encoded[train_idx], os.path.join(args.out_dir, "train.pt"))
    torch.save(encoded[val_idx], os.path.join(args.out_dir, "val.pt"))
    with open(os.path.join(args.out_dir, "vocab.json"), "w") as f:
        json.dump({"vocab": vocab, "max_len": args.max_len}, f, indent=2)

    print(f"[data_pipeline] Guardado: {len(train_idx)} train / {len(val_idx)} val en {args.out_dir}/")


if __name__ == "__main__":
    main()
