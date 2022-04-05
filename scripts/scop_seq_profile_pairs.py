import random

lookup_path = "../data/scop/scop_lookup.fix.tsv"
pssm_path = "../data/scop/scop_mmseqs_pssm.pssm"
res_path = "../data/scop/pairs.pssm"

seq_to_scop = {}
families = {}

with open(lookup_path) as f:
    for line in f:
        seq_id, scop_id = line.strip().split()
        seq_to_scop[seq_id] = scop_id
        families[scop_id] = []

seq_lines = {}

with open(pssm_path) as f:
    curr_seq = None
    for line in f:
        if line.startswith("#"):
            curr_seq = line[1:].strip()
            seq_lines[curr_seq] = []
        else:
            seq_lines[curr_seq].append(line.strip())

for (seq_id, pssm) in seq_lines.items():
    scop_id = seq_to_scop[seq_id]
    families[scop_id].append(pssm)

sorted_families = list(sorted(families.items(), key = lambda x: len(x[1]), reverse = True))
seq_pssm_pairs = []

def consensus_seq(lines):
    return "".join([s.split()[1] for s in lines[1:]])

for _, family in sorted_families:
    if len(family) <= 2:
        continue
    random.shuffle(family)
    for i in range(0, len(family) - (len(family) % 2), 2):
        seq_pssm_pairs.append((consensus_seq(family[i]), consensus_seq(family[i + 1]), family[i + 1]))

print("Number of seq-pssm pairs:", len(seq_pssm_pairs))

with open(res_path, "w") as f:
    for cns, pssm_cns, pssm in seq_pssm_pairs:
        f.write("#" + cns + "\n")
        f.write("#" + pssm_cns + "\n")
        f.write("\n".join(pssm) + "\n")
