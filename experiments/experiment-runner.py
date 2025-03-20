import subprocess
import os
import datetime

env = os.environ.copy()
env["OMP_NUM_THREADS"] = "8"

def run_conann(dataset, calib_sz, tune_sz, alpha, nlist, k):
    print(f"running conann: dataset={dataset}, calib_size={calib_sz}, tune_size={tune_sz}, alpha={alpha}, nlist={nlist}, k={k}")
    try:
        result = subprocess.run(["./build/eval/error", dataset, str(calib_sz), str(tune_sz), str(alpha), str(nlist), str(k)], 
                            capture_output=True, 
                            text=True,
                            cwd=os.path.abspath("../conann"),
                            check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Failed conann run with params: {dataset}, {calib_sz}, {tune_sz}, {alpha}, {nlist}, {k}\n")
        with open(f"Failed_conann_{dataset}_{calib_sz}_{tune_sz}_{alpha}_{nlist}_{k}.log", "a") as f:
            f.write(f"Error running conann with params: {dataset}, {calib_sz}, {tune_sz}, {alpha}, {nlist}, {k}\n")
            f.write(f"Timestamp: {datetime.datetime.now()}\n")
            f.write(f"Return code: {e.returncode}\n")
            f.write(f"stdout: {e.stdout}\n")
            f.write(f"stderr: {e.stderr}\n\n")

def run_faiss(dataset, calib_sz, nlist, k, starting_nprobe, alphas: tuple):
    print(f"running faiss: dataset={dataset}, calib_size={calib_sz}, nlist={nlist}, k={k}, alphas={alphas}")
    try:
        result = subprocess.run(["./build/eval/error", dataset, str(calib_sz), str(nlist), str(k), str(starting_nprobe), *[str(a) for a in alphas]], 
                            capture_output=True, 
                            text=True,
                            cwd=os.path.abspath("../faiss-1.9.0"),
                            check=True)
        print(result.stdout)
        return int(result.stdout.splitlines()[-1])
    except subprocess.CalledProcessError as e:
        print(f"Failed faiss run with params: {dataset}, {calib_sz}, {nlist}, {k}, {starting_nprobe}, {alphas}\n")
        with open(f"Failed_faiss_{dataset}_{calib_sz}_{nlist}_{k}_{starting_nprobe}.log", "a") as f:
            f.write(f"Error running faiss with params: {dataset}, {calib_sz}, {nlist}, {k}, {starting_nprobe}, {alphas}\n")
            f.write(f"Timestamp: {datetime.datetime.now()}\n")
            f.write(f"Return code: {e.returncode}\n")
            f.write(f"stdout: {e.stdout}\n")
            f.write(f"stderr: {e.stderr}\n\n")


datasets = ("bert", "glove", "sift1M", "deep10M", "gist")
alphas = (0.5, 0.4, 0.3, 0.2, 0.1, 0.05)
ks = (10, 100, 1000)
nlist = {"bert": 128, "glove": 1024, "sift1M": 1024, "deep10M": 1024, "gist": 1024}
nlist_sqrt_n = {"bert": 173, "glove": 1414, "sift1M": 1000, "deep10M": 3162, "gist": 1000}

for dataset in datasets:
    for k in ks:
        for alpha in alphas:
            run_conann(dataset, 0.5, 0.2, alpha, nlist[dataset], k)


for dataset in datasets:
    for k in ks:
        run_faiss(dataset, 0.5, nlist[dataset], k, 1, alphas)