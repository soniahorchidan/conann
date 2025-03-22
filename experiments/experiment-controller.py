import subprocess
import os
import datetime

"""
****************************************************
*                                                  *
*        Experiment Controller Script              *
*                                                  *
* Execute multiple test scenarios safely by catching and logging errors.
*  
* Convenience methods to run query sampling and ground truth execution from this python file.
* 
* NOTE:
* Still requires the hardcoded paths to the datasets to be correct unfortunately.
* All c++ code needs to be manually build (to release ideally) beforehand, sorry.
*                                                  
****************************************************
"""

env = os.environ.copy()
env["OMP_NUM_THREADS"] = "32"

def sample_dataset(dataset, sample_size, out_filename):
    print(f"Sample {sample_size} to {out_filename} from {dataset}")
    try:
        result = subprocess.run(["./Auncel/build/eval/sample_queries", dataset, str(sample_size), out_filename],
                            capture_output=True, 
                            text=True,
                            cwd=os.path.abspath(".."),
                            check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("something went wrong:")
        print(e.stdout)
        print(e.stderr)


def test_query_size(dataset):
    print("Query size for ", dataset)
    try:
        result = subprocess.run(["./build/eval/compute_gt", dataset, "-1"],
                            capture_output=True, 
                            text=True,
                            cwd=os.path.abspath("../Auncel"),
                            check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("something went wrong:")
        print(e.stdout)
        print(e.stderr)

def compute_gt(dataset, ks: tuple):
    print(f"running faiss: dataset={dataset}, ks={ks}")
    try:
        result = subprocess.run(["./build/eval/compute_gt", dataset, *[str(a) for a in ks]], 
                            capture_output=True, 
                            text=True,
                            cwd=os.path.abspath("../Auncel"),
                            check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Failed compute_gt run with params: {dataset}, {ks}\n")
        with open(f"Failed_compute_gt_{dataset}.log", "a") as f:
            f.write(f"Error running faiss with params: {dataset}, {ks}\n")
            f.write(f"Timestamp: {datetime.datetime.now()}\n")
            f.write(f"Return code: {e.returncode}\n")
            f.write(f"stdout: {e.stdout}\n")
            f.write(f"stderr: {e.stderr}\n\n")

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


# PARAMETERS:
datasets = ("bert", "glove", "sift1M", "deep10M", "gist")
alphas = (0.5, 0.4, 0.3, 0.2, 0.1, 0.05)
ks = (10, 100, 1000)
calib_sz = 0.5
tuning_sz = {"bert": 0.2, "glove": 0.1, "sift1M": 0.1, "deep10M": 0.1, "gist": 0.1}
nlist = {"bert": 128, "glove": 1024, "sift1M": 1024, "deep10M": 1024, "gist": 1024}
nlist_sqrt_n = {"bert": 173, "glove": 1414, "sift1M": 1000, "deep10M": 3162, "gist": 1000}
faiss_starting_nprobe = 1

"""
* NOTE: Number of queries in the current data folders labeled as (small sample for me):
* bert: 4578
* glove: 10000
* sift1M: 150000 (sift-query file had 10000!, might need new ground truths)
* deep10M: 10000
* gist: 1000 (too small: needs resampling it seems)
"""

# Possible command combo to sample new dataset:
# First prepare folder at ./data/gist with gist_base.fvecs inside
# sample_dataset("./data/gist/gist_base.fvecs", 10000, "queries.fvecs")
# compute_gt("gist", (1000, 100, 10))


"""
* NOTE: Remove when good to go:
"""
for dataset in datasets:
    test_query_size(dataset)
exit(0) 

# Primary experiment section running on multiple alphas and ks.
for dataset in datasets:
    for k in ks:
        for alpha in alphas:
            run_conann(dataset, calib_sz, tuning_sz[dataset], alpha, nlist[dataset], k)


for dataset in datasets:
    for k in ks:
        run_faiss(dataset, calib_sz, nlist[dataset], k, faiss_starting_nprobe, alphas)