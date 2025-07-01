import subprocess
import os
import datetime

env = os.environ.copy()
env["OMP_NUM_THREADS"] = "60"

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
    print("Query size for", dataset)
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
            f.write(f"Error running compute_gt with params: {dataset}, {ks}\n")
            f.write(f"Timestamp: {datetime.datetime.now()}\n")
            f.write(f"Return code: {e.returncode}\n")
            f.write(f"stdout: {e.stdout}\n")
            f.write(f"stderr: {e.stderr}\n\n")

def run_conann(experiment, dataset, calib_sz, tune_sz, alpha, nlist, k):
    print(f"running conann: experiment={experiment}, dataset={dataset}, calib_size={calib_sz}, tune_size={tune_sz}, alpha={alpha}, nlist={nlist}, k={k}")
    try:
        result = subprocess.run([f"./build/eval/{experiment}", dataset, str(calib_sz), str(tune_sz), str(alpha), str(nlist), str(k)], 
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

def run_conann_variable_k(dataset, calib_sz, tune_sz, alpha, nlist, lower_bound_k, upper_bound_k):
    print(f"running conann: experiment=variable_k, dataset={dataset}, calib_size={calib_sz}, tune_size={tune_sz}, alpha={alpha}, nlist={nlist}, lower_bound_k={lower_bound_k}, upper_bound_k={upper_bound_k}")
    try:
        result = subprocess.run([f"./build/eval/variable_k", dataset, str(calib_sz), str(tune_sz), str(alpha), str(nlist), str(lower_bound_k), str(upper_bound_k)], 
                            capture_output=True, 
                            text=True,
                            cwd=os.path.abspath("../conann"),
                            check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Failed conann variable_k run with params: {dataset}, {calib_sz}, {tune_sz}, {alpha}, {nlist}, {lower_bound_k}, {upper_bound_k}\n")
        with open(f"Failed_conann_{dataset}_{calib_sz}_{tune_sz}_{alpha}_{nlist}_variable_k_{lower_bound_k}_{upper_bound_k}.log", "a") as f:
            f.write(f"Error running conann variable_k with params: {dataset}, {calib_sz}, {tune_sz}, {alpha}, {nlist}, {lower_bound_k}, {upper_bound_k}\n")
            f.write(f"Timestamp: {datetime.datetime.now()}\n")
            f.write(f"Return code: {e.returncode}\n")
            f.write(f"stdout: {e.stdout}\n")
            f.write(f"stderr: {e.stderr}\n\n")

def run_faiss(experiment, dataset, calib_sz, nlist, k, starting_nprobe, alphas: tuple):
    print(f"running faiss: experiment={experiment}, dataset={dataset}, calib_size={calib_sz}, nlist={nlist}, k={k}, alphas={alphas}")
    try:
        result = subprocess.run([f"./build/eval/{experiment}", dataset, str(calib_sz), str(nlist), str(k), str(starting_nprobe), *[str(a) for a in alphas]], 
                            capture_output=True, 
                            text=True,
                            cwd=os.path.abspath("../faiss-1.9.0"),
                            check=True)
        print(result.stdout)
        # return int(result.stdout.splitlines()[-1])
    except subprocess.CalledProcessError as e:
        print(f"Failed faiss run with params: {dataset}, {calib_sz}, {nlist}, {k}, {starting_nprobe}, {alphas}\n")
        with open(f"Failed_faiss_{dataset}_{calib_sz}_{nlist}_{k}_{starting_nprobe}.log", "a") as f:
            f.write(f"Error running faiss with params: {dataset}, {calib_sz}, {nlist}, {k}, {starting_nprobe}, {alphas}\n")
            f.write(f"Timestamp: {datetime.datetime.now()}\n")
            f.write(f"Return code: {e.returncode}\n")
            f.write(f"stdout: {e.stdout}\n")
            f.write(f"stderr: {e.stderr}\n\n")
        
def run_faiss_variable_k(dataset, calib_sz, nlist, lower_bound_k, upper_bound_k, starting_nprobe, alpha):
    print(f"running faiss variable k: experiment=variable_k, dataset={dataset}, calib_size={calib_sz}, nlist={nlist}, k={lower_bound_k}, k={upper_bound_k}, alpha={alpha}")
    try:
        result = subprocess.run([f"./build/eval/variable_k", dataset, str(calib_sz), str(nlist), str(lower_bound_k), str(upper_bound_k), str(starting_nprobe), str(alpha)], 
                            capture_output=True, 
                            text=True,
                            cwd=os.path.abspath("../faiss-1.9.0"),
                            check=True)
        print(result.stdout)
        # return int(result.stdout.splitlines()[-1])
    except subprocess.CalledProcessError as e:
        print(f"Failed faiss variable k run with params: {dataset}, {calib_sz}, {nlist}, {lower_bound_k}, {upper_bound_k}, {starting_nprobe}, {alpha}\n")
        with open(f"Failed_faiss_variable_k_{dataset}_{calib_sz}_{nlist}_{lower_bound_k}_{upper_bound_k}_{starting_nprobe}.log", "a") as f:
            f.write(f"Error running faiss variable k with params: {dataset}, {calib_sz}, {nlist}, {lower_bound_k}, {upper_bound_k}, {starting_nprobe}, {alphas}\n")
            f.write(f"Timestamp: {datetime.datetime.now()}\n")
            f.write(f"Return code: {e.returncode}\n")
            f.write(f"stdout: {e.stdout}\n")
            f.write(f"stderr: {e.stderr}\n\n")


def run_auncel(experiment, dataset, calib_sz, k, alphas: tuple):
    print(f"running Auncel: experiment={experiment}, dataset={dataset}, calib_size={calib_sz}, k={k}, alphas={alphas}")
    try:
        result = subprocess.run([f"./build/eval/{experiment}", dataset, str(k), str(calib_sz), *[str(a) for a in alphas]], 
                            capture_output=True, 
                            text=True,
                            cwd=os.path.abspath("../Auncel"),
                            check=True)
        print(result.stdout)
        # return int(result.stdout.splitlines()[-1])
    except subprocess.CalledProcessError as e:
        print(f"Failed Auncel run with params: {dataset}, {k}, {calib_sz}, {alphas}\n")
        with open(f"Failed_auncel_{dataset}_{k}_{calib_sz}_{alphas}.log", "a") as f:
            f.write(f"Error running auncel with params: {dataset}, {k}, {calib_sz}, {alphas}\n")
            f.write(f"Timestamp: {datetime.datetime.now()}\n")
            f.write(f"Return code: {e.returncode}\n")
            f.write(f"stdout: {e.stdout}\n")
            f.write(f"stderr: {e.stderr}\n\n")


# GLOBAL PARAMETERS:
calib_sz = 0.5
tuning_sz = {"bert": 0.2, "glove": 0.1, "sift1M": 0.1, "deep10M": 0.1, "gist": 0.1, "fasttext": 0.1}
nlist = {"bert": 128, "glove": 1024, "sift1M": 1024, "deep10M": 1024, "gist": 1024, "fasttext": 1024}
faiss_starting_nprobe = 35

# sanity check
datasets = ("bert","sift1M", "deep10M", "gist",)
for dataset in datasets:
    test_query_size(dataset)

"""
* NOTE: Number of queries in the current data folder:
* bert: 10000
* glove: 10000
* sift1M: 10000
* deep10M: 10000
* gist: 1000 (small to account for limited memory)
"""

# Possible command combo to sample new dataset:
# First prepare folder at ./data/gist with gist_base.fvecs inside
# sample_dataset("./data/glove/db.fvecs", 10000, "queries.fvecs")
# compute_gt("glove", (1000, 100, 10))

# sample_dataset("./data/sift1M/sift_base.fvecs", 10000, "queries.fvecs")
# compute_gt("sift1M", (1000, 100, 10))

# sample_dataset("./data/gist/gist_base.fvecs", 10000, "queries.fvecs")
# compute_gt("gist", (1000, 100, 10))

# sample_dataset("./data/fasttext/db.fvecs", 10000, "queries.fvecs")
# compute_gt("fasttext", (1000, 100, 10))

"""
Experiment run for validity, efficiency, adaptivity and calibration times
in the special case of variable k where k is randomly sampled for each query
between the lower_bound and upper_bound.
"""
# Only works on the variable-k git branch
# datasets = ("bert","sift1M", "deep10M")
# upper_bounds = (10, 100, 1000)
# for dataset in datasets:
#     for upper_bound in upper_bounds:
#         run_faiss_variable_k(dataset, calib_sz, nlist[dataset], 1, upper_bound, faiss_starting_nprobe, 0.1)
# exit(0) # <-- TAKE CARE

"""
Experiment run for Auncel validity checks.
(figure 5)
"""
datasets = ("sift1M", "deep10M",)
alphas = (0.5,0.4,0.3,0.2,0.1,0.05,)
ks = (10,100,)
for dataset in datasets:
    for k in ks:
       run_auncel("effect_error", dataset, calib_sz, k, alphas)

"""
Main experiment run for validity, efficiency, adaptivity and calibration times.
(figures: 6, 7, 8; table: 5)
"""
datasets = ("bert","sift1M", "deep10M", "gist", "fasttext",)
alphas = (0.5,0.4,0.3,0.2,0.1,0.05,)
ks = (10,100,1000,)
for dataset in datasets:
    for k in ks:
        for alpha in alphas:
            run_conann("error", dataset, calib_sz, tuning_sz[dataset], alpha, nlist[dataset], k)
        run_faiss("error", dataset, calib_sz, nlist[dataset], k, faiss_starting_nprobe, alphas)

"""
Experiment run for latency.
(figure 9)
"""
datasets = ("deep10M", "gist",)
alphas = (0.1,)
ks = (10,100,1000,)
for dataset in datasets:
    for k in ks:
        for alpha in alphas:
            run_conann("latency", dataset, calib_sz, tuning_sz[dataset], alpha, nlist[dataset], k)
        run_faiss("latency", dataset, calib_sz, nlist[dataset], k, faiss_starting_nprobe, alphas)

"""
Experiment run for the comparison of different P values.
(table 4)
"""
datasets = ("bert","sift1M", "deep10M", "gist", "fasttext",)
alphas = (0.1,)
ks = (100,)
nlist_comparison = (512, 768, 1024, 1536, 2048,)
# Section running nlist comparisons.
for dataset in datasets:
    for clusters in nlist_comparison:
        for k in ks:
            for alpha in alphas:
                run_conann("error", dataset, calib_sz, tuning_sz[dataset], alpha, clusters, k)
            run_faiss("error", dataset, calib_sz, clusters, k, faiss_starting_nprobe, alphas)

"""
Experiment run with active vector compression using PQ.
(figure 10)
"""
datasets = ("sift1M",)
alphas = (0.5,0.6,0.7,0.8,0.9,)
ks = (100,)
for dataset in datasets:
    for k in ks:
        for alpha in alphas:
            run_conann("error_pq", dataset, calib_sz, tuning_sz[dataset], alpha, nlist[dataset], k)
        run_faiss("error_pq", dataset, calib_sz, nlist[dataset], k, faiss_starting_nprobe, alphas)
