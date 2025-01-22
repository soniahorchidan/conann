# Roadmap:

- [ ] run current Mondrian implementation on DEEP1M dataset to test larger calibration size
- [ ] generate ground truths for GLOVE
- [ ] [optional] extend GTs for GIST
- [ ] generate GTs for k=1000 for all datasets
- [ ] think about variable k - can optimization for larger k subsume smaller k?
- [ ] setup scripts to run all experiments



# How to build without GPU


Faiss:

```
cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -B build .
make -C build -j faiss
```

ConANN:

```
cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -B build .
make -C build -j demo_conann
```
