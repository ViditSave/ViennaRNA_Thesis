# ViennaRNA_Thesis


## Changelog:
1. Base Code 
2. Correctness Verification (Python framework to verify the 3 computed matrices)
3. Tiling (Naive Method - Does not work as rotate_aux_arrays cant be peeled)
4. Memory Allocation changes (Replace N X 1D arrays with ND arrays)
5. Remove update fms3 and fms5 arrays (Assume single strands)
6. Tiling (Add extra memory to enable tiling)


## Installation Help:
```
./configure --prefix=/s/chopin/l/grad/vidits/Vienna_Opt/compiled/Vienna_<Branch_Name>/
make install
```

## Github Help:

1. Create Personal Access Token to access Git CLI
2. Get URL to clone Repository
3. Create branch for changes
