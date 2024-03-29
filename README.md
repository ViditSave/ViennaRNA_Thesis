# ViennaRNA_Thesis


## Changelog:
1. Base Code 
2. Correctness Verification (Python framework to verify the 3 computed matrices)
3. Memory Allocation changes (Replace N X 1D arrays with ND arrays)
4. Remove update fms3 and fms5 arrays (Assume single strands)
5. Tiling (Add extra memory to enable tiling)
6. Test Framework (Automates benchmark process)



## Installation Help:

1. Run Testing Framework:
```
python3 compare.py <Internal_Selection> <Bench_1> <Bench_2>

# Internal Selection takes the value "Internal" or "NoInternal"
# Bench_1/2 takes values from the following: "Original" "OptimizedSSE" "OptimizedAVX2" "NoSIMD"
```

2. Test a different branch:
```
#git clone <URL>
#git branch --set-upstream-to=origin/<branch_name> <branch_name>
#git pull
```

3. Older branches:
```
./configure --prefix=/s/chopin/l/grad/vidits/Vienna_Opt/compiled/Vienna_<Branch_Name>/
make install
```

## Github Help:

1. Create Personal Access Token to access Git CLI
2. Get URL to clone Repository
3. Create branch for changes
