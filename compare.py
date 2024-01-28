import random
import os
import time
import sys


def compile_binary(case_name, perform_int, suffix):
    print("\tCreating a fresh copy of ViennaRNA")
    os.system("cd Source; rm -rf ViennaRNA-2.5.1; tar -xvf ViennaRNA-2.5.1.tar.gz > /dev/null; cd ..;")

    print("\tAdding Code for", case_name, "and compiling")
    SIMD_flag=""
    if (case_name == "Original"):
        print("\t\tNo Changes needed")
    elif (case_name == "OptimizedSSE"):
        os.system("cd Source/Optimized_Code; cp -r mfe.c loops utils ../ViennaRNA-2.5.1/src/ViennaRNA/; cd ../ViennaRNA-2.5.1/src/ViennaRNA/utils; mv sse4higher_order_functions.c higher_order_functions.c;")
    elif (case_name == "OptimizedAVX2"):
        os.system("cd Source/Optimized_Code; cp -r mfe.c loops utils ../ViennaRNA-2.5.1/src/ViennaRNA/; cd ../ViennaRNA-2.5.1/src/ViennaRNA/utils; mv avx2higher_order_functions.c higher_order_functions.c;")
    elif (case_name == "NoSIMD"):
        SIMD_flag="--disable-simd"
    else:
        print("Selected an incorrect option")
        exit()


    if (perform_int == "NoInternal"):
        os.system("cp Source/Optimized_Code/no_internal.c Source/ViennaRNA-2.5.1/src/ViennaRNA/loops/internal.c;")


    print("\tCompiling ViennaRNA Package")
    compiled_folder=case_name+suffix
    working_dir=os.getcwd()


    os.system("cd Source/ViennaRNA-2.5.1; ./configure --without-perl --without-python --without-swig "+SIMD_flag+" --prefix="+working_dir+"/Compiled/"+compiled_folder+"; make install -j16;")

    
    #compile_command="./compile.sh {};".format(case_name+suffix)
    #os.system(compile_command)






perform_int=sys.argv[1]
suffix=""
if (perform_int == "Internal"):
    print("Perform RNAfold with Internal Loops Computation")
    suffix=""
elif (perform_int == "NoInternal"):
    print("Perform RNAfold without Internal Loops Computation")
    suffix="_NoInt"
else:
    print("Incorrect Option Selected. Please set the first argument either as 'Internal' or 'NoInternal'")
    exit()    

first_case=sys.argv[2]
second_case=sys.argv[3]
allPossibleCases=("Original", "OptimizedSSE", "OptimizedAVX2", "NoSIMD")
if (first_case not in allPossibleCases):
    print("The first case is not a valid choice. Please select one of the following:", allPossibleCases)
    exit()
if (second_case not in allPossibleCases):
    print("The second case is not a valid choice. Please select one of the following:", allPossibleCases)
    exit()

if (os.path.isdir("Compiled/"+first_case+suffix)):
    print("Compiled Binaries for the first case already exists")
else:
    print("Compiling Binaries for",first_case)
    compile_binary(first_case, perform_int, suffix)

if (os.path.isdir("Compiled/"+second_case+suffix)):
    print("Compiled Binaries for the second case already exists")
else:
    print("Compiling Binaries for",second_case)
    compile_binary(second_case, perform_int, suffix)

first_case=first_case+suffix
second_case=second_case+suffix

print("Performing tests on", first_case, "and", second_case)
os.system("python3 test.py "+ first_case +" "+ second_case)


 
