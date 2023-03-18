import os
import time 
import random
import sys

cwd = os.getcwd()
rnaChar=["A","C","G","U"]

#compiled_dirs = ["compiled/Vienna_Unmodified/bin","compiled/Vienna_Blocked/bin"]
#compiled_dirs = ["compiled/Vienna_Unmodified/bin","compiled/Vienna_PeelFms3/bin"]
compiled_dirs = ["compiled/Vienna_Unmodified/bin","compiled/Vienna_PeelFms5_only/bin"]



#for N in [50,100,150,200,250,500,1000]:

for N in [1455,1500,1625,1985,2000,2500]:
    N = str(N)
    print('-'*50 + '\n\n  Processing N:'+ N+ '\n')
    
    matrix_out = []
    correct = 0
    incorrect = 0

    for repeat in range(5):
        text_file = open("logfile.txt", "w")
        text_file.write(''.join(random.choice(rnaChar) for i in range(int(N))))
        text_file.close()
			
        start = time.time()
        try:
            unmodified_out = os.popen(compiled_dirs[0]+"/RNAfold -i logfile.txt").read()
            unmodified_out = unmodified_out.split("The elapsed time in HAIRPIN")[0]
        except:
            print("Failed to perform folding on Unmodified code")
        end = time.time()
        exec_time_unmod = str(end-start)
        
        start = time.time()
        try:
            modified_out = os.popen(compiled_dirs[1]+"/RNAfold -i logfile.txt").read()
            modified_out = modified_out.split("The elapsed time in HAIRPIN")[0]
        except:
            print("Failed to perform folding on Modified code")
        end = time.time()
        exec_time_mod = str(end-start)

        if (unmodified_out == modified_out):
            print("Correct: ", (exec_time_unmod, exec_time_mod))
            correct += 1
        else:
            mod_out = modified_out.split("Matrix")
            unm_out = unmodified_out.split("Matrix")
            print("Incorrect Output")
            incorrect += 1
            if matrix_out == []:
                for index in range(min(len(mod_out), len(unm_out))):
                    matrix_out.append(0)
            for index in range(min(len(mod_out), len(unm_out))):
                #print("\t",index,mod_out[index] == unm_out[index])
                if mod_out[index] != unm_out[index]:
                    matrix_out[index] += 1

    print("\nSummary for N =",N,":")
    print("\tCorrect", correct)
    print("\tIncorrect", incorrect)
    print("\tIncorrect C, fML, fM1&Energy", matrix_out)

os.remove("logfile.txt")
#os.remove("rna.ps")
