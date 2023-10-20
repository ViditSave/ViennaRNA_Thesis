import random
import os
import time
import sys
import numpy

def remove_std_dev(orig_list):
    np_array = numpy.array(orig_list)

    mean = numpy.mean(np_array, axis=0)
    sdev = numpy.std(np_array, axis=0)

    clean_list = [elem for elem in orig_list  if (elem > mean - 2 * sdev)]
    clean_list = [elem for elem in clean_list if (elem < mean + 2 * sdev)]

    return clean_list





input_list = ['A', 'C' , 'G', 'U']

# Small/Compile Test
#N_list = [248, 657, 842, 1269]
#N_repeat = 10

# Middle Test
#N_list = [i for i in range(1000,10000,1000)]
N_list = [i for i in range(1000,4000,1000)]
N_repeat = 20

# Large Test
#N_list = [10000, 11000, 12000, 13000, 14000, 15000, 16000]
#N_repeat = 2


original = 'Compiled/' + sys.argv[1]
test = 'Compiled/' + sys.argv[2]

for N in N_list:
    print("Testing for N:",N,"\t for",sys.argv[1],"/",sys.argv[2],"\n\tIntermediate Speedup: ", end="")
    endOrig = []
    endMod = []
    actual_rep = 0
    for repeat in range(N_repeat):
        inp_str = "".join([random.choice(input_list) for i in range(N)])
        os.system("echo "+inp_str+" > input.txt")

        start = time.time()
        os.system("./"+ original +"/bin/RNAfold -i input.txt > f1.out")
        curOrig = time.time() - start

        start = time.time()
        os.system("./"+ test +"/bin/RNAfold -i input.txt > f2.out")
        curMod = time.time() - start
        actual_rep+=1
        if os.system("diff f1.out f2.out"):
            print("Error")
            endOrig.append(curOrig)
            endMod.append(curMod)
            #os.system("echo Original; cat f1.out; echo Modified; cat f2.out")
            break
        else:
            #print("Correct", end="\t")
            print(("{:.3f}".format(curOrig/curMod)), end=", ", flush=True)
            endOrig.append(curOrig)
            endMod.append(curMod)
    
    newEndOrig=remove_std_dev(endOrig)
    newEndMod=remove_std_dev(endMod)
    
    print("\n\tAverage Time:", sum(newEndOrig)/len(newEndOrig), " / ", sum(newEndMod)/len(newEndMod))
    print("\tSpeedup:",sum(newEndOrig)/sum(newEndMod))
    print("\tActualFirst :",endOrig)
    print("\tCleanedFirst:",newEndOrig)
    print("\tActualSecond :",endMod)
    print("\tCleanedSecond:",newEndMod, "\n\n")


os.system("rm f1.out f2.out input.txt rna.ps")


