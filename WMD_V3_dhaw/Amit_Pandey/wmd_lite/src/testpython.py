print("this is a test python notebook to be run using bash file \n")

import numpy as np
from multiprocessing import Pool
import json
import time

st = time.time()

dicti = {}

def fun(i):
    print(f"processing the i = {i} \n")
    dicti[i] = np.arange(i)
    
    return dicti
    
    
with Pool(10) as p:
      a = p.map(fun,np.arange(50,dtype=float))




print(" printing the output of map function : ",a ,"\n")


et = time.time()

print("time taken is : ", et-st)

a_file = open("testdic.json", "w")
json.dump(a, a_file)
a_file.close()

a_file = open("testdic.json", "r")
output = a_file.read()
print(output)

a_file.close()
