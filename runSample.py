import subprocess 

cmd = "./a.out"

#Just trying to find the sweet spot, when the gpu is faster than the cpu
for i in range(2000,2500): 
    o = subprocess.call([cmd, i])
    print(o.stdout)

