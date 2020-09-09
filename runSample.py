import subprocess 

cmd = "./a.out"

#Just trying to find the sweet spot, when the gpu is faster than the cpu
for i in range(200,1000): 
    o = subprocess.run([cmd, i], text=True, capture_output=True)
    print(o.stdout)

