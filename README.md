# AMP-R

Code for Approximate Message Passing algorithm with Refinemnt

Preparataion


1) Gurobi 
   - gurobi license is necessary
   - pip install gurobipy

3) LKH
   - pip install lkh
   - and LKH solver should be installed
     
- Ubuntu or Mac
```
wget http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.6.tgz
tar xvfz LKH-3.0.6.tgz
cd LKH-3.0.6
make
sudo cp LKH /usr/local/bin
```


- Window
    - Download exe file (http://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.exe) 


- Example code for LKH
```
import requests
import lkh

problem_str = requests.get('http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/A/A-n32-k5.vrp').text
problem = lkh.LKHProblem.parse(problem_str)

solver_path = '../LKH-3.0.6/LKH'
lkh.solve(solver_path, problem=problem, max_trials=10000, runs=10)
```

- You can change the solver path in Param.py
