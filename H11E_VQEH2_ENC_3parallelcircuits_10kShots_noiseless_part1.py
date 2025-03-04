# [[4,2,2]] implementation for hydrogen

import xacc

# import module with noise model
import noise_generation as ng
import json
import math
import numpy as np
import pickle
from collections import Counter
from pathlib import Path
import pickle
import os


import time

start_time = time.time()
# trial number 
run = 1    # verify

# part number for file 
n=1    # verify

# shots
shots = 10000  # verify

# [[4,2,2]] encoded hydrogen for optimal theta scanned by brute force VQE
    
# Create empty dictionary to collect measurement counts and coefficients of the Pauli 
# terms for every parameter scanned by vqe
buffer_info = {}

# [[4,2,2]] encoded hydrogen ansatz, q0 is ancilla 1 (post selection for incorrect state prep) 
# and q5 is ancilla 2 (for teleported rotation gate). 

# parameter value
t0 = -0.22967  # verify
# t0 = -0.23193

# Create dictionary within buffer_info for each parameter
buffer_info[t0] = {}

# circuits for x2x3 and zz Pauli terms executed for each point in the specified 

xacc.qasm(f'''
.compiler xasm
.circuit encoded
//.parameters t0
.qbit q
H(q[1]);
CX(q[1], q[0]);
CX(q[1], q[2]);
CX(q[1], q[3]);
CX(q[1], q[4]);
H(q[5]);
CX(q[5],q[2]);
CX(q[5],q[3]);
Ry(q[5], {-t0});
CX(q[1], q[0]);
Measure(q[0]);
Measure(q[5]);
Measure(q[1]);
Measure(q[2]);
Measure(q[3]);
Measure(q[4]);
H(q[7]);
CX(q[7], q[6]);
CX(q[7], q[8]);
CX(q[7], q[9]);
CX(q[7], q[10]);
H(q[11]);
CX(q[11],q[8]);
CX(q[11],q[9]);
Ry(q[11], {-t0});
CX(q[7], q[6]);
Measure(q[6]);
Measure(q[11]);
Measure(q[7]);
Measure(q[8]);
Measure(q[9]);
Measure(q[10]);
H(q[13]);
CX(q[13], q[12]);
CX(q[13], q[14]);
CX(q[13], q[15]);
CX(q[13], q[16]);
H(q[17]);
CX(q[17],q[14]);
CX(q[17],q[15]);
Ry(q[17], {-t0});
CX(q[13], q[12]);
Measure(q[12]);
Measure(q[17]);
Measure(q[13]);
Measure(q[14]);
Measure(q[15]);
Measure(q[16]);
''')

# adding hadamards to change basis of all qubits for X Pauli operator measurements
xacc.qasm(f'''
.compiler xasm
.circuit encodedx2x3
//.parameters t0
.qbit q
H(q[1]);
CX(q[1], q[0]);
CX(q[1], q[2]);
CX(q[1], q[3]);
CX(q[1], q[4]);
H(q[5]);
CX(q[5],q[2]);
CX(q[5],q[3]);
Ry(q[5], {-t0});
CX(q[1], q[0]);
Measure(q[0]);
Measure(q[5]);
H(q[1]);
H(q[2]);
H(q[3]);
H(q[4]);
Measure(q[1]);
Measure(q[2]);
Measure(q[3]);
Measure(q[4]);
H(q[7]);
CX(q[7], q[6]);
CX(q[7], q[8]);
CX(q[7], q[9]);
CX(q[7], q[10]);
H(q[11]);
CX(q[11],q[8]);
CX(q[11],q[9]);
Ry(q[11], {-t0});
CX(q[7], q[6]);
Measure(q[6]);
Measure(q[11]);
H(q[7]);
H(q[8]);
H(q[9]);
H(q[10]);
Measure(q[7]);
Measure(q[8]);
Measure(q[9]);
Measure(q[10]);
H(q[13]);
CX(q[13], q[12]);
CX(q[13], q[14]);
CX(q[13], q[15]);
CX(q[13], q[16]);
H(q[17]);
CX(q[17],q[14]);
CX(q[17],q[15]);
Ry(q[17], {-t0});
CX(q[13], q[12]);
Measure(q[12]);
Measure(q[17]);
H(q[13]);
H(q[14]);
H(q[15]);
H(q[16]);
Measure(q[13]);
Measure(q[14]);
Measure(q[15]);
Measure(q[16]);
''')
encoded = xacc.getCompiled('encoded')
encodedx2x3 = xacc.getCompiled('encodedx2x3')

# with open('422circuit1.txt', 'w') as f:
#     f.write(json.dumps(encoded.toString()))
# with open('422circuitx2x3.txt', 'w') as f:
#     f.write(json.dumps(encodedx2x3.toString()))

# Hamiltonian specified for radius = 0.75Å
Ham = '-3.49833E-01 - 3.88748E-01 Z1Z2 + 1.81771E-01 X2X3 - 3.88748E-01 Z1Z3 + 1.11772E-02 Z2Z3'
# dictionary to store measurement counts for all pauli terms with their coefficients
Pauli_H = {'I':{'coefficient': -3.49833E-01}, 'z1z2':{'coefficient': - 3.88748E-01}, 'x2x3': {'coefficient': 1.81771E-01}, 
               'z1z3': {'coefficient': - 3.88748E-01}, 'z2z3': {'coefficient': 1.11772E-02}}
r = '0.75Å'


# allocate qubits based on the number required to run the ansatz
q = xacc.qalloc(18)
qx2x3 = xacc.qalloc(18)

# qpu and options for qpu
options = {"error-model":False, "no-opt": True}
qpu = xacc.getAccelerator('honeywell:H1-1E', {'shots':shots, 
                                              'job-name':f'Enc noiseless, no-opt:True, part {n}',
                                              "options": json.dumps(options)})    # verify

qpu_details = {'qpu_name': 'honeywell:H1-1E', 'shots':shots, 'job-name': f'Enc noiseless, no-opt:True, part {n}', "options": json.dumps(options),
                  'ansatzX': encodedx2x3.toString(), 'ansatzZ':encoded.toString()}  # verify - matches qpu
# execute each parametrized circuit on same qpu but respective buffers 
qpu.execute(q, encoded)
qpu.execute(qx2x3, encodedx2x3)
#         print(q.getMeasurementCounts(), qx2x3.getMeasurementCounts())
#             for point in points:
for key in Pauli_H.keys():
    if key != 'x2x3' and key != 'I':
        # Counts from parallel circuit simulation
        Pauli_H[key]['OriginalCounts'] = q.getMeasurementCounts()
        # empty dictionary for counts after circuits are subsequently separated
        Pauli_H[key]['MeasurementCounts'] = {}
        Pauli_H[key]['correctCounts'] = {}

Pauli_H['I']['MeasurementCounts'] = {}
# empty dictionary for counts from parallel circuit simulation
Pauli_H['I']['OriginalCounts'] = {}
# empty dictionary for counts after circuits are subsequently separated
Pauli_H['x2x3']['MeasurementCounts'] = {}

# Dicitionary to collect counts of parallel circuit simulation
Pauli_H['x2x3']['OriginalCounts'] = qx2x3.getMeasurementCounts()

# test for right measurment counts going to the right pauli term
#         print(Pauli_H['z2z3']['MeasurementCounts']==q.getMeasurementCounts(), 
#               Pauli_H['z1z3']['MeasurementCounts']==q.getMeasurementCounts(),
#              Pauli_H['z1z2']['MeasurementCounts']==q.getMeasurementCounts(),
#              Pauli_H['x2x3']['MeasurementCounts']==qx2x3.getMeasurementCounts(),
#              Pauli_H['I']['MeasurementCounts'] == {}
#              )
buffer_info[t0] = Pauli_H

q.resetBuffer()
qx2x3.resetBuffer()

# path for qpu details
path = "/home/dev/local/Encoded_VQE_H2/422 encoding /Quantinuum_H1-1E/H1-1E_simulation_files"

# path for results counts
path_counts = f"/home/dev/local/Encoded_VQE_H2/422 encoding /Quantinuum_H1-1E/H1-1E_simulation_files/encoded_counts_noiseless_{run}/"
pickle.dump(buffer_info, open(os.path.join(path_counts, f"H1-1E_VQEH2_Enc_3parallelCircs_{shots}shots_run{run}_part{n}.pkl"), "wb"))
pickle.dump(qpu_details, open(os.path.join(path, f"H1-1E_VQEH2_Enc_QPUdetails_3parallelCircs__{shots}shots_run{run}_part{n}.pkl"), "wb"))    
print("--- %s seconds ---" % (time.time() - start_time))
