"""
Parse Concorde benchmark data from Waterloo website and update ground truth.
"""

import re
import pandas as pd

# Raw data from the webpage
concorde_data_raw = """
burma14,1,0.06
ulysses16,1,0.22
gr17,1,0.08
gr21,1,0.03
ulysses22,1,0.53
gr24,1,0.07
fri26,1,0.07
bayg29,1,0.09
bays29,1,0.13
dantzig42,1,0.23
swiss42,1,0.13
att48,1,0.56
gr48,1,0.67
hk48,1,0.17
eil51,1,0.73
berlin52,1,0.29
brazil58,1,0.68
st70,1,0.50
eil76,1,0.30
pr76,1,1.86
gr96,1,6.71
rat99,1,0.95
kroA100,1,1.00
kroB100,1,2.36
kroC100,1,0.96
kroD100,1,1.00
kroE100,1,2.44
rd100,1,0.67
eil101,1,0.74
lin105,1,0.59
pr107,1,1.03
gr120,1,2.23
pr124,1,3.64
bier127,1,1.65
ch130,1,2.13
pr136,1,3.97
gr137,1,3.42
pr144,1,2.58
ch150,1,3.03
kroA150,1,5.00
kroB150,1,4.23
pr152,1,7.93
u159,1,1.00
si175,3,13.09
brg180,1,1.46
rat195,5,22.23
d198,3,11.82
kroA200,1,6.59
kroB200,1,3.91
gr202,1,5.01
ts225,1,20.52
tsp225,1,15.01
pr226,1,4.35
gr229,3,38.61
gil262,1,13.06
pr264,1,2.67
a280,3,5.37
pr299,3,17.49
lin318,1,9.74
rd400,15,148.42
fl417,5,57.75
gr431,13,133.29
pr439,15,216.75
pcb442,9,49.92
d493,5,113.32
att532,7,109.52
ali535,3,53.14
si535,3,43.13
pa561,17,246.82
u574,1,23.04
rat575,25,363.07
p654,3,26.52
d657,13,260.37
gr666,3,49.86
u724,11,225.44
rat783,1,37.88
dsj1000,7,410.32
pr1002,1,34.30
si1032,1,25.47
u1060,21,571.43
vm1084,11,604.78
pcb1173,19,468.27
d1291,45,27393.72
rl1304,1,189.20
rl1323,25,3742.25
nrw1379,19,578.42
fl1400,5,1548.51
u1432,3,223.70
fl1577,7,6705.04
d1655,5,263.03
vm1748,17,2223.65
u1817,887,449230.55
rl1889,83,10023.02
d2103,169,11179253.91
u2152,309,45204.53
u2319,13,7067.93
pr2392,1,116.86
pcb3038,313,80828.87
fl3795,21,69886.48
fnl4461,213,53420.13
rl5915,161,2319671.71
rl5934,205,588936.85
pla7397,101,428996.2
"""

# Parse the data
instances = []
for line in concorde_data_raw.strip().split("\n"):
    parts = line.split(",")
    name = parts[0]
    nodes = int(parts[1])
    time_s = float(parts[2])
    instances.append(
        {
            "instance": name,
            "concorde_nodes": nodes,
            "concorde_time_s": time_s,
            "solver": "Concorde 99.12.15",
            "machine": "Compaq XP1000 (500 MHz)",
        }
    )

df = pd.DataFrame(instances)
df.to_csv("concorde_benchmarks.csv", index=False)
print(f"Saved {len(df)} Concorde benchmark entries")
print(df.head(20))
