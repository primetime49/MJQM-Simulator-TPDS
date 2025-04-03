## Abstract
Modern data centers feature an extensive array of cores that handle quite a diverse range of jobs. Recent traces, shared by leading cloud data center enterprises like Google and Alibaba, reveal that the constant increase in data center services and computational power is accompanied by a growing variability in service demand requirements. The number of cores needed for a job can vary widely, ranging from one to several thousand, and the number of seconds a core is held by a job can span more than five orders of magnitude. In this context of extreme variability, the policies governing the allocation of cores to jobs play a crucial role in the performance of data centers. It is widely acknowledged that the First-In First-Out (FIFO) policy tends to underutilize available computing capacity due to the varying magnitudes of core requests. However, the impact of the extreme variability in service demands on job waiting and response times, which has been deeply investigated in traditional queuing models, is not as well understood in the case of data centers, as we will show. To address this issue, we investigate the dynamics of a data center cluster through analytical models in simple cases, and discrete event simulations based on real data. 

The simulations helped us understand some aspects that are too complex to be analyzed through mathematical models. Using simulation, we are also able to emulate the behavior of real-world datacenters. The figures present in the paper are generated from simulation results, which were performed with different parameters and service time distributions. The results were parsed and plotted into figures using the _Matplotlib_ library in Python.

## Requirements
### Software requirements
- C++14/17/20
- Python 3

### Hardware requirements
We don't specify the hardware requirements for our simulations. But to give an estimation, we run our simulations to get the results published in the paper on a cluster node composed of 20 core Intel(R) Xeon(R) Gold 6148 CPU @ 2.40 GHz, 200 GB of ECC RAM. Storage is on a 30 TB NAS, and everything is hosted on a Nutanix hyperconvergent architecture. For the simulation involving bounded pareto service time distribution, we run 100 million events with 60 independent runs. While for other service time distributions, we run 30 million events with 40 independent runs. The independent runs are done to compute the confidence interval for each metric we captured. Each independent runs take approximately 3 minutes. It is possible to change the number of events and/or independent runs by modifying the given .sh scripts, but results may become less reliable.

## Experiment workflow

### Option 1: Only do the plotting from pre-saved simulation results
1. `cd Only_Plotting`
2. `chmod +x script-figures.sh`
3. `./script-figures.sh`

The plots will be generated inside folder [Figures](Only_Plotting/Figures/)

### Option 2: Do all the simulations first, then do the plotting
1. `cd Sim_and_Plotting`
2. `chmod +x script-figures.sh res_Fig3a.sh res_others.sh generate_all.sh`
3. `./generate_all.sh`

The simulation results will be generated inside foler [Results](Sim_and_Plotting/Results/), while the plots will be generated inside folder [Figures](Sim_and_Plotting/Figures/)
