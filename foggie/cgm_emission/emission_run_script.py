import subprocess
from multiprocessing import Pool

# List of halos and resolutions
halos = ['4123'] #['8508','5036','5016','4123','2392']
resolutions = [2.7,4.5,5.7,6]#[round(r, 2) for r in [i * 0.3 for i in range(1, int(10 / 0.3) + 1)]]

# Function to run a single job
def run_job(args):
    halo, res = args
    cmd = [
        "python", "emission_mass_maps_dynamic.py",
        "--halo", str(halo),
        "--output", "RD0042",
        "--system", "vida_local",
        "--plot", "emission_FRB",
        "--ions", "HI,CII,CIII,CIV,OVI,MgII,SiII,SiIII,SiIV",
        "--res_kpc", str(res),
        "--fov_kpc", "100",
        "--scale_factor", "10"
    ]
    print(f"Running halo {halo}, res {res}...")
    subprocess.run(cmd)

# Prepare jobs
job_list = [(halo, res) for halo in halos for res in resolutions]

if __name__ == "__main__":
    with Pool(processes=4) as pool:  # Adjust to your number of available cores
        pool.map(run_job, job_list)

