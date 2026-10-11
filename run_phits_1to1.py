import argparse
import os
import sys
import time
import subprocess
from pathlib import Path
import pandas as pd

from generate_inputs import _parse_thickness_list, _fmt_thickness, SAMPLE_TEMPLATE, BLANK_TEMPLATE, COMPOSITION, DENSITY, format_material
import gdrive_helper

def parse_track_flux(filepath: Path) -> tuple:
    lines = filepath.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5 and parts[0] == "1" and parts[1] == "104":
            try:
                flux = float(parts[-2])
                err = float(parts[-1])
                return flux, err
            except ValueError:
                pass
    raise ValueError(f"Not found in {filepath}")

def build_my_jobs(checklist_csv: str, runner_id: int, num_runners: int, output_base: Path, omp: int, maxcas: int):
    df = pd.read_csv(checklist_csv)
    output_base.mkdir(parents=True, exist_ok=True)
    
    my_jobs = []
    for i, row in df.iterrows():
        if i % num_runners != (runner_id - 1): continue
        
        e_mev = float(row['energy_mev'])
        energy_kev = int(round(e_mev * 1000.0))
        sample = str(row['sample']).upper()
        th = float(row['thickness_cm'])
        
        e0_mev = energy_kev / 1000.0
        emax_mev = e0_mev * 1.05
        emin_peak = e0_mev * 0.990
        emax_peak = e0_mev * 1.010
        
        if sample == "BLANK":
            job_name = f"blank_{energy_kev}keV"
            z_sample_back = 50.0 + th
            jdir = output_base / job_name
            jdir.mkdir(parents=True, exist_ok=True)
            jinp = jdir / f"{job_name}.inp"
            content = BLANK_TEMPLATE.format(
                energy_kev=energy_kev, e0_mev=e0_mev, emax_mev=emax_mev, emin_peak=emin_peak, emax_peak=emax_peak,
                omp=omp, maxcas=maxcas, thickness_str=_fmt_thickness(th), z_sample_back=z_sample_back,
            )
            jinp.write_text(content, encoding="utf-8")
            my_jobs.append({"name": job_name, "dir": jdir, "inp": jinp, "energy_mev": e_mev, "sample": sample, "thickness_cm": th})
        else:
            job_name = f"{sample}_{energy_kev}keV_{_fmt_thickness(th)}cm"
            z_sample_back = 50.0 + th
            jdir = output_base / job_name
            jdir.mkdir(parents=True, exist_ok=True)
            jinp = jdir / f"{job_name}.inp"
            mat_dict = COMPOSITION.get(sample.lower(), {})
            rho = DENSITY.get(sample.lower(), 1.0)
            mat_r1_str = format_material(mat_dict)
            content = SAMPLE_TEMPLATE.format(
                sample_name=sample, energy_kev=energy_kev, thickness_str=_fmt_thickness(th),
                e0_mev=e0_mev, emax_mev=emax_mev, emin_peak=emin_peak, emax_peak=emax_peak,
                omp=omp, maxcas=maxcas, rho=rho, mat_r1_str=mat_r1_str, z_sample_back=z_sample_back
            )
            jinp.write_text(content, encoding="utf-8")
            my_jobs.append({"name": job_name, "dir": jdir, "inp": jinp, "energy_mev": e_mev, "sample": sample, "thickness_cm": th})
            
    return my_jobs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-id", type=int, required=True)
    parser.add_argument("--num-runners", type=int, default=20)
    parser.add_argument("--primaries", type=int, default=100000000)
    parser.add_argument("--gdrive-folder", type=str, required=True)
    parser.add_argument("--checklist", type=str, default="missing_ph.csv")
    parser.add_argument("--omp", type=int, default=2)
    parser.add_argument("--github-user", type=str, default="local")
    args = parser.parse_args()
    
    drive_service = gdrive_helper.get_drive_service()
    subfolders = gdrive_helper.get_or_create_subfolders(drive_service, parent_id=args.gdrive_folder)
    flux_folder_id = subfolders.get("02_Flux_Data")
    
    q = f"'{flux_folder_id}' in parents and trashed = false"
    res = drive_service.files().list(q=q, fields="files(id, name)", pageSize=1000).execute()
    existing_files = {f['name'] for f in res.get('files', [])}
    
    work_dir = Path("phits_work")
    my_jobs = build_my_jobs(args.checklist, args.runner_id, args.num_runners, work_dir / "inputs", args.omp, args.primaries)
    
    for job in my_jobs:
        # Expected CSV name exactly matching Master Grid formats so aggregation is trivial
        csv_name = f"{job['sample']}_{job['energy_mev']:.3f}_{job['thickness_cm']:.3f}.csv"
        
        if csv_name in existing_files:
            print(f"[{csv_name}] ALREADY ON DRIVE. SKIPPING.")
            continue
            
        print(f"--- RUNNING PHITS: {csv_name} ---")
        t0 = time.time()
        res = subprocess.run(["phits", str(job['inp'].absolute())], cwd=str(job['dir']), capture_output=True, text=True)
        if res.returncode != 0:
            print(f"PHITS ERROR: {res.stderr[-300:]}")
            continue
            
        flux_file = job['dir'] / "flux.out"
        peak, err = parse_track_flux(flux_file) if flux_file.exists() else (0.0, 0.0)
        
        df = pd.DataFrame([{
            "job_name": job['name'],
            "energy_mev": job['energy_mev'],
            "sample": job['sample'],
            "thickness_cm": job['thickness_cm'],
            "peak_flux": peak,
            "peak_err": err,
            "total_flux": peak,
            "total_err": err
        }])
        
        out_csv = job['dir'] / csv_name
        df.to_csv(out_csv, index=False)
        gdrive_helper.upload_file_to_folder(drive_service, str(out_csv), flux_folder_id)
        print(f"Uploaded {csv_name} in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
