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
    """Doc file T-Track (F4 equivalent) trong vung detector (reg 104), tra ve (flux, r_err)."""
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
    raise ValueError(f"Khong tim thay du lieu flux vung reg 104 trong {filepath}")

def setup_phits_environment(tar_path: Path, target_dir: Path) -> tuple:
    target_dir.mkdir(parents=True, exist_ok=True)
    phits_bin = target_dir / "phits"
    if not phits_bin.exists():
        print(f">>> [SETUP] Dang giai nen goi PHITS tu {tar_path}...")
        subprocess.run(["tar", "-zxf", str(tar_path.resolve()), "-C", str(target_dir.resolve())], check=True)
        if not phits_bin.exists():
            matches = list(target_dir.rglob("phits"))
            if matches:
                phits_bin = matches[0]
        subprocess.run(["chmod", "+x", str(phits_bin)], check=True)
    
    print(f">>> [SETUP] PHITS binary san sang tai: {phits_bin}")
    return phits_bin.resolve(), target_dir.resolve()

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
                job_name=job_name,
                mat_upper=sample,
                mat_comp=mat_r1_str,
                sample_name=sample, energy_kev=energy_kev, thickness_str=_fmt_thickness(th),
                e0_mev=e0_mev, emax_mev=emax_mev, emin_peak=emin_peak, emax_peak=emax_peak,
                omp=omp, maxcas=maxcas, density=rho, mat_r1_str=mat_r1_str, z_sample_back=z_sample_back
            )
            jinp.write_text(content, encoding="utf-8")
            my_jobs.append({"name": job_name, "dir": jdir, "inp": jinp, "energy_mev": e_mev, "sample": sample, "thickness_cm": th})
            
    return my_jobs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-id", type=int, required=True)
    parser.add_argument("--num-runners", type=int, default=100)
    parser.add_argument("--primaries", type=int, default=100000000)
    parser.add_argument("--gdrive-folder", type=str, required=True)
    parser.add_argument("--checklist", type=str, default="missing_ph.csv")
    parser.add_argument("--omp", type=int, default=2)
    parser.add_argument("--package-path", type=str, default="phits_pack/phits_package.tar.gz")
    parser.add_argument("--timeout-hours", type=float, default=5.4)
    parser.add_argument("--github-user", type=str, default="local")
    args = parser.parse_args()
    
    drive_service = gdrive_helper.get_drive_service()
    subfolders = gdrive_helper.get_or_create_subfolders(drive_service, parent_id=args.gdrive_folder)
    flux_folder_id = subfolders.get("02_Flux_Data")
    
    q = f"'{flux_folder_id}' in parents and trashed = false"
    res = drive_service.files().list(q=q, fields="files(id, name)", pageSize=1000).execute()
    existing_files = {f['name'] for f in res.get('files', [])}
    
    work_dir = Path("/tmp/phits_work").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    
    phits_bin, phits_root = setup_phits_environment(Path(args.package_path), work_dir / "bin")
    
    my_jobs = build_my_jobs(args.checklist, args.runner_id, args.num_runners, work_dir / "inputs", args.omp, args.primaries)
    
    for job in my_jobs:
        # Expected CSV name exactly matching Master Grid formats so aggregation is trivial
        csv_name = f"{job['sample']}_{job['energy_mev']:.3f}_{job['thickness_cm']:.3f}.csv"
        
        if csv_name in existing_files:
            print(f"[{csv_name}] ALREADY ON DRIVE. SKIPPING.")
            continue
            
        print(f"--- RUNNING PHITS: {csv_name} ---")
        t0 = time.time()
        
        env = os.environ.copy()
        env["PHITSPATH"] = str(phits_root)
        env["OMP_NUM_THREADS"] = str(args.omp)
        
        inp_name = job['inp'].name
        cmd = f'"{phits_bin}" < "{inp_name}"'
        res = subprocess.run(cmd, shell=True, cwd=str(job['dir']), env=env, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"PHITS ERROR / RETURNCODE {res.returncode}: {(res.stderr or res.stdout)[-300:]}")
            
        peak_out = job['dir'] / f"peak_{job['name']}.out"
        track_out = job['dir'] / f"track_{job['name']}.out"
        
        p_val, p_err = (0.0, 0.0)
        t_val, t_err = (0.0, 0.0)
        
        if peak_out.exists():
            try:
                p_val, p_err = parse_track_flux(peak_out)
            except Exception as e:
                print(f"Warning parsing peak: {e}")
                
        if track_out.exists():
            try:
                t_val, t_err = parse_track_flux(track_out)
            except Exception as e:
                print(f"Warning parsing track: {e}")
                
        df = pd.DataFrame([{
            "job_name": job['name'],
            "energy_mev": job['energy_mev'],
            "sample": job['sample'],
            "thickness_cm": job['thickness_cm'],
            "peak_flux": p_val,
            "peak_err": p_err,
            "total_flux": t_val,
            "total_err": t_err
        }])
        
        out_csv = job['dir'] / csv_name
        df.to_csv(out_csv, index=False)
        gdrive_helper.upload_file_to_folder(drive_service, str(out_csv), flux_folder_id)
        existing_files.add(csv_name)
        print(f"Uploaded {csv_name} in {time.time()-t0:.1f}s | Peak: {p_val:.6e}")

if __name__ == "__main__":
    main()
