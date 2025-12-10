import csv
import os
from pathlib import Path

def main():
    base_dir = Path(__file__).parent
    files = [
        base_dir / 'final_data/Experiment_1/Scale_0.20/evaluation_0.2.csv',
        base_dir / 'final_data/Experiment_1/Scale_0.50/evaluation_0.5.csv',
        base_dir / 'final_data/Experiment_1/Scale_0.90/evaluation_0.9.csv'
    ]
    scales = [0.2, 0.5, 0.9]
    output_file = base_dir / 'psnr_results.txt'

    # Clear previous results
    if output_file.exists():
        output_file.unlink()

    for f, s in zip(files, scales):
        if not f.exists():
            print(f"File not found: {f}")
            continue
            
        try:
            with open(f, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                psnrs = []
                for row in reader:
                    try:
                        psnrs.append(float(row['psnr_y']))
                    except ValueError:
                        continue
                
                with open(output_file, 'a') as outfile:
                    if psnrs:
                        avg_psnr = sum(psnrs) / len(psnrs)
                        outfile.write(f"Scale {s}: Mean PSNR_Y = {avg_psnr:.4f}\n")
                        print(f"Scale {s}: Mean PSNR_Y = {avg_psnr:.4f}")
                    else:
                        outfile.write(f"Scale {s}: No valid PSNR data found.\n")
                        print(f"Scale {s}: No valid PSNR data found.")
        except Exception as e:
            with open(output_file, 'a') as outfile:
                outfile.write(f"Error reading {f}: {e}\n")
            print(f"Error reading {f}: {e}")

if __name__ == "__main__":
    main()
