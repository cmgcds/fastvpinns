import os
import sys
from pathlib import Path
import subprocess
import yaml

if __name__ == "__main__":

    # throw error if the input_script.yaml file is not provided
    if len(sys.argv) != 2:
        print("[ERROR] : Please provide the input_script.yaml file")
        exit(1)

    # read the input_script.yaml file
    with open(sys.argv[1], 'r') as f:
        config = yaml.safe_load(f)

    # Experiment name
    experiment_name = "ICCFD_Kovasznay"

    # N_cells_x
    N_cells_x = [1, 3, 6, 12]
    N_cells_y = [1, 4, 8, 16]

    for n_cell_x, n_cell_y in zip(N_cells_x, N_cells_y):
        print(
            "===================================================================================================="
        )
        print(
            "===================================================================================================="
        )
        print(
            " Running the experiment for n_cell_x = {} and n_cell_y = {}".format(n_cell_x, n_cell_y)
        )
        print(
            "===================================================================================================="
        )
        print(
            "===================================================================================================="
        )

        config["experimentation"]["output_path"] = f"output/convergence_{n_cell_x}x{n_cell_y}"
        config["geometry"]["internal_mesh_params"]["n_cells_x"] = n_cell_x
        config["geometry"]["internal_mesh_params"]["n_cells_y"] = n_cell_y

        config["wandb"]["project_name"] = experiment_name
        config["wandb"][
            "wandb_run_prefix"
        ] = f"FastVPINNs_Grid_Convergence_Updated_{n_cell_x}x{n_cell_y}"

        total_cells = n_cell_x * n_cell_y

        epochs = 6000 + 600 * total_cells

        config["model"]["epochs"] = epochs

        output_file = f"logs/convergence_{n_cell_x}x{n_cell_y}.log"

        with open("temp.yaml", 'w') as f:
            yaml.dump(config, f)

        # Run the experiment, Span a subprocess and wait for that subprocess to complete
        # Create a file name under experiment_logs_folder, with name as the experiment name and the run prefix
        with open(output_file, 'w') as f:
            for i in range(5):
                print(
                    f"Running the experiment for n_cell_x = {n_cell_x} and n_cell_y = {n_cell_y} : Iteration {i+1}"
                )
                completed_process = subprocess.run(
                    ["python3", "main_nse2d_kovasznay_convergence.py", "temp.yaml"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                for line in completed_process.stdout.decode().splitlines():
                    print(line)
                    f.write(line + "\n")

        # Check if the subprocess completed successfully
        if completed_process.returncode != 0:
            print(
                "[ERROR] : Experiment failed for n_cell_x = {} and n_cell_y = {}".format(
                    n_cell_x, n_cell_y
                )
            )

        # Remove the temp.yaml file
        os.remove("temp.yaml")
