# Author  : Thivin Anandh D
# Purpose : This script is used to run before releasing a project
#           - Compute the coverage percentage locally and updates the coverage badge in README.md
#           - Obtains the LICENSE file and updates the README.md with License information

import subprocess

# run the test process locally to update the coverage percentage
process = subprocess.Popen(["python3", "-m", "coverage", "run", "-m", "pytest", "-v", "tests/"])

# wait for the process to complete
process.wait()

# check if the process is successful
if process.returncode != 0:
    print("Error in running the tests")
    exit(1)

# Extract coverage percentage
result = subprocess.run(["python3", "-m", "coverage", "report"], capture_output=True, text=True, check=True)
lines = result.stdout.splitlines()
for line in lines:
    if 'TOTAL' in line:
        coverage = line.split()[3]

# strip the percentage symbol
coverage = coverage[:-1]
print(f"Coverage: {coverage}%")
# Update the coverage badge in README.md
with open("README.md", "r") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if "![Coverage]" in line:
            print("Updating coverage badge")
            lines[i] = f"![Coverage](https://img.shields.io/badge/Coverage-{coverage}%25-brightgreen)"

with open("README.md", "w") as f:
    f.writelines(lines)
