import subprocess


def main():
    cmd = ["ruff", "format"]
    proc = subprocess.run(cmd)
    exit(proc.returncode)
