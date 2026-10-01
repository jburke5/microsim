import subprocess


def main():
    cmd = ["ruff", "check"]
    proc = subprocess.run(cmd)
    exit(proc.returncode)
