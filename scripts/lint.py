import subprocess


def main():
    cmd = ["flake8"]
    proc = subprocess.run(cmd)
    exit(proc.returncode)
