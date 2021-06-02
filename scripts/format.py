import subprocess


def main():
    cmd = ["black", "."]
    proc = subprocess.run(cmd)
    exit(proc.returncode)
