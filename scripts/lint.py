#!/usr/bin/env python3
import subprocess


def main():
    cmd = ["flake8"]
    proc = subprocess.run(cmd)
    exit(proc.returncode)


if __name__ == "__main__":
    main()
