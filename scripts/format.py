#!/usr/bin/env python3
import subprocess


def main():
    cmd = ["black", "."]
    proc = subprocess.run(cmd)
    exit(proc.returncode)


if __name__ == "__main__":
    main()
