import subprocess


def main():
    args = [
        "python",
        "-m",
        "unittest",
        "discover",
        "--start-directory",
        "./microsim/test",
        "--pattern",
        "test_*.py",
    ]
    proc = subprocess.run(args)
    exit(proc.returncode)
