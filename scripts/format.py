import subprocess


shared_autopep8_options = ["--aggressive", "--aggressive"]


def main():
    cmd = ["autopep8", "--in-place", "--recursive", "."] + shared_autopep8_options
    proc = subprocess.run(cmd)
    exit(proc.returncode)


def diffmain():
    cmd = ["autopep8", "--diff", "--recursive", "."] + shared_autopep8_options
    proc = subprocess.run(cmd)
    exit(proc.returncode)
