import subprocess

def train():
    for i in range(5):
        result = subprocess.Popen(["python3", "evaluate_few_evals.py", f"general.seed={i}"])
        result.wait()

if __name__ == '__main__':
    train()
