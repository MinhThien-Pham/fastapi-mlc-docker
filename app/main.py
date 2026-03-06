from fastapi import FastAPI
from pathlib import Path
import subprocess

app = FastAPI()

MLC_CLI_PATH = Path("/workspace/mlc-cli")

def run_command(command, cwd):
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True)

@app.get("/")
def root():
    return {"message": "Welcome to FastAPI + MLC CLI"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/setup-check")
def setup_check():
    if not MLC_CLI_PATH.exists():
        return {
            "status": "error",
            "message": "mlc-cli repo not found",
            "repo_exists": False,
            "path": str(MLC_CLI_PATH),
        }

    git_result = run_command(["git", "remote", "get-url", "origin"], MLC_CLI_PATH)
    go_result = run_command(["go", "version"], MLC_CLI_PATH)

    return {
        "status": "ok",
        "repo_exists": True,
        "path": str(MLC_CLI_PATH),
        "origin": git_result.stdout.strip(),
        "go_version": go_result.stdout.strip(),
        "git_returncode": git_result.returncode,
        "go_returncode": go_result.returncode,
        "git_stderr": git_result.stderr.strip(),
        "go_stderr": go_result.stderr.strip(),
    }

@app.post("/inference")
def inference(prompt: str):
    return {"prompt": prompt, "response": "MLC-CLI integration"}
