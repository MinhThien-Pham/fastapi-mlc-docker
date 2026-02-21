from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to KRSP_SecureLLM API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/inference")
def inference(prompt: str):
    return {"prompt": prompt, "response": "MLC-CLI integration"}
