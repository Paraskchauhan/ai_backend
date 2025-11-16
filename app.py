from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API is working"}

# Example endpoint for prediction
@app.post("/predict")
def predict(data: dict):
    # Yaha aap apna AI model use karoge
    return {"result": "AI prediction here"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
