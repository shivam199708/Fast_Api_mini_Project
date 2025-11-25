from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load model
model = joblib.load("iris_model.pkl")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Root route
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction endpoint for API
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(input: IrisInput):
    data = [[
        input.sepal_length,
        input.sepal_width,
        input.petal_length,
        input.petal_width
    ]]
    prediction = model.predict(data)[0]
    return {"prediction": int(prediction)}

# Prediction endpoint for HTML form
@app.post("/predict_html", response_class=HTMLResponse)
def predict_html(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(data)[0]
    return templates.TemplateResponse("index.html", {"request": request, "prediction": int(prediction)})
