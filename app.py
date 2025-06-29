import sys
from flask import Flask , render_template , request
import logging

from loan_approval.pipeline.prediction_pipeline import CustomData  , PredictionPipeline

# Configure logging
# logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict" , methods=["GET" , "POST"])
def prediction():
    try:
        if request.method == "GET":
            return render_template("forms.html")
        else:

            data=CustomData(
                Gender=request.form.get("gender"),
                Married=request.form.get("married"),
                Dependents=request.form.get("dependents"),
                Education=request.form.get("education"),
                Self_Employed=request.form.get("selfEmployed"),
                ApplicantIncome=int(request.form.get("applicantIncome")),
                CoapplicantIncome=float(request.form.get("coapplicantIncome")),
                LoanAmount=float(request.form.get("loanAmount")),
                Loan_Amount_Term=float(request.form.get("loanAmountTerm")),
                Credit_History=float(request.form.get("creditHistory")),
                Property_Area=request.form.get("propertyArea"),
            )
            predict_pipeline = PredictionPipeline()
            df = data.get_data_as_dataframe()
            result = predict_pipeline.predict(df)
            final_result = "approved" if result[0] == 1.0 else "reject"

            return render_template("result.html" , result=final_result)
            
    except Exception as e:
        print(e , sys)
        return render_template("forms.html", error_message=str(e))  # create error.html



if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8000,
        debug=True
    )