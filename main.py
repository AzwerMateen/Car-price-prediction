from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and data
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv("Cleaned Car.csv")


@app.route('/')
def index():
    # Get unique sorted lists
    companies = sorted(car['company'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = sorted(car['fuel_type'].unique())

    # Create a mapping of Company -> List of Models for efficient JS filtering
    model_map = {}
    for company in companies:
        models = car[car['company'] == company]['name'].unique().tolist()
        model_map[company] = sorted(models)

    return render_template(
        'index.html',
        companies=companies,
        model_map=model_map,
        years=years,
        fuel_types=fuel_types
    )


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract and cast data safely
        company = request.form.get('company')
        car_model = request.form.get('car_model')
        year = int(request.form.get('year'))
        fuel_type = request.form.get('fuel_type')
        kms_driven = int(request.form.get('kms_driven'))

        # Create DataFrame matching the model's training columns exactly
        input_df = pd.DataFrame(
            [[car_model, company, year, kms_driven, fuel_type]],
            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
        )

        prediction = model.predict(input_df)[0]

        # Format the output for better UX
        return f"Rs. {max(0, round(prediction, 2)):,}"

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True)