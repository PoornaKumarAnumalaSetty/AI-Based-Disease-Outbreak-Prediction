from flask import Flask, render_template, request, jsonify
import base64
from io import BytesIO
import matplotlib.pyplot as plt

app = Flask(__name__)

# Dummy prediction function
def predict_cases(state, disease, current_cases):
    # Replace this with actual prediction logic
    predicted_cases = int(current_cases) * 1.2  # Simple dummy prediction
    return round(predicted_cases)

def create_prediction_graph(state, disease, actual_cases, predicted_cases):
    fig, ax = plt.subplots()
    ax.bar(['Actual', 'Predicted'], [actual_cases, predicted_cases], color=['blue', 'orange'])
    ax.set_ylabel('Number of Cases')
    ax.set_title(f'Actual vs Predicted Cases for {disease.capitalize()} in {state.upper()}')

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graph_base64 = base64.b64encode(image_png).decode('utf-8')

    return graph_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    state = data['state']
    disease = data['disease']
    current_cases = int(data['current_cases'])

    predicted_cases = predict_cases(state, disease, current_cases)
    graph = create_prediction_graph(state, disease, current_cases, predicted_cases)

    return jsonify({
        'prediction': predicted_cases,
        'graph': graph
    })

if __name__ == '__main__':
    app.run(debug=True)
