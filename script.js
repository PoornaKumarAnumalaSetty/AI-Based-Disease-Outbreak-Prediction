function predictOutbreak() {
    // Get values from the input fields
    const state = document.getElementById('state').value;
    const disease = document.getElementById('disease').value;
    const currentCases = document.getElementById('cases').value;

    // Validate input
    if (!state || !disease || !currentCases) {
        alert("Please fill in all fields.");
        return;
    }

    // Make the POST request to the Flask server
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ state: state, disease: disease, current_cases: currentCases }),
    })
    .then(response => response.json())
    .then(data => {
        // Update the prediction result
        document.getElementById('prediction-text').textContent = `Predicted Cases: ${data.prediction}`;
        document.getElementById('prediction').style.display = 'block';

        // Set the image source for the prediction graph
        const img = document.getElementById('prediction-graph');
        img.src = 'data:image/png;base64,' + data.graph; // Set the image source
        img.style.display = 'block'; // Show the graph
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while predicting the outbreak.');
    });
}
