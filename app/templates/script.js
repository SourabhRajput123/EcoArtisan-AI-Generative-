document.getElementById('generate-button').addEventListener('click', async () => {
    const prompt = document.getElementById('prompt').value;
    const response = await fetch('http://localhost:5000/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
    });
    
    const data = await response.json();
    document.getElementById('result').innerText = `Generated Idea: ${data.idea}`;
});
