document.getElementById('uploadForm').onsubmit = async function(e) {
    e.preventDefault();
    const formData = new FormData();
    formData.append('model', document.getElementById('model').value);
    formData.append('file', document.getElementById('file').files[0]);

    const response = await fetch('/predict', {
        method: 'POST',
        body: formData,
    });

    const result = await response.json();
    document.getElementById('output').innerHTML = `<pre>${JSON.stringify(result, null, 2)}</pre>`;
}
