const inputImage = document.getElementById("inputImage");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const predictionElement = document.getElementById("prediction");

async function loadModel() {
  const model = await tf.loadLayersModel("output_directory/model.json");
  return model;
}

function preprocessImage(image) {
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const inputTensor = tf.browser.fromPixels(imageData, 3).toFloat();
  const normalizedTensor = inputTensor.div(tf.scalar(255));
  const batchedTensor = normalizedTensor.expandDims(0);
  return batchedTensor;
}

async function classifyImage(model, image) {
  const processedImage = preprocessImage(image);
  const prediction = await model.predict(processedImage).data();
  const topPrediction = Array.from(prediction)
    .map((value, index) => ({ value, index }))
    .sort((a, b) => b.value - a.value)[0];
  return topPrediction;
}

inputImage.addEventListener("change", async (event) => {
  if (event.target.files && event.target.files[0]) {
    const image = new Image();
    image.src = URL.createObjectURL(event.target.files[0]);
    image.onload = async () => {
      const model = await loadModel();
      const result = await classifyImage(model, image);
      predictionElement.innerHTML = `Predicted class: ${result.index}, Confidence: ${result.value.toFixed(2)}`;
    };
  }
});
