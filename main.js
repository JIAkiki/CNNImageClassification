// Load the ImageNet class names from a text file
async function loadClassNames() {
  const response = await fetch('imagenet_labels.txt');
  const text = await response.text();
  const class_names = text.split('\n');
  return class_names;
}

// Load the pre-trained TensorFlow.js model from TensorFlow Hub
async function loadModel() {
  const model = await tf.loadGraphModel('https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v1_100_224/classification/3/default/1', {fromTFHub: true});
  return model;
}

// Preprocess the input image to the format required by the model
function preprocessImage(image) {
  const canvas = document.createElement('canvas');
  canvas.width = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0);
  const tensorImage = tf.browser.fromPixels(canvas);
  const resizedImage = tf.image.resizeBilinear(tensorImage, [224, 224]);
  const rescaledImage = resizedImage.div(tf.scalar(255.0));
  const batchedImage = rescaledImage.expandDims(0);
  return batchedImage;
}

// Predict the class of the input image using the model
async function predict(model, image) {
  const preprocessedImage = preprocessImage(image);
  const prediction = model.predict(preprocessedImage);
  const classIndices = prediction.argMax(-1).dataSync();
  const topK = await prediction.topk(5);
  const topNIndices = await topK.indices.data();
  const topNProbabilities = await topK.values.data();
  return { classIndices, topNIndices, topNProbabilities };
}

// Load an image from a file input
async function loadImage(file) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (event) => {
      const image = new Image();
      image.crossOrigin = 'anonymous';
      image.src = event.target.result;
      image.onload = () => resolve(image);
    };
    reader.readAsDataURL(file);
  });
}

// The main function of the application
async function main() {
  const model = await loadModel();
  const class_names = await loadClassNames();

  let score = 0;
  let currentClassIndex;

  function updateScore() {
    document.getElementById('scoreCounter').innerText = score;
  }

  async function displayNextClass() {
    currentClassIndex = Math.floor(Math.random() * class_names.length);
    const className = class_names[currentClassIndex];
    document.getElementById('nextClass').innerText = `Upload an image of: ${className}`;
  }

  const inputImage = document.getElementById('inputImage');
  inputImage.addEventListener("change", async () => {
    const file = inputImage.files[0];
    const image = await loadImage(file);

    const { classIndices, topNIndices, topNProbabilities } = await predict(model, image);
    const className = class_names[classIndices[0]];

    let predictionDisplay = `Predicted: ${className} (${(topNProbabilities[0] * 100).toFixed(2)}%)<br>`;
    for (let i = 1; i < topNIndices.length; i++) {
      predictionDisplay += `${i + 1}. ${class_names[topNIndices[i]]} (${(topNProbabilities[i] * 100).toFixed(2)}%)<br>`;
    }
    // Update the prediction display with the top 5 predicted classes and their probabilities
    document.getElementById("prediction").innerHTML = predictionDisplay;

    // Check if the prediction matches the current class, and update the score if it does
    if (classIndices[0] === currentClassIndex) {
      score++;
      updateScore();
    }

    // Display a new class for the user to upload an image of
    displayNextClass();
  });

  // Initialize the application by displaying the first class
  displayNextClass();
}

// Run the main function
main();
