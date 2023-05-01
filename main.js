async function loadClassNames() {
  const response = await fetch('imagenet_labels.txt');
  const text = await response.text();
  const class_names = text.split('\n');
  return class_names;
}

async function loadModel() {
  const model = await tf.loadGraphModel('https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v1_100_224/classification/3/default/1', {fromTFHub: true});
  return model;
}

function preprocessImage(image) {
  const tensorImage = tf.browser.fromPixels(image);
  const resizedImage = tf.image.resizeBilinear(tensorImage, [224, 224]);
  const rescaledImage = resizedImage.div(tf.scalar(255.0));
  const batchedImage = rescaledImage.expandDims(0);
  return batchedImage;
}

async function predict(model, image) {
  const preprocessedImage = preprocessImage(image);
  const prediction = model.predict(preprocessedImage);
  const classIndices = prediction.argMax(-1).dataSync();
  const topK = await prediction.topk(5);
  const topNIndices = await topK.indices.data();
  const topNProbabilities = await topK.values.data();
  return { classIndices, topNIndices, topNProbabilities };
}

async function loadImage(src) {
  return new Promise((resolve) => {
    const image = new Image();
    image.crossOrigin = 'anonymous';
    image.src = src;
    image.onload = () => resolve(image);
  });
}

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
    const imageURL = URL.createObjectURL(file);
    const image = await loadImage(imageURL);

    const tensorImage = await tf.browser.fromPixels(image);
    const { classIndices, topNIndices, topNProbabilities } = await predict(model, tensorImage);
    const className = class_names[classIndices[0]];

    let predictionDisplay = `Predicted: ${className} (${(topNProbabilities[0] * 100).toFixed(2)}%)<br>`;
    for (let i = 1; i < topNIndices.length; i++) {
      predictionDisplay += `${i + 1}. ${class_names[topNIndices[i]]} (${(topNProbabilities[i] * 100).toFixed(2)}%)<br>`;
    }
    document.getElementById("prediction").innerHTML = predictionDisplay;

    if (classIndices[0] === currentClassIndex) {
      score++;
      updateScore();
    }
    displayNextClass();
  });

  displayNextClass();
}

main();
