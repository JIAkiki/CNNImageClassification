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

async function predict(model, image) {
  const preprocessedImage = preprocessImage(image);
  const prediction = model.predict(preprocessedImage);
  const classIndices = prediction.argMax(-1).dataSync();
  const topK = await prediction.topk(5);
  const topNIndices = await topK.indices.data();
  const topNProbabilities = await topK.values.data();
  return { classIndices, topNIndices, topNProbabilities };
}

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

async function computeSaliencyMap(model, image, classIndex) {
  const preprocessedImage = preprocessImage(image);
  const gradients = tf.grad((x) => model.predict(x));
  const gradTensor = gradients(preprocessedImage);
  const dy = tf.oneHot(tf.tensor1d([classIndex], 'int32'), 1000).reshape([1, 1, 1, 1000]);
  const saliencyMap = tf.sum(gradTensor.mul(dy), -1);
  const maxVal = saliencyMap.max();
  const minVal = saliencyMap.min();
  const normalizedSaliencyMap = saliencyMap.sub(minVal).div(maxVal.sub(minVal));
  return normalizedSaliencyMap.squeeze().mul(255).toInt();
}


function drawSaliencyMap(saliencyMap, canvas, width, height) {
  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(width, height);
  const data = imageData.data;
  const mapData = saliencyMap.dataSync();
  for (let i = 0; i < width * height; i++) {
    const intensity = mapData[i];
    data[i * 4] = intensity;
    data[i * 4 + 1] = intensity;
    data[i * 4 + 2] = intensity;
    data[i * 4 + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
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
    const image = await loadImage(file);

    const { classIndices, topNIndices, topNProbabilities } = await predict(model, image);
    const className = class_names[classIndices[0]];

    let predictionDisplay = `Predicted: ${className} (${(topNProbabilities[0] * 100).toFixed(2)}%)<br>`;
    for (let i = 1; i < topNIndices.length; i++) {
      predictionDisplay += `${i + 1}. ${class_names[topNIndices[i]]} (${(topNProbabilities[i] * 100).toFixed(2)}%)<br>`;
    }
    document.getElementById("prediction").innerHTML = predictionDisplay;

    const saliencyMap = await computeSaliencyMap(model, image, classIndices[0]);
    const canvas = document.getElementById('saliencyMap');
    canvas.width = image.width;
    canvas.height = image.height;
    drawSaliencyMap(saliencyMap, canvas, image.width, image.height);

    if (classIndices[0] === currentClassIndex) {
      score++;
      updateScore();
    }
    displayNextClass();
});

  displayNextClass();
}

main();
