async function loadModel() {
  const model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
  return model;
}

async function loadClassNames() {
  const response = await fetch('https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_labels.txt');
  const text = await response.text();
  const classNames = text.trim().split('\n');
  return classNames;
}

async function loadImage(src) {
  return new Promise((resolve) => {
    const image = new Image();
    image.crossOrigin = 'anonymous';
    image.src = src;
    image.onload = () => resolve(image);
  });
}

let score = 0;
let class_names = [];
let currentClassIndex;

function displayNextClass() {
  currentClassIndex = Math.floor(Math.random() * class_names.length);
  document.getElementById('classToUpload').innerText = class_names[currentClassIndex];
}

function updateScore() {
  document.getElementById('score').innerText = `Score: ${score}`;
}

async function predict(model, image) {
  const preprocessedImage = preprocessImage(image);
  const prediction = model.predict(preprocessedImage);
  const classIndex = prediction.argMax(-1).dataSync()[0];
  return classIndex;
}

function preprocessImage(image) {
  const resizedImage = tf.image.resizeBilinear(image, [224, 224]);
  const rescaledImage = resizedImage.div(tf.scalar(255.0));
  const batchedImage = rescaledImage.expandDims(0);
  return batchedImage;
}

async function handleImageUpload() {
  const file = inputImage.files[0];
  const imageURL = URL.createObjectURL(file);
  const image = await loadImage(imageURL);

  const tensorImage = await tf.browser.fromPixels(image);
  const predictedClassIndex = await predict(model, tensorImage);
  const predictedClassName = class_names[predictedClassIndex];
  document.getElementById('prediction').innerText = `Predicted: ${predictedClassName}`;

  if (predictedClassIndex === currentClassIndex) {
    score++;
    updateScore();
  }

  displayNextClass();

  // Distorted background image
  const distortedCanvas = document.createElement('canvas');
  distortedCanvas.width = window.innerWidth;
  distortedCanvas.height = window.innerHeight;
  const distortedCtx = distortedCanvas.getContext('2d');
  distortedCtx.drawImage(image, 0, 0, distortedCanvas.width, distortedCanvas.height);
  distortedCtx.globalCompositeOperation = 'overlay';
  distortedCtx.fillStyle = 'rgba(128, 128, 128, 0.5)';
  distortedCtx.fillRect(0, 0, distortedCanvas.width, distortedCanvas.height);

  const distortedDataURL = distortedCanvas.toDataURL();
  document.body.style.backgroundImage = `url(${distortedDataURL})`;
}

async function main() {
  const model = await loadModel();
  class_names = await loadClassNames();
  displayNextClass();

  const inputImage = document.getElementById('inputImage');
  inputImage.addEventListener('change', handleImageUpload);
}

main();
