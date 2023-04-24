
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
  const resizedImage = tf.image.resizeBilinear(image, [224, 224]);
  const rescaledImage = resizedImage.div(tf.scalar(255.0));
  const batchedImage = rescaledImage.expandDims(0);
  return batchedImage;
}

async function predict(model, image) {
  const preprocessedImage = preprocessImage(image);
  const prediction = model.predict(preprocessedImage);
  const classIndex = prediction.argMax(-1).dataSync()[0];
  return classIndex;
}

async function loadImage(src) {
  return new Promise((resolve) => {
    const image = new Image();
    image.crossOrigin = 'anonymous';
    image.src = src;
    image.onload = () => resolve(image);
  });
}

function getRandomClass(class_names) {
  const randomIndex = Math.floor(Math.random() * class_names.length);
  return class_names[randomIndex];
}

let score = 0;
function updateScore() {
  score++;
  document.getElementById('score').innerText = `Score: ${score}`;
}

function displayNextClass(class_names) {
  const randomIndex = Math.floor(Math.random() * class_names.length);
  const className = class_names[randomIndex];
  document.getElementById('classToGuess').innerText = className;
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

  async function handleImageUpload() {
    const file = inputImage.files[0];
    const imageURL = URL.createObjectURL(file);
    const image = await loadImage(imageURL);

    const tensorImage = await tf.browser.fromPixels(image);
    const predictedClassIndex = await predict(model, tensorImage);

    if (predictedClassIndex === currentClassIndex) {
      score++;
      updateScore();
    }

    displayNextClass();
  }

  const inputImage = document.getElementById('inputImage');
  inputImage.addEventListener('change', handleImageUpload);

  displayNextClass();
}

main();
