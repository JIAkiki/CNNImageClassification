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

async function loadStyleModel() {
  const model = await tf.loadGraphModel('https://tfhub.dev/google/tfjs-model/magenta/arbitrary-image-stylization-v1-256/1/default/1', { fromTFHub: true });
  return model;
}

async function applyStyle(image, model) {
  const style_image = await tf.browser.fromPixels(await loadImage('path/to/van-gogh-style-image.jpg'));
  const content_image = image;
  const style_image_resized = tf.image.resizeBilinear(style_image, [256, 256]);
  const content_image_resized = tf.image.resizeBilinear(content_image, [256, 256]);
  const style_input = style_image_resized.expandDims();
  const content_input = content_image_resized.expandDims();
  const outputs = await model.executeAsync({ 'style_image': style_input, 'content_image': content_input });
  const stylized_image = outputs[0].squeeze();
  return stylized_image;
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
    const predictedClassName = class_names[predictedClassIndex];
    document.getElementById('prediction').innerText = `Predicted: ${predictedClassName}`;

    if (predictedClassIndex === currentClassIndex) {
      score++;
      updateScore();
    }

    // Apply the style to the image and set it as the background
    const styleModel = await loadStyleModel();
    const styledTensorImage = await applyStyle(tensorImage, styleModel);
    const styledImageBlob = await new Promise((resolve) => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      canvas.width = styledTensorImage.shape[1];
      canvas.height = styledTensorImage.shape[0];
      tf.browser.toPixels(styledTensorImage, canvas).then(() => {
        canvas.toBlob(resolve, 'image/png');
      });
    });
    const styledImageURL = URL.createObjectURL(styledImageBlob);
    document.body.style.backgroundImage = `url(${styledImageURL})`;

    displayNextClass();
  }

  const inputImage = document.getElementById('inputImage');
  inputImage.addEventListener('change', handleImageUpload);

  displayNextClass();
}

main();
