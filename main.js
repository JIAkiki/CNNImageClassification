const class_names = ["BACKGROUND_Google", "Faces", "Faces_easy", "Leopards",
      "Motorbikes", "accordion", "airplanes", "anchor", "ant",
      "barrel", "bass", "beaver", "binocular", "bonsai", "brain",
      "brontosaurus", "buddha", "butterfly", "camera", "cannon",
      "car_side", "ceiling_fan", "cellphone", "chair", "chandelier",
      "cougar_body", "cougar_face", "crab", "crayfish", "crocodile",
      "crocodile_head", "cup", "dalmatian", "dollar_bill", "dolphin",
      "dragonfly", "electric_guitar", "elephant", "emu", "euphonium",
      "ewer", "ferry", "flamingo", "flamingo_head", "garfield", "gerenuk",
      "gramophone", "grand_piano", "hawksbill", "headphone", "hedgehog",
      "helicopter", "ibis", "inline_skate", "joshua_tree", "kangaroo",
      "ketch", "lamp", "laptop", "llama", "lobster", "lotus", "mandolin",
      "mayfly", "menorah", "metronome", "minaret", "nautilus", "octopus",
      "okapi", "pagoda", "panda", "pigeon", "pizza", "platypus", "pyramid",
      "revolver", "rhino", "rooster", "saxophone", "schooner", "scissors",
      "scorpion", "sea_horse", "snoopy", "soccer_ball", "stapler", "starfish",
      "stegosaurus", "stop_sign", "strawberry", "sunflower", "tick", "trilobite",
      "umbrella", "watch", "water_lilly", "wheelchair", "wild_cat", "windsor_chair",
      "wrench", "yin_yang"]

async function loadModel() {
  const model = await tf.loadLayersModel('output_directory/model.json');
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

function speak(text) {
  const utterance = new SpeechSynthesisUtterance(text);
  speechSynthesis.speak(utterance);
}


async function main() {
  const model = await loadModel();
  const inputImage = document.getElementById('inputImage');
  inputImage.addEventListener('change', async () => {
    const file = inputImage.files[0];
    const imageURL = URL.createObjectURL(file);
    const image = await loadImage(imageURL);

    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

    const tensorImage = await tf.browser.fromPixels(canvas);
    const classIndex = await predict(model, tensorImage);
    const className = class_names[classIndex];
    document.getElementById('prediction').innerText = className;
  });

  const predictDrawingButton = document.getElementById('predictDrawing');
  predictDrawingButton.addEventListener('click', async () => {
    const tensorImage = await tf.browser.fromPixels(canvas);
    const classIndex = await predict(model, tensorImage);
    const className = class_names[classIndex];
    document.getElementById('prediction').innerText = className;
  });
}

main();
