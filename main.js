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
      "wrench", "yin_yang"] // Replace with the actual class names list.

async function loadModel() {
  const model = await tf.loadLayersModel('output_directory/model.json');
  return model;
}

function preprocessImage(image) {
  // Resize the image to match the input size of the model
  const resizedImage = tf.image.resizeBilinear(image, [224, 224]);

  // Rescale the image to match the range used during training (0-1)
  const rescaledImage = resizedImage.div(tf.scalar(255.0));

  // Add an extra dimension to match the expected input shape [1, 224, 224, 3]
  const batchedImage = rescaledImage.expandDims(0);

  return batchedImage;
}

function displayPrediction(prediction) {
  const predictionElement = document.getElementById("prediction");
  predictionElement.textContent = `Predicted class: ${class_names[prediction]}`;
}

document.getElementById("inputImage").addEventListener("change", async function (event) {
  const model = await loadModel();
  const image = new Image();
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");

  image.src = URL.createObjectURL(event.target.files[0]);
  image.onload = async function () {
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    const preprocessedImage = preprocessImage(image);
    const prediction = model.predict(preprocessedImage).argMax(-1).dataSync()[0];
    displayPrediction(prediction);
  };
});
