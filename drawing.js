const drawingCanvas = document.getElementById('drawingCanvas');
const ctx = drawingCanvas.getContext('2d');

let drawing = false;

drawingCanvas.addEventListener('mousedown', (event) => {
  drawing = true;
  ctx.beginPath();
  ctx.moveTo(event.clientX - drawingCanvas.offsetLeft, event.clientY - drawingCanvas.offsetTop);
});

drawingCanvas.addEventListener('mousemove', (event) => {
  if (!drawing) return;
  ctx.lineTo(event.clientX - drawingCanvas.offsetLeft, event.clientY - drawingCanvas.offsetTop);
  ctx.stroke();
});

drawingCanvas.addEventListener('mouseup', () => {
  drawing = false;
});

drawingCanvas.addEventListener('mouseout', () => {
  drawing = false;
});
