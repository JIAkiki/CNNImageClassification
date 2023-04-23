const drawingCanvas = document.getElementById('drawingCanvas');
const ctx = drawingCanvas.getContext('2d');

let drawing = false;

function getCursorPosition(canvas, event) {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  return { x, y };
}

drawingCanvas.addEventListener('mousedown', (event) => {
  drawing = true;
  const cursorPosition = getCursorPosition(drawingCanvas, event);
  ctx.beginPath();
  ctx.moveTo(cursorPosition.x, cursorPosition.y);
});

drawingCanvas.addEventListener('mousemove', (event) => {
  if (!drawing) return;
  const cursorPosition = getCursorPosition(drawingCanvas, event);
  ctx.lineTo(cursorPosition.x, cursorPosition.y);
  ctx.stroke();
});

drawingCanvas.addEventListener('mouseup', () => {
  drawing = false;
});

drawingCanvas.addEventListener('mouseout', () => {
  drawing = false;
});
