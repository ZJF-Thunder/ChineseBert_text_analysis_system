var canvas = document.getElementById('bg-canvas');
var ctx = canvas.getContext('2d');

var width = window.innerWidth;
var height = window.innerHeight;

canvas.width = width;
canvas.height = height;

function createGradient() {
    var gradient = ctx.createRadialGradient(
        width / 2, height / 2, 0,
        width / 2, height / 2, Math.sqrt(width * width + height * height) / 2
    );
    gradient.addColorStop(0, '#36b7e1');
    gradient.addColorStop(1, '#1565c0');
    return gradient;
}

function draw() {
    ctx.fillStyle = createGradient();
    ctx.fillRect(0, 0, width, height);
}

function resize() {
    width = window.innerWidth;
    height = window.innerHeight;

    canvas.width = width;
    canvas.height = height;

    draw();
}

draw();

window.addEventListener('resize', resize);
