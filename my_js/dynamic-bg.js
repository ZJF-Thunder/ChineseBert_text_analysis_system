var maxRadius = 300;    // 背景最大半径
var smoothness = 40;    // 背景平滑过渡程度

var doc = document.documentElement;
var body = document.body;

var width = Math.max(doc.clientWidth, window.innerWidth || 0);
var height = Math.max(doc.clientHeight, window.innerHeight || 0);

body.addEventListener('mousemove', function (e) {
    var x = e.clientX;
    var y = e.clientY;

    var centerX = width / 2;
    var centerY = height / 2;

    var deltaX = x - centerX;
    var deltaY = y - centerY;

    var dist = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
    var radius = Math.min(maxRadius, dist);

    var percent = radius / maxRadius;
    var color1 = [255, 255, 255];
    var color2 = [50, 100, 150];

    var gradientColor = blendColors(color1, color2, percent);

    var gradientStr = 'radial-gradient(at ' + x + 'px ' + y + 'px, rgba('
        + gradientColor[0] + ', ' + gradientColor[1] + ', ' + gradientColor[2] + ', 0.7), transparent ' + smoothness + 'px)';
    body.style.backgroundImage = gradientStr;
});

function blendColors(color1, color2, percent) {
    var r = Math.round(lerp(color1[0], color2[0], percent));
    var g = Math.round(lerp(color1[1], color2[1], percent));
    var b = Math.round(lerp(color1[2], color2[2], percent));

    return [r, g, b];
}

function lerp(start, end, percent) {
    return (start + percent * (end - start));
}
