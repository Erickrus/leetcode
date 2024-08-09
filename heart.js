const CANVAS_WIDTH = 640;
const CANVAS_HEIGHT = 480;
const CANVAS_CENTER_X = CANVAS_WIDTH / 2;
const CANVAS_CENTER_Y = CANVAS_HEIGHT / 2;
const IMAGE_ENLARGE = 11;
const HEART_COLOR = "#FF69B4";

function heartFunction(t, shrinkRatio = IMAGE_ENLARGE) {
    let x = 16 * Math.pow(Math.sin(t), 3);
    let y = -(13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t));

    x *= shrinkRatio;
    y *= shrinkRatio;

    x += CANVAS_CENTER_X;
    y += CANVAS_CENTER_Y;

    return [x, y];
}

function scatterInside(x, y, beta = 0.15) {
    const ratioX = -beta * Math.log(Math.random());
    const ratioY = -beta * Math.log(Math.random());

    const dx = ratioX * (x - CANVAS_CENTER_X);
    const dy = ratioY * (y - CANVAS_CENTER_Y);

    return [x - dx, y - dy];
}

function shrink(x, y, ratio) {
    const force = -1 / Math.pow(((x - CANVAS_CENTER_X) ** 2 + (y - CANVAS_CENTER_Y) ** 2), 0.6);
    const dx = ratio * force * (x - CANVAS_CENTER_X);
    const dy = ratio * force * (y - CANVAS_CENTER_Y);

    return [x - dx, y - dy];
}

function curve(p) {
    return 2 * (2 * Math.sin(4 * p)) / (2 * Math.PI);
}

class Heart {
    constructor(generateFrame = 20) {
        this._points = new Set();
        this._edgeDiffusionPoints = new Set();
        this._centerDiffusionPoints = new Set();
        this.allPoints = {};
        this.build(2000);

        this.generateFrame = generateFrame;
        for (let frame = 0; frame < generateFrame; frame++) {
            this.calc(frame);
        }
    }

    build(number) {
        for (let i = 0; i < number; i++) {
            const t = Math.random() * 2 * Math.PI;
            const [x, y] = heartFunction(t);
            this._points.add(`${x},${y}`);
        }

        this._points.forEach(point => {
            const [x, y] = point.split(',').map(Number);
            for (let i = 0; i < 3; i++) {
                const [sx, sy] = scatterInside(x, y, 0.05);
                this._edgeDiffusionPoints.add(`${sx},${sy}`);
            }
        });

        const pointList = Array.from(this._points);
        for (let i = 0; i < 6000; i++) {
            const [px, py] = pointList[Math.floor(Math.random() * pointList.length)].split(',').map(Number);
            const [sx, sy] = scatterInside(px, py, 0.17);
            this._centerDiffusionPoints.add(`${sx},${sy}`);
        }
    }

    static calcPosition(x, y, ratio) {
        const force = 1 / Math.pow(((x - CANVAS_CENTER_X) ** 2 + (y - CANVAS_CENTER_Y) ** 2), 0.52);

        const dx = ratio * force * (x - CANVAS_CENTER_X) + (Math.random() * 2 - 1);
        const dy = ratio * force * (y - CANVAS_CENTER_Y) + (Math.random() * 2 - 1);

        return [x - dx, y - dy];
    }

    calc(generateFrame) {
        const ratio = 10 * curve(generateFrame / 10 * Math.PI);

        const haloRadius = Math.floor(4 + 6 * (1 + curve(generateFrame / 10 * Math.PI)));
        const haloNumber = Math.floor(3000 + 4000 * Math.abs(Math.pow(curve(generateFrame / 10 * Math.PI), 2)));

        const allPoints = [];
        const heartHaloPoint = new Set();

        for (let i = 0; i < haloNumber; i++) {
            const t = Math.random() * 4 * Math.PI;
            let [x, y] = heartFunction(t, 11.5);
            [x, y] = shrink(x, y, haloRadius);

            if (!heartHaloPoint.has(`${x},${y}`)) {
                heartHaloPoint.add(`${x},${y}`);
                x += Math.floor(Math.random() * 29) - 14;
                y += Math.floor(Math.random() * 29) - 14;
                const size = Math.floor(Math.random() * 2) + 1;
                allPoints.push([x, y, size]);
            }
        }

        this._points.forEach(point => {
            let [x, y] = point.split(',').map(Number);
            [x, y] = Heart.calcPosition(x, y, ratio);
            const size = Math.floor(Math.random() * 3) + 1;
            allPoints.push([x, y, size]);
        });

        this._edgeDiffusionPoints.forEach(point => {
            let [x, y] = point.split(',').map(Number);
            [x, y] = Heart.calcPosition(x, y, ratio);
            const size = Math.floor(Math.random() * 2) + 1;
            allPoints.push([x, y, size]);
        });

        this._centerDiffusionPoints.forEach(point => {
            let [x, y] = point.split(',').map(Number);
            [x, y] = Heart.calcPosition(x, y, ratio);
            const size = Math.floor(Math.random() * 2) + 1;
            allPoints.push([x, y, size]);
        });

        this.allPoints[generateFrame % this.generateFrame] = allPoints;
    }

    render(ctx, renderFrame) {
        this.allPoints[renderFrame % this.generateFrame].forEach(([x, y, size]) => {
            ctx.fillStyle = HEART_COLOR;
            ctx.fillRect(x, y, size, size);
        });
    }
}

function draw(ctx, heart, renderFrame = 0) {
    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    heart.render(ctx, renderFrame);
    setTimeout(() => draw(ctx, heart, renderFrame + 1), 160);
}

window.onload = function () {
    const canvas = document.createElement('canvas');
    canvas.width = CANVAS_WIDTH;
    canvas.height = CANVAS_HEIGHT;
    document.body.appendChild(canvas);
    const ctx = canvas.getContext('2d');

    const heart = new Heart();
    draw(ctx, heart);

    const authorLabel = document.createElement('div');
    authorLabel.innerText = "Author";
    authorLabel.style.position = "absolute";
    authorLabel.style.top = "50%";
    authorLabel.style.left = "50%";
    authorLabel.style.color = HEART_COLOR;
    authorLabel.style.transform = "translate(-50%, -50%)";
    document.body.appendChild(authorLabel);

    const titleLabel = document.createElement('div');
    titleLabel.innerText = "...";
    titleLabel.style.position = "absolute";
    titleLabel.style.top = "10%";
    titleLabel.style.left = "50%";
    titleLabel.style.color = HEART_COLOR;
    titleLabel.style.fontSize = "18px";
    titleLabel.style.transform = "translateX(-50%)";
    document.body.appendChild(titleLabel);
};
