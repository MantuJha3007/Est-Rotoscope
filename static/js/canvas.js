class InteractiveCanvas {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        
        this.currentImage = null;
        this.currentMaskImage = null;
        this.points = []; // [{x, y, label}]
        this.activeMode = 1; // 1 = positive, 0 = negative
        this.maskOpacity = 0.55;

        this.onPointAdded = null; // callback(x, y, label)

        this.initEvents();
    }

    initEvents() {
        // Prevent default context menu on right click
        this.canvas.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            this.handleCanvasClick(e, 0); // 0 = Negative point
        });

        // Left click
        this.canvas.addEventListener('click', (e) => {
            this.handleCanvasClick(e, this.activeMode);
        });
    }

    setMode(mode) {
        this.activeMode = mode;
    }

    setFrameImage(img) {
        this.currentImage = img;
        this.canvas.width = img.naturalWidth || img.width;
        this.canvas.height = img.naturalHeight || img.height;
        this.render();
    }

    setMaskImage(img) {
        this.currentMaskImage = img;
        this.render();
    }

    clearMask() {
        this.currentMaskImage = null;
        this.points = [];
        this.render();
    }

    setPoints(pointsList) {
        this.points = pointsList || [];
        this.render();
    }

    handleCanvasClick(e, label) {
        if (!this.currentImage) return;

        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;

        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;

        this.points.push({ x, y, label });
        this.render();

        if (this.onPointAdded) {
            this.onPointAdded(x, y, label);
        }
    }

    render() {
        if (!this.currentImage) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            return;
        }

        // 1. Draw base video frame
        this.ctx.drawImage(this.currentImage, 0, 0, this.canvas.width, this.canvas.height);

        // 2. Draw mask overlay if available (RGBA with transparency)
        if (this.currentMaskImage) {
            this.ctx.save();
            this.ctx.drawImage(this.currentMaskImage, 0, 0, this.canvas.width, this.canvas.height);
            this.ctx.restore();
        }

        // 3. Draw prompt points
        this.points.forEach(pt => {
            const isPositive = pt.label === 1;
            const color = isPositive ? '#00ff66' : '#ff3333';
            
            // Outer glow ring
            this.ctx.beginPath();
            this.ctx.arc(pt.x, pt.y, 8, 0, Math.PI * 2);
            this.ctx.fillStyle = isPositive ? 'rgba(0, 255, 102, 0.4)' : 'rgba(255, 51, 51, 0.4)';
            this.ctx.fill();

            // Inner circle
            this.ctx.beginPath();
            this.ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
            this.ctx.fillStyle = color;
            this.ctx.fill();

            // Border
            this.ctx.lineWidth = 2;
            this.ctx.strokeStyle = '#ffffff';
            this.ctx.stroke();
        });
    }
}

