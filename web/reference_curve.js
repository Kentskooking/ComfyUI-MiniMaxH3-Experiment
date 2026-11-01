import { app } from "../../scripts/app.js";

export function curveValue(time, mode, constant = 0.5, start = 0, end = 1, cycles = 1, phase = 0) {
    if (mode === "constant") return constant;
    let value;
    const position = time * cycles + phase;
    switch (mode) {
        case "sine": value = (1 - Math.cos(2 * Math.PI * position)) / 2; break;
        case "triangle": value = 1 - Math.abs(2 * (position - Math.floor(position)) - 1); break;
        case "linear": value = time; break;
        case "ease_in": value = (2 ** (10 * time) - 1) / 1023; break;
        case "ease_out": value = 1 - (2 ** (10 * (1 - time)) - 1) / 1023; break;
        case "ease_in_out": value = time <= 0.5 ? (2 ** (20 * time) - 1) / 2046 : 1 - (2 ** (20 * (1 - time)) - 1) / 2046; break;
        default: return NaN;
    }
    return start + (end - start) * value;
}

export function drawCurve(canvas, values) {
    const ctx = canvas.getContext("2d");
    const width = canvas.width, height = canvas.height;
    const left = 62, right = width - 24, top = 48, bottom = height - 48;
    ctx.fillStyle = "#18212c";
    ctx.fillRect(0, 0, width, height);
    ctx.font = "15px sans-serif";
    ctx.fillStyle = "#e2e8f0";
    ctx.textAlign = "left";
    ctx.fillText(`Reference mix over clip duration - ${values.reference_curve}`, 16, 25);
    for (const ratio of [0, 0.25, 0.5, 0.75, 1]) {
        const y = bottom - ratio * (bottom - top);
        ctx.strokeStyle = "#354354";
        ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
        ctx.fillStyle = "#b4c2d2";
        ctx.textAlign = "right";
        ctx.fillText(ratio === 0 ? "A / 0" : ratio === 1 ? "B / 1" : String(ratio), left - 9, y + 5);
    }
    ctx.textAlign = "center";
    for (const time of [0, 0.25, 0.5, 0.75, 1]) {
        ctx.fillText(`${time * 100}%`, left + time * (right - left), bottom + 22);
    }
    ctx.strokeStyle = "#68dacb";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let i = 0; i <= 800; i++) {
        const time = i / 800;
        const ratio = curveValue(time, values.reference_curve, values.reference_latent_mix,
            values.curve_start, values.curve_end, values.curve_cycles, values.curve_phase);
        const x = left + time * (right - left), y = bottom - ratio * (bottom - top);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.textAlign = "left";
    ctx.font = "12px sans-serif";
    ctx.fillStyle = "#b4c2d2";
    ctx.fillText("Reference time; applied weights average H3 frame groups. Context stays fixed.", 16, height - 8);
}

app.registerExtension({
    name: "H3Experiments.ReferenceCurve",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "H3ExperimentsOmniReferenceBlend") return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onCreated?.apply(this, arguments);
            const canvas = document.createElement("canvas");
            canvas.width = 720;
            canvas.height = 270;
            canvas.style.cssText = "width:100%;height:100%;object-fit:contain;display:block;background:#18212c;border-radius:6px;";
            const preview = this.addDOMWidget("reference_curve_preview", "h3_reference_curve", canvas,
                { serialize: false, getMinHeight: () => 200, getMaxHeight: () => 240 });
            preview.computeSize = width => [width, 220];
            const defaults = { reference_curve: "constant", reference_latent_mix: 0.5,
                curve_start: 0, curve_end: 1, curve_cycles: 1, curve_phase: 0 };
            const redraw = () => {
                const values = { ...defaults };
                for (const widget of this.widgets) {
                    if (widget.name in values) values[widget.name] = widget.value;
                }
                drawCurve(canvas, values);
            };
            for (const widget of this.widgets) {
                if (!(widget.name in defaults)) continue;
                const callback = widget.callback;
                widget.callback = function () {
                    const result = callback?.apply(this, arguments);
                    redraw();
                    return result;
                };
            }
            const onConfigure = this.onConfigure;
            this.onConfigure = function () {
                const result = onConfigure?.apply(this, arguments);
                redraw();
                this.setSize([this.size[0], Math.max(this.size[1], 510)]);
                return result;
            };
            this.setSize([Math.max(this.size[0], 530), Math.max(this.size[1], 510)]);
            redraw();
            return result;
        };
    },
});
