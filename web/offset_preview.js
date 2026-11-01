import { app } from "../../scripts/app.js";

const offsetInputs = {
    H3ExperimentsVirtualHeadStart: "head_start_seconds",
    H3ExperimentsGeneratedTimelineOffset: "forward_seconds",
};

app.registerExtension({
    name: "H3Experiments.OffsetPreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const inputName = offsetInputs[nodeData.name];
        if (!inputName) return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onCreated?.apply(this, arguments);
            const control = this.widgets.find(widget => widget.name === inputName);
            const container = document.createElement("div");
            container.style.cssText = "box-sizing:border-box;padding:4px 12px;color:var(--input-text,#ddd);font:12px sans-serif;";
            const label = document.createElement("div");
            const slider = document.createElement("input");
            slider.type = "range";
            slider.min = "0";
            slider.max = "15";
            slider.step = "any";
            slider.disabled = true;
            slider.setAttribute("aria-label", "Offset in seconds (read-only preview)");
            slider.style.cssText = "width:100%;margin:6px 0 0;accent-color:#68dacb;opacity:1;pointer-events:none;";
            container.append(label, slider);
            const preview = this.addDOMWidget("offset_preview", "h3_offset_preview", container,
                { serialize: false, getMinHeight: () => 48, getMaxHeight: () => 48 });
            preview.computeSize = width => [width, 48];
            let executedValue;
            const redraw = () => {
                const connected = this.inputs?.some(input => input.name === inputName && input.link != null);
                const value = connected ? executedValue : Number(control.value);
                const known = Number.isFinite(value);
                label.textContent = known ? `${value} s / 15 s${connected ? " (last run)" : ""}` : "Connected FLOAT — run to preview";
                slider.style.visibility = known ? "visible" : "hidden";
                if (known) slider.value = String(value);
            };
            const callback = control.callback;
            control.callback = function () {
                const result = callback?.apply(this, arguments);
                redraw();
                return result;
            };
            const onExecuted = this.onExecuted;
            this.onExecuted = function (message) {
                const result = onExecuted?.apply(this, arguments);
                executedValue = message.h3_offset_seconds?.[0];
                redraw();
                return result;
            };
            const onConfigure = this.onConfigure;
            this.onConfigure = function () {
                const result = onConfigure?.apply(this, arguments);
                executedValue = undefined;
                redraw();
                return result;
            };
            const onConnectionsChange = this.onConnectionsChange;
            this.onConnectionsChange = function () {
                const result = onConnectionsChange?.apply(this, arguments);
                executedValue = undefined;
                queueMicrotask(redraw);
                return result;
            };
            this.setSize([this.size[0], Math.max(this.size[1], this.computeSize()[1])]);
            redraw();
            return result;
        };
    },
});
