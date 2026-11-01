import { app } from "../../scripts/app.js";

const offsetInputs = {
    H3ExperimentsVirtualHeadStart: "head_start_seconds",
    H3ExperimentsGeneratedTimelineOffset: "forward_seconds",
};

app.registerExtension({
    name: "H3Experiments.OffsetControls",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const inputName = offsetInputs[nodeData.name];
        if (!inputName) return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onCreated?.apply(this, arguments);
            const control = this.widgets.find(widget => widget.name === inputName);
            let executedValue;
            const isConnected = () => this.inputs?.some(input => input.name === inputName && input.link != null);
            const slider = this.addWidget("slider", `${inputName}_slider`, control.value, (value, ...args) => {
                if (isConnected()) {
                    sync();
                    return;
                }
                const previous = control.value;
                control.value = value;
                control.callback?.call(control, value, ...args);
                this.onWidgetChanged?.(control.name, control.value, previous, control);
            }, { ...control.options, serialize: false });
            slider.tooltip = "Drag to change seconds, or edit the number field above. Both control the same value.";
            const sync = () => {
                const connected = isConnected();
                slider.disabled = connected;
                slider.options.read_only = connected;
                slider.label = connected
                    ? (Number.isFinite(executedValue) ? "seconds (FLOAT, last run)" : "seconds (FLOAT, run to update)")
                    : "seconds";
                slider.value = connected && Number.isFinite(executedValue) ? executedValue : control.value;
                this.setDirtyCanvas(true, true);
            };
            const callback = control.callback;
            control.callback = function () {
                const result = callback?.apply(this, arguments);
                sync();
                return result;
            };
            const onExecuted = this.onExecuted;
            this.onExecuted = function (message) {
                const result = onExecuted?.apply(this, arguments);
                executedValue = message.h3_offset_seconds?.[0];
                sync();
                return result;
            };
            const onConfigure = this.onConfigure;
            this.onConfigure = function () {
                const result = onConfigure?.apply(this, arguments);
                executedValue = undefined;
                sync();
                return result;
            };
            const onConnectionsChange = this.onConnectionsChange;
            this.onConnectionsChange = function () {
                const result = onConnectionsChange?.apply(this, arguments);
                executedValue = undefined;
                queueMicrotask(sync);
                return result;
            };
            this.setSize([this.size[0], Math.max(this.size[1], this.computeSize()[1])]);
            sync();
            return result;
        };
    },
});
