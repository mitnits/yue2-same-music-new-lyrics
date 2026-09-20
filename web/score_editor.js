// yue2-same-music-new-lyrics: Score Editor front-end. The server decides what flows downstream; this file keeps the text box in sync.
//
// The node sends, after each run:  text = the score that went downstream,
//                                  fill = the score to put in the box when the box is auto-filled (not user-edited),
//                                  hash = fingerprint of that fill (kept in the hidden 'autofilled_hash' widget so the
//                                         server can tell "auto-filled" from "edited by the user" on the next run).
import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "yue2sml.scoreEditor",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "Yue2SmlScoreEditor") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      const h = this.widgets?.find((w) => w.name === "autofilled_hash");
      if (h) {
        h.type = "hidden";
        h.computeSize = () => [0, -4];
      }
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const fill = Array.isArray(message?.fill) ? message.fill.join("") : message?.fill;
      const hash = Array.isArray(message?.hash) ? message.hash.join("") : message?.hash;
      if (typeof fill !== "string" || !fill) return;
      const box = this.widgets?.find((w) => w.name === "edited_abc");
      const h = this.widgets?.find((w) => w.name === "autofilled_hash");
      if (!box || !h) return;
      if (box.value !== fill) box.value = fill;
      h.value = hash || "";
      this.setDirtyCanvas(true, true);
    };
  },
});
