// Lyric Aligner: after the node runs, embed the aligner page in the node and offer to open it in a tab.
import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "yue2sml.lyricAligner",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "Yue2SmlLyricAligner") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      const box = document.createElement("div");
      box.style.cssText = "display:flex;flex-direction:column;gap:4px;height:100%;min-height:120px;";
      const bar = document.createElement("div");
      bar.style.cssText = "display:flex;gap:6px;align-items:center;font-size:12px;";
      const open = document.createElement("button");
      open.textContent = "Open aligner in a new tab";
      open.style.cssText = "font-size:12px;padding:2px 8px;";
      const note = document.createElement("span");
      note.textContent = "Queue once to create the session, then edit here or in the tab.";
      note.style.opacity = "0.7";
      bar.append(open, note);
      const frame = document.createElement("iframe");
      frame.style.cssText = "flex:1;width:100%;border:1px solid #444;border-radius:6px;background:#1e1e22;min-height:80px;";
      box.append(bar, frame);
      this._yue2Frame = frame;
      this._yue2Open = open;
      open.addEventListener("click", () => { if (this._yue2Url) window.open(this._yue2Url, "_blank"); });
      this.addDOMWidget("aligner", "ALIGNER", box, { serialize: false, hideOnZoom: false });
      this.setSize([Math.max(this.size[0], 760), Math.max(this.size[1], 520)]);
      // if the workflow was reloaded, point at the session of the current name widget
      const nameW = this.widgets?.find((w) => w.name === "name");
      if (nameW?.value) this._yue2SetSession(nameW.value, false);
    };

    nodeType.prototype._yue2SetSession = function (name, reload) {
      const url = `/yue2sml/aligner?name=${encodeURIComponent(name)}`;
      this._yue2Url = url;
      if (this._yue2Frame && (reload || this._yue2Frame.src === "" || !this._yue2Frame.src.endsWith(encodeURIComponent(name)))) {
        this._yue2Frame.src = url;
      } else if (this._yue2Frame && reload) {
        this._yue2Frame.contentWindow?.location.reload();
      }
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const name = Array.isArray(message?.aligner) ? message.aligner[0] : message?.aligner;
      if (typeof name === "string") {
        this._yue2SetSession(name, true);
        if (this._yue2Frame) this._yue2Frame.src = this._yue2Url + "&t=" + Date.now();
      }
    };
  },
});
