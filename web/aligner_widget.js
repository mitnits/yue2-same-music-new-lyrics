// Lyric Aligner node UI. The aligner page is NOT embedded in the canvas (canvas zoom would shrink it into
// an unreadable postage stamp); instead the node offers two buttons: a full-window overlay and a new tab.
import { app } from "../../scripts/app.js";

function ensureOverlay() {
  let ov = document.getElementById("yue2sml-aligner-overlay");
  if (ov) return ov;
  ov = document.createElement("div");
  ov.id = "yue2sml-aligner-overlay";
  ov.style.cssText = "position:fixed;inset:0;z-index:10000;background:rgba(0,0,0,.7);display:none;flex-direction:column;";
  const bar = document.createElement("div");
  bar.style.cssText = "display:flex;gap:10px;align-items:center;padding:6px 12px;background:#26262b;color:#e8e8ec;font:13px system-ui,sans-serif;border-bottom:1px solid #3a3a42;";
  const title = document.createElement("span"); title.id = "yue2sml-aligner-title"; title.style.fontWeight = "600";
  const hint = document.createElement("span"); hint.textContent = "Edits save automatically. Close this and queue the workflow again to render."; hint.style.opacity = ".7";
  const tab = document.createElement("button"); tab.textContent = "open in new tab"; tab.style.cssText = "margin-left:auto;font:inherit;padding:4px 10px;";
  const close = document.createElement("button"); close.textContent = "✕ close"; close.style.cssText = "font:inherit;padding:4px 10px;";
  bar.append(title, hint, tab, close);
  const frame = document.createElement("iframe"); frame.id = "yue2sml-aligner-frame";
  frame.style.cssText = "flex:1;width:100%;border:0;background:#1e1e22;";
  ov.append(bar, frame);
  document.body.appendChild(ov);
  const hide = () => { ov.style.display = "none"; frame.src = "about:blank"; };
  close.addEventListener("click", hide);
  tab.addEventListener("click", () => { if (ov.dataset.url) window.open(ov.dataset.url, "_blank"); });
  window.addEventListener("keydown", (e) => { if (e.key === "Escape" && ov.style.display !== "none") hide(); });
  return ov;
}

function openOverlay(name) {
  const ov = ensureOverlay();
  const url = `/yue2sml/aligner?name=${encodeURIComponent(name)}&t=${Date.now()}`;
  ov.dataset.url = url;
  document.getElementById("yue2sml-aligner-title").textContent = `Lyric Aligner · ${name}`;
  document.getElementById("yue2sml-aligner-frame").src = url;
  ov.style.display = "flex";
}

app.registerExtension({
  name: "yue2sml.lyricAligner",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "Yue2SmlLyricAligner") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      const node = this;
      const nameOf = () => node.widgets?.find((w) => w.name === "name")?.value || "my_song";
      node.addWidget("button", "Open Lyric Aligner", null, () => openOverlay(nameOf()));
      node.addWidget("button", "Open in new tab", null, () => window.open(`/yue2sml/aligner?name=${encodeURIComponent(nameOf())}`, "_blank"));
      node._yue2Status = node.addWidget("text", "status", "queue once to create the session", () => {}, { serialize: false });
      node._yue2Status.disabled = true;
      if (node.size[0] < 420) node.setSize([420, node.size[1]]);
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const name = Array.isArray(message?.aligner) ? message.aligner[0] : message?.aligner;
      const text = Array.isArray(message?.text) ? message.text[0] : message?.text;
      if (this._yue2Status && typeof text === "string") this._yue2Status.value = text.slice(0, 160);
      // refresh the overlay if it is showing this session
      const ov = document.getElementById("yue2sml-aligner-overlay");
      if (ov && ov.style.display !== "none" && typeof name === "string" && ov.dataset.url?.includes(encodeURIComponent(name))) {
        document.getElementById("yue2sml-aligner-frame").contentWindow?.location.reload();
      }
    };
  },
});
