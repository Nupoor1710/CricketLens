"use strict";

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};
const pct = (x) => Math.round(x * 100) + "%";
const pretty = (s) => s.replace(/[_-]/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

let result = null; // last API response

/* ---------------- tabs ---------------- */
function showTab(name) {
  for (const t of ["identify", "report"]) {
    $("tab-" + t).setAttribute("aria-selected", String(t === name));
    $("view-" + t).hidden = t !== name;
  }
}
$("tab-identify").onclick = () => showTab("identify");
$("tab-report").onclick = () => showTab("report");

/* ---------------- status + roster ---------------- */
let people = [];
let lastFile = null;

async function loadStatus() {
  const data = await (await fetch("/api/status")).json();
  people = data.people || [];
  const m = data.metrics;
  $("untrained").hidden = data.ready;
  const roster = $("roster");
  roster.replaceChildren();
  $("names").replaceChildren(...people.map((p) => { const o = el("option"); o.value = p.name; return o; }));

  if (!people.length) roster.append(el("li", "empty", "Nobody yet. Add a player above."));
  const max = Math.max(1, ...people.map((p) => p.count));
  for (const p of people) {
    const li = el("li");
    li.dataset.name = p.name.toLowerCase();
    li.append(el("span", "nm", p.name));
    const r = m && m.report && m.report[p.id];
    li.append(el("span", "meta", r ? pct(r.recall) + " right" : ""));
    const bar = el("span", "bar");
    const fill = el("i");
    fill.style.width = (p.count / max) * 100 + "%";
    bar.append(fill);
    li.append(bar, el("span", "meta", p.count + (p.count === 1 ? " reference photo" : " reference photos")));
    roster.append(li);
  }
  applyFilter();
  const total = people.reduce((a, p) => a + p.count, 0);
  $("sheet-foot").textContent = `${people.length} players · ${total} reference photos`;
  $("squad-sum").textContent = `Squad on file · ${people.length} players`;
  if (m) renderReport(m);
  else $("report").replaceChildren(el("p", "note", "Optional: add 6+ photos per player, then run python train.py to get an accuracy report here."));
}

function applyFilter() {
  const q = $("find").value.trim().toLowerCase();
  document.querySelectorAll("#roster li[data-name]").forEach((li) => { li.hidden = q && !li.dataset.name.includes(q); });
}
$("find").oninput = applyFilter;

async function sendTeach(body) {
  const res = await fetch("/api/teach", { method: "POST", body });
  const d = await res.json();
  if (!res.ok) throw new Error(d.error || "Something went wrong.");
  return d;
}

$("add-go").onclick = async () => {
  const name = $("add-name").value.trim();
  const files = [...$("add-files").files];
  const msg = $("add-msg");
  if (!name || !files.length) { msg.textContent = "Enter a name and choose photos."; return; }
  const body = new FormData();
  body.append("name", name);
  files.forEach((f) => body.append("images", f));
  $("add-go").disabled = true;
  msg.textContent = "Learning faces… the first time can take a minute.";
  try {
    const d = await sendTeach(body);
    msg.textContent = `Added ${d.added} photo(s) of ${d.name}` + (d.skipped ? `, ${d.skipped} skipped (no clear face).` : ".");
    $("add-name").value = ""; $("add-files").value = "";
    await loadStatus();
  } catch (e) { msg.textContent = e.message; }
  $("add-go").disabled = false;
};

/* ---------------- upload ---------------- */
const drop = $("drop");
drop.onclick = () => $("file").click();
drop.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); } };
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  if (e.dataTransfer.files[0]) analyse(e.dataTransfer.files[0]);
};
$("file").onchange = (e) => { if (e.target.files[0]) analyse(e.target.files[0]); };
document.addEventListener("paste", (e) => {
  const f = [...(e.clipboardData?.files || [])].find((x) => x.type.startsWith("image/"));
  if (f) analyse(f);
});
$("again").onclick = () => {
  $("stage").hidden = true;
  drop.hidden = false;
  $("file").value = "";
  result = null;
  showInPhoto();
};

async function analyse(file) {
  if (!file.type.startsWith("image/")) return;
  lastFile = file;
  result = null;
  showInPhoto();
  drop.hidden = true;
  $("stage").hidden = false;
  $("preview").src = URL.createObjectURL(file);
  $("faces").replaceChildren();
  clearBoxes();
  $("scan").hidden = false;
  setStatus("Looking for faces…");

  const body = new FormData();
  body.append("image", file);
  try {
    const res = await fetch("/api/identify", { method: "POST", body });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Something went wrong.");
    result = data;
    render();
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    $("scan").hidden = true;
  }
}

function setStatus(msg, isErr = false) {
  const s = $("status");
  s.textContent = msg;
  s.className = "status" + (isErr ? " err" : "");
}

/* ---------------- results ---------------- */
const threshold = () => Number($("thr").value) / 100;
$("thr").oninput = () => { $("thr-out").textContent = $("thr").value + "%"; if (result) render(); };

function showInPhoto() {
  const box = $("inphoto"), list = $("inphoto-list");
  list.replaceChildren();
  if (!result) { box.hidden = true; return; }
  box.hidden = false;
  const named = result.faces.filter((f) => f.top[0].prob >= threshold());
  if (!named.length) {
    list.append(el("li", "none", result.faces.length ? "Nobody in the squad matched." : "No face found."));
    return;
  }
  named.forEach((f) => {
    const li = el("li");
    li.append(el("span", "nm", f.top[0].name), el("span", "pc", pct(f.top[0].prob) + " match"));
    list.append(li);
  });
}

function clearBoxes() { document.querySelectorAll(".box").forEach((b) => b.remove()); }

function render() {
  showInPhoto();
  clearBoxes();
  const list = $("faces");
  list.replaceChildren();
  const faces = result.faces;

  if (!faces.length) {
    setStatus("No clear face found. Try a sharper photo where the face is turned towards the camera.", true);
    return;
  }
  const named = faces.filter((f) => f.top[0].prob >= threshold()).length;
  setStatus(
    faces.length === 1
      ? ""
      : `${faces.length} faces found, ${named} matched to a player.`
  );

  faces.forEach((f, i) => {
    const top = f.top[0];
    const known = top.prob >= threshold();
    const n = String(i + 1);

    // overlay box on the photo
    const [x1, y1, x2, y2] = f.box;
    const box = el("div", "box " + (known ? "known" : "unknown"));
    Object.assign(box.style, {
      left: x1 * 100 + "%", top: y1 * 100 + "%",
      width: (x2 - x1) * 100 + "%", height: (y2 - y1) * 100 + "%",
    });
    if (faces.length > 1) box.append(el("span", "tag", n));
    box.dataset.i = n;
    $("photo-wrap").append(box);

    // result card
    const card = el("article", "face");
    const head = el("div", "face-head");
    if (faces.length > 1) head.append(el("span", "chip", n));
    head.append(el("h2", "who" + (known ? "" : " unknown"), known ? top.name : "Not in the squad"));
    card.append(head);
    card.append(
      el("p", "sure",
        known
          ? `${pct(top.prob)} match`
          : `Closest match is ${top.name} at ${pct(top.prob)}, below your ${$("thr").value}% bar.`)
    );

    const stumps = el("div", "stumps");
    stumps.setAttribute("role", "img");
    stumps.setAttribute("aria-label", `Confidence ${pct(top.prob)}`);
    const lit = Math.round(top.prob * 20);
    for (let k = 0; k < 20; k++) {
      stumps.append(el("i", k < lit ? "on" + (known ? "" : " low") : ""));
    }
    card.append(stumps);

    const alts = el("ul", "alts");
    f.top.slice(1).forEach((a) => {
      const li = el("li");
      li.append(el("span", "", a.name), el("span", "", pct(a.prob)));
      alts.append(li);
    });
    if (alts.children.length) card.append(alts);

    // teach the model: wrong or unknown face -> tell it who this is
    const fix = el("button", "link teach-btn", known ? "Wrong person? Teach it" : "Teach it who this is");
    const form = el("div", "teach");
    form.hidden = true;
    const inp = el("input");
    inp.setAttribute("list", "names");
    inp.placeholder = "Player's full name";
    const save = el("button", "btn", "Save");
    form.append(inp, save);
    fix.onclick = () => { form.hidden = !form.hidden; if (!form.hidden) inp.focus(); };
    save.onclick = async () => {
      const name = inp.value.trim();
      if (!name) return;
      save.disabled = true;
      save.textContent = "Saving…";
      const body = new FormData();
      body.append("name", name);
      body.append("images", lastFile);
      body.append("face", String(i));
      try {
        const d = await sendTeach(body);
        await loadStatus();
        await analyse(lastFile);
        setStatus(`Learned ${d.name}. Results updated.`);
      } catch (e) { setStatus(e.message, true); save.disabled = false; save.textContent = "Save"; }
    };
    card.append(fix, form);

    card.onmouseenter = () => box.classList.add("active");
    card.onmouseleave = () => box.classList.remove("active");
    list.append(card);
  });
}

/* ---------------- report tab ---------------- */
function table(headers, rows, cls) {
  const wrap = el("div", "tbl-wrap");
  const t = el("table", cls);
  const thead = el("thead");
  const hr = el("tr");
  headers.forEach((h, i) => hr.append(el("th", i === 0 && cls === "cm" ? "rowh" : "", h)));
  thead.append(hr);
  t.append(thead);
  const tb = el("tbody");
  rows.forEach((r) => tb.append(r));
  t.append(tb);
  wrap.append(t);
  return wrap;
}

function renderReport(m) {
  const root = $("report");
  root.replaceChildren();
  root.className = "report";

  const stats = el("div", "stats");
  [
    [pct(m.accuracy), `accuracy on ${m.n_test} unseen photos`],
    [String(m.classes.length), "players"],
    [String(m.n_train + m.n_test), "photos used"],
    [String(m.skipped), "photos skipped (no clear face)"],
  ].forEach(([v, l]) => {
    const s = el("div", "stat");
    s.append(el("b", "", v), el("span", "", l));
    stats.append(s);
  });
  root.append(stats);

  // classifier comparison
  root.append(el("h3", "", "Classifiers compared"));
  root.append(el("p", "note", "Each was scored with cross-validation on the same face embeddings. The best one is deployed."));
  const best = Math.max(...Object.values(m.comparison).map((c) => c.mean));
  const rows = Object.entries(m.comparison).map(([name, c]) => {
    const tr = el("tr");
    tr.append(el("td", "", name));
    tr.append(el("td", c.mean === best ? "best" : "", pct(c.mean)));
    tr.append(el("td", "", "± " + (c.std * 100).toFixed(1) + " pts"));
    return tr;
  });
  root.append(table(["Classifier", "Accuracy", "Spread"], rows));

  // per player
  root.append(el("h3", "", "Per player"));
  root.append(el("p", "note", "Precision: when the model says this player, how often is it right? Recall: of this player's photos, how many did it find?"));
  const prow = m.classes.map((c) => {
    const r = m.report[c];
    const tr = el("tr");
    tr.append(el("td", "", pretty(c)));
    [r.precision, r.recall, r["f1-score"]].forEach((v) => tr.append(el("td", "", pct(v))));
    tr.append(el("td", "", String(m.counts[c])));
    return tr;
  });
  root.append(table(["Player", "Precision", "Recall", "F1", "Photos"], prow));

  // confusion matrix
  root.append(el("h3", "", "Confusion matrix"));
  root.append(el("p", "note", "Rows are who was really in the photo; columns are who the model said. A strong diagonal means few mix-ups."));
  const cmRows = m.classes.map((c, i) => {
    const tr = el("tr");
    tr.append(el("th", "rowh", pretty(c)));
    const total = m.confusion[i].reduce((a, b) => a + b, 0) || 1;
    m.confusion[i].forEach((v, j) => {
      const share = v / total;
      const td = el("td", i === j ? "diag" : "", String(v));
      td.style.background =
        i === j
          ? `rgba(11,42,91,${0.25 + 0.75 * share})`
          : v > 0 ? `rgba(179,32,42,${0.15 + 0.6 * share})` : "transparent";
      if (i !== j && v > 0) td.style.color = "#5b0f15";
      tr.append(td);
    });
    return tr;
  });
  root.append(table(["", ...m.classes.map(pretty)], cmRows, "cm"));
}

loadStatus().catch(() => {
  const n = $("untrained");
  n.hidden = false;
  n.textContent = "Can't reach the server. Start it with python app.py.";
});