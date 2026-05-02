// --- 1. CONFIGURATION ---
const scale = 5;
const pitchWidth = 120;
const pitchHeight = 80;

let arrows = [];
let tempStart = null;
let activeId = null;
let currentPage = 0;
const pageSize = 4;

const svg = d3.select("#canvas")
  .attr("width", pitchWidth * scale)
  .attr("height", pitchHeight * scale);

const x = val => val * scale;
const y = val => val * scale;

// --- 2. MARKERS & PITCH ---
const defs = svg.append("defs");
const createMarker = (id, color) => {
  defs.append("marker")
    .attr("id", id).attr("viewBox", "0 0 10 10")
    .attr("refX", 8).attr("refY", 5)
    .attr("markerWidth", 6).attr("markerHeight", 6)
    .attr("orient", "auto")
    .append("path").attr("d", "M 0 0 L 10 5 L 0 10 z").attr("fill", color);
};

createMarker("head-black", "#333");
createMarker("head-red", "#ff4444");
createMarker("head-ghost", "#aaa");

const drawPitch = () => {
  const g = svg.append("g").attr("stroke", "#333").attr("fill", "none").attr("stroke-width", 2);
  g.append("rect").attr("width", x(120)).attr("height", y(80));
  g.append("line").attr("x1", x(60)).attr("y1", 0).attr("x2", x(60)).attr("y2", y(80));
  g.append("circle").attr("cx", x(60)).attr("cy", y(40)).attr("r", x(10));
  g.append("rect").attr("x", x(0)).attr("y", y(18)).attr("width", x(18)).attr("height", y(44));
  g.append("rect").attr("x", x(120-18)).attr("y", y(18)).attr("width", x(18)).attr("height", y(44));
};
drawPitch();

const ghost = svg.append("line")
  .attr("stroke", "#aaa").attr("stroke-dasharray", "4")
  .attr("marker-end", "url(#head-ghost)")
  .style("visibility", "hidden");

// --- 3. INTERACTION ---
svg.on("mousemove", (event) => {
  if (tempStart) {
    const [mX, mY] = d3.pointer(event);
    ghost.style("visibility", "visible")
      .attr("x1", x(tempStart.x)).attr("y1", y(tempStart.y))
      .attr("x2", mX).attr("y2", mY);
  }
});

svg.on("click", (event) => {
  const [mX, mY] = d3.pointer(event);
  if (!tempStart) {
    tempStart = { x: mX / scale, y: mY / scale };
  } else {
    const newArrow = { 
      id: "a" + Date.now(), 
      x1: tempStart.x, y1: tempStart.y, 
      x2: mX / scale, y2: mY / scale,
      height: "Ground pass", bodyPart: "Right foot", underPressure: false, duration: 1.0 
    };
    arrows.push(newArrow);
    tempStart = null;
    activeId = newArrow.id;
    ghost.style("visibility", "hidden");
    currentPage = Math.max(0, Math.ceil(arrows.length / pageSize) - 1);
    render();
  }
});

// --- 4. STATE SYNC & DELETION ---
function updateArrowProperties() {
  const arrow = arrows.find(a => a.id === activeId);
  if (arrow) {
    arrow.height = d3.select("#input-height").property("value");
    arrow.bodyPart = d3.select("#input-bodypart").property("value");
    arrow.underPressure = d3.select("#input-pressure").property("checked");
    arrow.duration = parseFloat(d3.select("#input-duration").property("value"));
    d3.select("#duration-val").text(arrow.duration.toFixed(1));
    render();
  }
}

// Logic to delete the active arrow
d3.select("#btn-delete-active").on("click", () => {
  if (activeId) {
    arrows = arrows.filter(a => a.id !== activeId);
    activeId = null;
    render();
  }
});

// Logic to clear everything
d3.select("#btn-clear-all").on("click", () => {
  if (confirm("Delete all plays?")) {
    arrows = [];
    activeId = null;
    currentPage = 0;
    render();
  }
});

d3.selectAll("#input-height, #input-bodypart, #input-pressure, #input-duration")
  .on("input", updateArrowProperties);

// --- 5. RENDER ENGINE ---
function render() {
  // Lines[cite: 5]
  const lines = svg.selectAll(".arrow-line").data(arrows, d => d.id);
  lines.enter().append("line")
    .attr("class", "arrow-line")
    .style("cursor", "pointer")
    .on("click", (e, d) => { e.stopPropagation(); activeId = d.id; render(); })
    .merge(lines)
    .attr("x1", d => x(d.x1)).attr("y1", d => y(d.y1))
    .attr("x2", d => x(d.x2)).attr("y2", d => y(d.y2))
    .attr("stroke", d => d.id === activeId ? "#ff4444" : "#333")
    .attr("stroke-width", d => d.id === activeId ? 4 : 2)
    .attr("marker-end", d => d.id === activeId ? "url(#head-red)" : "url(#head-black)");
  lines.exit().remove();

  // Panel Animation[cite: 6]
  const activeArrow = arrows.find(a => a.id === activeId);
  const panel = d3.select("#edit-panel");
  if (activeArrow) {
    panel.classed("open", true);
    d3.select("#input-height").property("value", activeArrow.height);
    d3.select("#input-bodypart").property("value", activeArrow.bodyPart);
    d3.select("#input-pressure").property("checked", activeArrow.underPressure);
    d3.select("#input-duration").property("value", activeArrow.duration);
    d3.select("#duration-val").text(activeArrow.duration.toFixed(1));
  } else {
    panel.classed("open", false);
  }

  // List[cite: 5]
  const totalPages = Math.max(1, Math.ceil(arrows.length / pageSize));
  if (currentPage >= totalPages) currentPage = totalPages - 1;
  
  const start = currentPage * pageSize;
  const pagedData = arrows.slice(start, start + pageSize);

  const items = d3.select("#menu-list").selectAll(".menu-item").data(pagedData, d => d.id);
  items.enter().append("div").attr("class", "menu-item")
    .merge(items)
    .classed("active", d => d.id === activeId)
    .html(d => `<strong>Pass</strong>: ${d.height}<br><small>${d.x1.toFixed(0)},${d.y1.toFixed(0)} &rarr; ${d.x2.toFixed(0)},${d.y2.toFixed(0)}</small>`)
    .on("click", (e, d) => { activeId = d.id; render(); });
  items.exit().remove();

  renderPaginationControls(totalPages);
}

function renderPaginationControls(totalPages) {
  let ctrl = d3.select("#pagination-controls").html("");
  if (totalPages <= 1) return;
  ctrl.append("button").text("PREV").property("disabled", currentPage === 0).on("click", () => { currentPage--; render(); });
  ctrl.append("span").text(`${currentPage + 1} / ${totalPages}`);
  ctrl.append("button").text("NEXT").property("disabled", currentPage >= totalPages - 1).on("click", () => { currentPage++; render(); });
}

render(); // Initial call