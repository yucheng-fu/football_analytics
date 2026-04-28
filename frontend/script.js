// --- 1. CONFIGURATION & STATE ---
const scale = 5;
const pitchWidth = 120;
const pitchHeight = 80;

let arrows = [];
let tempStart = null;
let activeId = null;

// Pagination State
let currentPage = 0;
const pageSize = 5; 

const svg = d3.select("#canvas")
  .attr("width", pitchWidth * scale)
  .attr("height", pitchHeight * scale);

const x = val => val * scale;
const y = val => val * scale;

// --- 2. MARKER DEFINITIONS (Arrowheads) ---
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

// --- 3. DRAW THE PITCH ---
const drawPitch = () => {
  const g = svg.append("g").attr("stroke", "#333").attr("fill", "none").attr("stroke-width", 2);
  // Outer Boundary & Halfway
  g.append("rect").attr("width", x(120)).attr("height", y(80));
  g.append("line").attr("x1", x(60)).attr("y1", 0).attr("x2", x(60)).attr("y2", y(80));
  g.append("circle").attr("cx", x(60)).attr("cy", y(40)).attr("r", x(10));
  // Penalty Areas
  g.append("rect").attr("x", x(0)).attr("y", y(18)).attr("width", x(18)).attr("height", y(44));
  g.append("rect").attr("x", x(102)).attr("y", y(18)).attr("width", x(18)).attr("height", y(44));
  // Goal Zones (6-yard boxes)
  g.append("rect").attr("x", x(0)).attr("y", y(30)).attr("width", x(6)).attr("height", y(20));
  g.append("rect").attr("x", x(114)).attr("y", y(30)).attr("width", x(6)).attr("height", y(20));
};
drawPitch();

// --- 4. GHOST ARROW (The Preview) ---
const ghost = svg.append("line")
  .attr("stroke", "#aaa").attr("stroke-dasharray", "4")
  .attr("marker-end", "url(#head-ghost)")
  .style("visibility", "hidden");

// --- 5. INTERACTION LOGIC ---
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
      x2: mX / scale, y2: mY / scale 
    };
    arrows.push(newArrow);
    tempStart = null;
    ghost.style("visibility", "hidden");
    
    // Jump to the last page to see the new arrow
    currentPage = Math.max(0, Math.ceil(arrows.length / pageSize) - 1);
    render();
  }
});

// --- 6. RENDERING ---
function render() {
  // Line Drawing (Always render ALL arrows on pitch)
  const lines = svg.selectAll(".arrow-line").data(arrows, d => d.id);
  
  lines.enter().append("line")
    .attr("class", "arrow-line")
    .merge(lines)
    .attr("x1", d => x(d.x1)).attr("y1", d => y(d.y1))
    .attr("x2", d => x(d.x2)).attr("y2", d => y(d.y2))
    .attr("stroke", d => d.id === activeId ? "#ff4444" : "#333")
    .attr("stroke-width", d => d.id === activeId ? 4 : 2)
    .attr("marker-end", d => d.id === activeId ? "url(#head-red)" : "url(#head-black)");

  lines.exit().remove();

  // Sidebar List (Paged)
  const totalPages = Math.ceil(arrows.length / pageSize);
  const start = currentPage * pageSize;
  const pagedData = arrows.slice(start, start + pageSize);

  const items = d3.select("#menu-list").selectAll(".menu-item").data(pagedData, d => d.id);
  
  items.enter().append("div")
    .attr("class", "menu-item")
    .merge(items)
    .classed("active", d => d.id === activeId)
    .html(d => `<strong>Pass</strong>: ${d.x1.toFixed(0)},${d.y1.toFixed(0)} &rarr; ${d.x2.toFixed(0)},${d.y2.toFixed(0)}`)
    .on("click", (e, d) => {
      activeId = d.id;
      render();
    });

  items.exit().remove();

  renderPaginationControls(totalPages);
}

function renderPaginationControls(totalPages) {
  let controls = d3.select("#pagination-controls");
  if (controls.empty()) {
    controls = d3.select("#sidebar").append("div").attr("id", "pagination-controls");
  }

  controls.html("");
  if (totalPages <= 1) return;

  controls.append("button")
    .text("PREV")
    .property("disabled", currentPage === 0)
    .on("click", () => { currentPage--; render(); });

  controls.append("span")
    .text(` Page ${currentPage + 1} of ${totalPages} `);

  controls.append("button")
    .text("NEXT")
    .property("disabled", currentPage >= totalPages - 1)
    .on("click", () => { currentPage++; render(); });
}