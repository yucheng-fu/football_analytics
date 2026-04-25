const scale = 5;

const pitchWidth = 120;
const pitchHeight = 80;

const svg = d3.select("#canvas")
  .attr("width", pitchWidth * scale)
  .attr("height", pitchHeight * scale);

// helper scale functions
const x = d => d * scale;
const y = d => d * scale;

// background
svg.append("rect")
  .attr("width", x(pitchWidth))
  .attr("height", y(pitchHeight))
  .attr("fill", "#ffffff");

// outer boundary
svg.append("rect")
  .attr("x", x(0))
  .attr("y", y(0))
  .attr("width", x(120))
  .attr("height", y(80))
  .attr("fill", "none")
  .attr("stroke", "black");

// halfway line
svg.append("line")
  .attr("x1", x(60))
  .attr("y1", y(0))
  .attr("x2", x(60))
  .attr("y2", y(80))
  .attr("stroke", "black");

// center circle (radius = 10 yards ≈ 10 units here)
svg.append("circle")
  .attr("cx", x(60))
  .attr("cy", y(40))
  .attr("r", x(10))
  .attr("fill", "none")
  .attr("stroke", "black");

// center spot
svg.append("circle")
  .attr("cx", x(60))
  .attr("cy", y(40))
  .attr("r", 1.5)
  .attr("fill", "black");

// penalty areas
// left
svg.append("rect")
  .attr("x", x(0))
  .attr("y", y(18))
  .attr("width", x(18))
  .attr("height", y(44))
  .attr("fill", "none")
  .attr("stroke", "black");

// right
svg.append("rect")
  .attr("x", x(102))
  .attr("y", y(18))
  .attr("width", x(18))
  .attr("height", y(44))
  .attr("fill", "none")
  .attr("stroke", "black");

// 6-yard boxes
svg.append("rect")
  .attr("x", x(0))
  .attr("y", y(30))
  .attr("width", x(6))
  .attr("height", y(20))
  .attr("fill", "none")
  .attr("stroke", "black");

svg.append("rect")
  .attr("x", x(114))
  .attr("y", y(30))
  .attr("width", x(6))
  .attr("height", y(20))
  .attr("fill", "none")
  .attr("stroke", "black");

// penalty spots
svg.append("circle")
  .attr("cx", x(12))
  .attr("cy", y(40))
  .attr("r", 1.5)
  .attr("fill", "black");

svg.append("circle")
  .attr("cx", x(108))
  .attr("cy", y(40))
  .attr("r", 1.5)
  .attr("fill", "black");