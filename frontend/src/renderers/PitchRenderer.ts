import * as d3 from "d3";
import type { PassArrow } from "../models/PassArrow";
import type { Point } from "../types";
import { probabilityToArrowColor } from "../utils/probabilityColor";

interface PitchRendererConfig {
  canvasSelector: string;
  scale: number;
  pitchWidth: number;
  pitchHeight: number;
}

interface PointerData {
  rawX: number;
  rawY: number;
  scaledX: number;
  scaledY: number;
}

interface CanvasHandlers {
  onMouseMove: (event: MouseEvent) => void;
  onCanvasClick: (event: MouseEvent) => void;
}

export class PitchRenderer {
  scale: number;
  pitchWidth: number;
  pitchHeight: number;
  svg: any;
  ghost: any;
  markerDefs: any;

  constructor({ canvasSelector, scale, pitchWidth, pitchHeight }: PitchRendererConfig) {
    this.scale = scale;
    this.pitchWidth = pitchWidth;
    this.pitchHeight = pitchHeight;

    this.svg = d3
      .select(canvasSelector)
      .attr("width", this.x(pitchWidth))
      .attr("height", this.y(pitchHeight));

    this.initMarkers();
    this.drawPitch();
    this.ghost = this.initGhostLine();
  }

  x(value: number) {
    return value * this.scale;
  }

  y(value: number) {
    return value * this.scale;
  }

  initMarkers() {
    const defs = this.svg.append("defs");
    this.markerDefs = defs;
    this.createMarker(defs, "head-ghost", "#aaa");
  }

  createMarker(defs: any, id: string, color: string) {
    defs
      .append("marker")
      .attr("id", id)
      .attr("viewBox", "0 0 10 10")
      .attr("refX", 8)
      .attr("refY", 5)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M 0 0 L 10 5 L 0 10 z")
      .attr("fill", color);
  }

  drawPitch() {
    const g = this.svg.append("g").attr("stroke", "#333").attr("fill", "none").attr("stroke-width", 2);
    g.append("rect").attr("width", this.x(120)).attr("height", this.y(80));
    g.append("line").attr("x1", this.x(60)).attr("y1", 0).attr("x2", this.x(60)).attr("y2", this.y(80));
    g.append("circle").attr("cx", this.x(60)).attr("cy", this.y(40)).attr("r", this.x(10));

    g.append("rect").attr("x", this.x(0)).attr("y", this.y(18)).attr("width", this.x(18)).attr("height", this.y(44));
    g.append("rect").attr("x", this.x(102)).attr("y", this.y(18)).attr("width", this.x(18)).attr("height", this.y(44));

    g.append("rect").attr("x", this.x(0)).attr("y", this.y(30)).attr("width", this.x(6)).attr("height", this.y(20));
    g.append("rect").attr("x", this.x(114)).attr("y", this.y(30)).attr("width", this.x(6)).attr("height", this.y(20));

    g.append("rect").attr("x", this.x(-2)).attr("y", this.y(36)).attr("width", this.x(2)).attr("height", this.y(8));
    g.append("rect").attr("x", this.x(120)).attr("y", this.y(36)).attr("width", this.x(2)).attr("height", this.y(8));

    g.append("line").attr("x1", this.x(0)).attr("y1", this.y(36)).attr("x2", this.x(0)).attr("y2", this.y(44));
    g.append("line").attr("x1", this.x(120)).attr("y1", this.y(36)).attr("x2", this.x(120)).attr("y2", this.y(44));

    g.append("circle").attr("cx", this.x(12)).attr("cy", this.y(40)).attr("r", this.x(0.6)).attr("fill", "#333");
    g.append("circle").attr("cx", this.x(108)).attr("cy", this.y(40)).attr("r", this.x(0.6)).attr("fill", "#333");
  }

  initGhostLine() {
    return this.svg
      .append("line")
      .attr("stroke", "#aaa")
      .attr("stroke-dasharray", "4")
      .attr("marker-end", "url(#head-ghost)")
      .style("visibility", "hidden");
  }

  updateGhostLine(start: Point, mousePoint: Point) {
    this.ghost
      .style("visibility", "visible")
      .attr("x1", this.x(start.x))
      .attr("y1", this.y(start.y))
      .attr("x2", mousePoint.x)
      .attr("y2", mousePoint.y);
  }

  hideGhostLine() {
    this.ghost.style("visibility", "hidden");
  }

  getPointer(event: MouseEvent): PointerData {
    const [rawX, rawY] = d3.pointer(event);
    return {
      rawX,
      rawY,
      scaledX: rawX / this.scale,
      scaledY: rawY / this.scale,
    };
  }

  bindCanvasEvents({ onMouseMove, onCanvasClick }: CanvasHandlers) {
    this.svg.on("mousemove", (event: MouseEvent) => onMouseMove(event));
    this.svg.on("click", (event: MouseEvent) => onCanvasClick(event));
  }

  cancelPendingArrow() {
    this.hideGhostLine();
  }

  getArrowColor(arrow: PassArrow) {
    return probabilityToArrowColor(arrow.getSuccessProbability());
  }

  syncArrowHeadMarkers(arrows: PassArrow[]) {
    const markers = this.markerDefs.selectAll(".arrow-head-marker").data(arrows, (d: PassArrow) => d.id);

    const merged = markers
      .enter()
      .append("marker")
      .attr("class", "arrow-head-marker")
      .attr("id", (d: PassArrow) => `head-${d.id}`)
      .attr("viewBox", "0 0 10 10")
      .attr("refX", 8)
      .attr("refY", 5)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto");

    merged.append("path").attr("d", "M 0 0 L 10 5 L 0 10 z");

    merged
      .merge(markers)
      .select("path")
      .attr("fill", (d: PassArrow) => this.getArrowColor(d));

    markers.exit().remove();
  }

  renderArrows(arrows: PassArrow[], activeId: string | null, onArrowClick: (id: string) => void) {
    this.syncArrowHeadMarkers(arrows);
    const lines = this.svg.selectAll(".arrow-line").data(arrows, (d: PassArrow) => d.id);

    lines
      .enter()
      .append("line")
      .attr("class", "arrow-line")
      .style("cursor", "pointer")
      .on("click", (event: MouseEvent, arrow: PassArrow) => {
        event.stopPropagation();
        onArrowClick(arrow.id);
      })
      .merge(lines)
      .attr("x1", (d: PassArrow) => this.x(d.x1))
      .attr("y1", (d: PassArrow) => this.y(d.y1))
      .attr("x2", (d: PassArrow) => this.x(d.x2))
      .attr("y2", (d: PassArrow) => this.y(d.y2))
      .attr("stroke", (d: PassArrow) => this.getArrowColor(d))
      .attr("stroke-width", (d: PassArrow) => (d.id === activeId ? 4 : 2))
      .attr("marker-end", (d: PassArrow) => `url(#head-${d.id})`);

    lines.exit().remove();
  }
}
