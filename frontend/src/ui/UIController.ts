import * as d3 from "d3";
import type { ArrowProperties } from "../types";
import type { PassArrow } from "../models/PassArrow";

interface PaginationConfig {
  currentPage: number;
  totalPages: number;
  onPrev: () => void;
  onNext: () => void;
}

interface ListArrowView {
  id: string;
  height: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export class UIController {
  editPanel: any;
  heightInput: any;
  bodyPartInput: any;
  pressureInput: any;
  durationInput: any;
  durationValue: any;
  menuList: any;
  paginationControls: any;
  deleteButton: any;
  clearAllButton: any;
  closePanelButton: any;
  predictButton: any;
  predictionStatus: any;

  constructor() {
    this.editPanel = d3.select("#edit-panel");
    this.heightInput = d3.select("#input-height");
    this.bodyPartInput = d3.select("#input-bodypart");
    this.pressureInput = d3.select("#input-pressure");
    this.durationInput = d3.select("#input-duration");
    this.durationValue = d3.select("#duration-val");
    this.menuList = d3.select("#menu-list");
    this.paginationControls = d3.select("#pagination-controls");
    this.deleteButton = d3.select("#btn-delete-active");
    this.clearAllButton = d3.select("#btn-clear-all");
    this.closePanelButton = d3.select("#close-panel");
    this.predictButton = d3.select("#btn-predict-active");
    this.predictionStatus = d3.select("#prediction-status");
  }

  bindFormChange(handler: () => void) {
    d3.selectAll("#input-height, #input-bodypart, #input-pressure, #input-duration").on("input", handler);
  }

  bindDeleteActive(handler: () => void) {
    this.deleteButton.on("click", handler);
  }

  bindClearAll(handler: () => void) {
    this.clearAllButton.on("click", handler);
  }

  bindClosePanel(handler: () => void) {
    this.closePanelButton.on("click", handler);
  }

  bindPredictActive(handler: () => void) {
    this.predictButton.on("click", handler);
  }

  getArrowFormValues(): ArrowProperties {
    return {
      height: this.heightInput.property("value"),
      bodyPart: this.bodyPartInput.property("value"),
      underPressure: this.pressureInput.property("checked"),
      duration: parseFloat(this.durationInput.property("value")),
    };
  }

  setDurationText(duration: number) {
    this.durationValue.text(duration.toFixed(1));
  }

  renderEditPanel(activeArrow: PassArrow | null) {
    if (!activeArrow) {
      this.editPanel.classed("open", false);
      this.predictionStatus.text("");
      return;
    }

    this.editPanel.classed("open", true);
    this.heightInput.property("value", activeArrow.height);
    this.bodyPartInput.property("value", activeArrow.bodyPart);
    this.pressureInput.property("checked", activeArrow.underPressure);
    this.durationInput.property("value", activeArrow.duration);
    this.setDurationText(activeArrow.duration);
    this.predictionStatus.text(this.getPredictionStatus(activeArrow));
  }

  renderPlayList(arrows: ListArrowView[], activeId: string | null, onItemClick: (id: string) => void) {
    const items = this.menuList.selectAll(".menu-item").data(arrows, (d: ListArrowView) => d.id);

    items
      .enter()
      .append("div")
      .attr("class", "menu-item")
      .merge(items)
      .classed("active", (d: ListArrowView) => d.id === activeId)
      .html(
        (d: ListArrowView) =>
          `<strong>Pass</strong>: ${d.height}<br><small>(${d.x1.toFixed(0)}, ${d.y1.toFixed(0)}) &rarr; (${d.x2.toFixed(0)}, ${d.y2.toFixed(0)})</small>`,
      )
      .on("click", (_event: MouseEvent, d: ListArrowView) => onItemClick(d.id));

    items.exit().remove();
  }

  renderPaginationControls({ currentPage, totalPages, onPrev, onNext }: PaginationConfig) {
    const controls = this.paginationControls.html("");
    if (totalPages <= 1) {
      return;
    }

    controls
      .append("button")
      .text("PREV")
      .property("disabled", currentPage === 0)
      .on("click", onPrev);

    controls.append("span").text(`${currentPage + 1} / ${totalPages}`);

    controls
      .append("button")
      .text("NEXT")
      .property("disabled", currentPage >= totalPages - 1)
      .on("click", onNext);
  }

  setPredictionStatus(message: string) {
    this.predictionStatus.text(message);
  }

  setPredictButtonLoading(isLoading: boolean) {
    this.predictButton.property("disabled", isLoading);
    this.predictButton.text(isLoading ? "Predicting..." : "Predict Pass Success");
  }

  private getPredictionStatus(activeArrow: PassArrow) {
    if (activeArrow.prediction === null || activeArrow.probability === null) {
      return "No prediction yet.";
    }

    const classLabel = activeArrow.prediction === 1 ? "success" : "failure";
    const classProbability = Math.round(activeArrow.probability * 100);
    return `Prediction: ${classLabel}. (Confidence: ${classProbability}%)` 
  }
}
