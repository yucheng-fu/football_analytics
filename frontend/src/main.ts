import { APP_CONFIG } from "./config/constants";
import { PassArrow } from "./models/PassArrow";
import { PitchRenderer } from "./renderers/PitchRenderer";
import { PassPredictionService } from "./services/passPredictionService";
import { PlayStore } from "./state/PlayStore";
import { UIController } from "./ui/UIController";
import { createArrowId } from "./utils/id";
import { toPassPredictionPayload } from "./utils/passPredictionPayload";

class PassDesignerApp {
  store: PlayStore;
  pitchRenderer: PitchRenderer;
  ui: UIController;
  predictionService: PassPredictionService;

  constructor() {
    this.store = new PlayStore(APP_CONFIG.pageSize);
    this.pitchRenderer = new PitchRenderer({
      canvasSelector: "#canvas",
      scale: APP_CONFIG.scale,
      pitchWidth: APP_CONFIG.pitchWidth,
      pitchHeight: APP_CONFIG.pitchHeight,
    });
    this.ui = new UIController();
    this.predictionService = new PassPredictionService();
  }

  init() {
    this.pitchRenderer.bindCanvasEvents({
      onMouseMove: (event) => this.handleMouseMove(event),
      onCanvasClick: (event) => this.handleCanvasClick(event),
    });

    this.ui.bindFormChange(() => this.handleFormUpdate());
    this.ui.bindDeleteActive(() => this.handleDeleteActive());
    this.ui.bindClearAll(() => this.handleClearAll());
    this.ui.bindClosePanel(() => this.handleClosePanel());
    this.ui.bindPredictActive(() => {
      void this.handlePredictActive();
    });
    window.addEventListener("keydown", (event) => this.handleKeyDown(event));

    this.render();
  }

  handleMouseMove(event: MouseEvent) {
    if (!this.store.tempStart) {
      return;
    }

    const pointer = this.pitchRenderer.getPointer(event);
    this.pitchRenderer.updateGhostLine(this.store.tempStart, { x: pointer.rawX, y: pointer.rawY });
  }

  handleCanvasClick(event: MouseEvent) {
    const pointer = this.pitchRenderer.getPointer(event);

    if (!this.store.tempStart) {
      this.store.setTempStart({ x: pointer.scaledX, y: pointer.scaledY });
      return;
    }

    const newArrow = new PassArrow({
      id: createArrowId(),
      x1: this.store.tempStart.x,
      y1: this.store.tempStart.y,
      x2: pointer.scaledX,
      y2: pointer.scaledY,
    });

    this.store.addArrow(newArrow);
    this.store.clearTempStart();
    this.pitchRenderer.hideGhostLine();
    this.render();
  }

  handleFormUpdate() {
    const values = this.ui.getArrowFormValues();
    const updated = this.store.updateActiveArrow(values);

    if (!updated) {
      return;
    }

    this.ui.setDurationText(values.duration);
    this.render();
  }

  handleDeleteActive() {
    this.store.deleteActiveArrow();
    this.render();
  }

  handleClearAll() {
    if (!window.confirm("Delete all plays?")) {
      return;
    }

    this.store.clearAll();
    this.render();
  }

  handleClosePanel() {
    this.store.clearActiveArrow();
    this.render();
  }

  handleKeyDown(event: KeyboardEvent) {
    if (event.key !== "Escape") {
      return;
    }

    if (!this.store.tempStart) {
      return;
    }

    this.store.clearTempStart();
    this.pitchRenderer.cancelPendingArrow();
    this.ui.setPredictionStatus("Arrow drawing cancelled.");
  }

  async handlePredictActive() {
    const activeArrow = this.store.getActiveArrow();
    if (!activeArrow) {
      this.ui.setPredictionStatus("Select a pass before predicting.");
      return;
    }

    this.ui.setPredictButtonLoading(true);
    this.ui.setPredictionStatus("Requesting prediction...");

    try {
      const payload = toPassPredictionPayload(activeArrow);
      const response = await this.predictionService.predictPass(payload);
      activeArrow.prediction = response.prediction;
      activeArrow.probability = response.probability;
      activeArrow.predictedAt = response.timestamp;
      this.ui.setPredictionStatus("Prediction saved on selected pass.");
      this.render();
    } catch (error) {
      const message = error instanceof Error ? error.message : "Prediction failed.";
      this.ui.setPredictionStatus(`Prediction failed: ${message}`);
    } finally {
      this.ui.setPredictButtonLoading(false);
    }
  }

  render() {
    this.store.ensurePageBounds();
    const activeArrow = this.store.getActiveArrow();

    this.pitchRenderer.renderArrows(this.store.arrows, this.store.activeId, (id) => {
      this.store.setActiveArrow(id);
      this.render();
    });

    this.ui.renderEditPanel(activeArrow);
    this.ui.renderPlayList(this.store.getPagedArrows(), this.store.activeId, (id) => {
      this.store.setActiveArrow(id);
      this.render();
    });

    this.ui.renderPaginationControls({
      currentPage: this.store.currentPage,
      totalPages: this.store.getTotalPages(),
      onPrev: () => {
        this.store.goToPrevPage();
        this.render();
      },
      onNext: () => {
        this.store.goToNextPage();
        this.render();
      },
    });
  }
}

const app = new PassDesignerApp();
app.init();
