import type { PassArrow } from "../models/PassArrow";
import type { ArrowProperties, Point } from "../types";

export class PlayStore {
  pageSize: number;
  arrows: PassArrow[];
  activeId: string | null;
  currentPage: number;
  tempStart: Point | null;

  constructor(pageSize: number) {
    this.pageSize = pageSize;
    this.arrows = [];
    this.activeId = null;
    this.currentPage = 0;
    this.tempStart = null;
  }

  setTempStart(point: Point) {
    this.tempStart = point;
  }

  clearTempStart() {
    this.tempStart = null;
  }

  addArrow(arrow: PassArrow) {
    this.arrows.push(arrow);
    this.activeId = arrow.id;
    this.currentPage = Math.max(0, Math.ceil(this.arrows.length / this.pageSize) - 1);
  }

  setActiveArrow(id: string) {
    this.activeId = id;
  }

  clearActiveArrow() {
    this.activeId = null;
  }

  getActiveArrow() {
    return this.arrows.find((arrow) => arrow.id === this.activeId) ?? null;
  }

  updateActiveArrow(properties: Partial<ArrowProperties>) {
    const activeArrow = this.getActiveArrow();
    if (!activeArrow) {
      return false;
    }

    Object.assign(activeArrow, properties);
    return true;
  }

  deleteActiveArrow() {
    if (!this.activeId) {
      return;
    }

    this.arrows = this.arrows.filter((arrow) => arrow.id !== this.activeId);
    this.activeId = null;
  }

  clearAll() {
    this.arrows = [];
    this.activeId = null;
    this.currentPage = 0;
    this.tempStart = null;
  }

  getTotalPages() {
    return Math.max(1, Math.ceil(this.arrows.length / this.pageSize));
  }

  getPagedArrows(): PassArrow[] {
    const start = this.currentPage * this.pageSize;
    return this.arrows.slice(start, start + this.pageSize);
  }

  goToPrevPage() {
    this.currentPage = Math.max(0, this.currentPage - 1);
  }

  goToNextPage() {
    const totalPages = this.getTotalPages();
    this.currentPage = Math.min(totalPages - 1, this.currentPage + 1);
  }

  ensurePageBounds() {
    const totalPages = this.getTotalPages();
    if (this.currentPage >= totalPages) {
      this.currentPage = totalPages - 1;
    }
  }
}
