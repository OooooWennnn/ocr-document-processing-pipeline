"use client";
/* eslint-disable @next/next/no-img-element -- The original page and editable regions share exact coordinates. */
import {
  type CSSProperties,
  type SyntheticEvent,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  ChevronLeft,
  ChevronRight,
  Download,
  FileText,
  Save,
  Undo2,
} from "lucide-react";
import {
  Box,
  getResult,
  JobResponse,
  saveDocument,
  sourceImageUrl,
} from "@/lib/api";
import {
  documentHtml,
  exportDocument,
  pageDimensions,
  regionFontSize,
  regionTextLayout,
} from "@/lib/document-export";

type Drafts = Record<number, Record<string, string>>;

const LOW_CONFIDENCE_SCORE = 0.85;

function getPositionStyle(box: Box, container: Box): CSSProperties {
  const containerWidth = container.x_max - container.x_min;
  const containerHeight = container.y_max - container.y_min;

  return {
    left: `${((box.x_min - container.x_min) / containerWidth) * 100}%`,
    top: `${((box.y_min - container.y_min) / containerHeight) * 100}%`,
    width: `${((box.x_max - box.x_min) / containerWidth) * 100}%`,
    height: `${((box.y_max - box.y_min) / containerHeight) * 100}%`,
  };
}

/** Edit positioned text and cells, keeping page drafts during navigation. */
export default function DocumentWorkspace({
  jobId,
  initial,
}: {
  jobId: number;
  initial: JobResponse;
}) {
  const [documentData, setDocumentData] = useState(initial);
  const [activePageIndex, setActivePageIndex] = useState(0);
  const [drafts, setDrafts] = useState<Drafts>({});
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null);
  const [view, setView] = useState<"document" | "source">("document");
  const [zoom, setZoom] = useState("fit");
  const [showRegions, setShowRegions] = useState(true);
  const [showScores, setShowScores] = useState(true);
  const [canvasScale, setCanvasScale] = useState(1);
  const [canvasReady, setCanvasReady] = useState(false);
  const [sourceImageSize, setSourceImageSize] = useState({ width: 0, height: 0 });
  const [sourceImageError, setSourceImageError] = useState(false);
  const [saving, setSaving] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [saveMessage, setSaveMessage] = useState("");
  const paperRef = useRef<HTMLDivElement>(null);
  const editorRef = useRef<HTMLTextAreaElement>(null);

  const pages = documentData.pages ?? [];
  const activePage = pages[activePageIndex];
  const dimensions = activePage
    ? pageDimensions(activePage, sourceImageSize)
    : { width: 800, height: 1100 };
  const unsavedChangeCount = Object.values(drafts).reduce(
    (total, pageDrafts) => total + Object.keys(pageDrafts).length,
    0,
  );

  // Fetch positioned regions for older results without running OCR again.
  useEffect(() => {

    // Refresh cached results from the previous split view without running OCR again.
    const hasPositionedRegions = initial.pages?.every(
      (documentPage) => documentPage.regions !== undefined,
    );

    if (hasPositionedRegions) {
      return;
    }

    let isMounted = true;

    getResult(jobId)
      .then((result) => {
        if (isMounted) setDocumentData(result);
      })
      .catch((error) => {
        if (isMounted) {
          setErrorMessage(
            error instanceof Error
              ? error.message
              : "Could not load the document layout.",
          );
        }
      });

    return () => {
      isMounted = false;
    };
  }, [jobId, initial]);

  // Scale token fonts to the displayed page width.
  useEffect(() => {
    const element = paperRef.current;

    if (!element) {
      return;
    }

    const observer = new ResizeObserver(() => {
      setCanvasScale(element.getBoundingClientRect().width / dimensions.width);
      setCanvasReady(true);
    });

    observer.observe(element);

    return () => observer.disconnect();
  }, [dimensions.width, activePageIndex]);

  // Warn before leaving only while unsaved changes exist.
  useEffect(() => {
    if (!unsavedChangeCount) {
      return;
    }

    const handleBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
    };

    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => window.removeEventListener("beforeunload", handleBeforeUnload);
  }, [unsavedChangeCount]);

  if (!activePage) {
    return <p role="alert">No document page is available.</p>;
  }

  const pageDrafts = drafts[activePage.page_index];
  const pageBounds: Box = {
    x_min: 0,
    y_min: 0,
    x_max: dimensions.width,
    y_max: dimensions.height,
  };
  const regions = (activePage.regions ?? []).map((region) => ({
    ...region,
    edited: region.edited || pageDrafts?.[region.id] !== undefined,
    value: pageDrafts?.[region.id] ?? region.value,
  }));
  const selectedRegion = regions.find((region) => region.id === selectedRegionId);
  const textRegions = showScores ? regions.filter((region) => region.value) : [];
  const reviewCount = textRegions.filter(
    (region) =>
      region.score == null ||
      region.score < LOW_CONFIDENCE_SCORE ||
      region.mapping_warning,
  ).length;

  /** Select a region for both the source highlight and editor. */
  const selectRegion = (id: string) => {
    setSelectedRegionId(id);
    setSaveMessage("");
  };

  /** Update the draft value; remove it when it matches the saved value. */
  const updateDraft = (id: string, value: string) => {
    const savedRegion = activePage.regions?.find((region) => region.id === id);
    const pageDrafts = { ...drafts[activePage.page_index] };

    if (value === savedRegion?.value) {
      delete pageDrafts[id];
    } else {
      pageDrafts[id] = value;
    }

    setDrafts({ ...drafts, [activePage.page_index]: pageDrafts });
    setSaveMessage("");
  };

  /** Save changed regions across pages, then refresh the result and clear drafts. */
  const saveChanges = async () => {
    setSaving(true);
    setErrorMessage("");
    setSaveMessage("");

    try {
      const pageUpdates = Object.entries(drafts)
        .filter(([, pageDrafts]) => Object.keys(pageDrafts).length)
        .map(([index, pageDrafts]) => ({
          page_index: Number(index),
          regions: Object.entries(pageDrafts).map(([id, value]) => ({ id, value })),
        }));

      const savedDocument = await saveDocument(jobId, pageUpdates);

      setDocumentData(savedDocument);
      setDrafts({});
      setSaveMessage(
        "Document saved. Text and table cells share the same page layout.",
      );
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Could not save the document.",
      );
    } finally {
      setSaving(false);
    }
  };

  /** Download HTML or JSON with current edits; this does not save to the database. */
  const downloadDocument = (format: "html" | "json") => {
    const content =
      format === "html"
        ? documentHtml(documentData, drafts)
        : JSON.stringify(exportDocument(documentData, drafts), null, 2);

    const url = URL.createObjectURL(
      new Blob([content], {
        type:
          format === "html" ? "text/html;charset=utf-8" : "application/json",
      }),
    );

    const link = window.document.createElement("a");
    link.href = url;
    link.download = `document-${jobId}.${format}`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  /** Change pages and reset selection and preview state while preserving drafts. */
  const changePage = (targetPageIndex: number) => {
    setActivePageIndex(targetPageIndex);
    setSelectedRegionId(null);
    setSourceImageSize({ width: 0, height: 0 });
    setSourceImageError(false);
  };

  const handleSourceImageLoad = (event: SyntheticEvent<HTMLImageElement>) => {
    const width = event.currentTarget.naturalWidth;
    const height = event.currentTarget.naturalHeight;

    setSourceImageSize({ width, height });

    if (!activePage.image_width || !activePage.image_height) {
      setDocumentData((previous) => ({
        ...previous,
        pages: previous.pages?.map((documentPage) =>
          documentPage.page_index === activePage.page_index
            ? { ...documentPage, image_width: width, image_height: height }
            : documentPage,
        ),
      }));
    }

    setSourceImageError(false);
  };

  const discardChanges = () => {
    setDrafts({});
    setSaveMessage("");
  };

  return (
    <section className="unified-workspace">
      <div className="document-toolbar">
        <div>
          <p className="eyebrow">ONE DOCUMENT / EVERY REGION IN PLACE</p>
          <h2>Your extracted document.</h2>
          <p className="muted small">
            Titles, addresses, notes and table cells stay in their original
            positions. Click any region to edit.
          </p>
        </div>
        <div className="unified-export">
          <button
            className="button secondary"
            onClick={() => downloadDocument("html")}
          >
            <FileText size={15} />
            Export document
          </button>
          <button
            className="button quiet"
            onClick={() => downloadDocument("json")}
          >
            <Download size={15} />
            JSON
          </button>
        </div>
      </div>
      <nav className="page-navigation" aria-label="Document pages">
        <button
          className="button quiet"
          disabled={saving || activePageIndex === 0}
          onClick={() => changePage(activePageIndex - 1)}
        >
          <ChevronLeft size={16} />
          Previous
        </button>
        <label>
          Page{" "}
          <select
            aria-label="Document page"
            disabled={saving}
            value={activePageIndex}
            onChange={(event) => changePage(Number(event.target.value))}
          >
            {pages.map((documentPage, pageOptionIndex) => (
              <option key={documentPage.page_index} value={pageOptionIndex}>
                {documentPage.page_index + 1} of {pages.length}
                {Object.keys(drafts[documentPage.page_index] ?? {}).length
                  ? " • edited"
                  : ""}
              </option>
            ))}
          </select>
        </label>
        <button
          className="button quiet"
          disabled={saving || activePageIndex === pages.length - 1}
          onClick={() => changePage(activePageIndex + 1)}
        >
          Next
          <ChevronRight size={16} />
        </button>
      </nav>
      {!!activePage.warnings?.length && (
        <div className="notice warning">{activePage.warnings.join(" ")}</div>
      )}
      {errorMessage && (
        <div className="notice error" role="alert">
          {errorMessage}
        </div>
      )}
      <div className="unified-layout">
        <section className="panel document-canvas-panel">
          <div className="canvas-toolbar">
            <div
              className="view-switch"
              role="group"
              aria-label="Document view"
            >
              <button
                className={view === "document" ? "active" : ""}
                onClick={() => setView("document")}
              >
                Extracted document
              </button>
              <button
                className={view === "source" ? "active" : ""}
                onClick={() => setView("source")}
              >
                Original source
              </button>
            </div>
            <div className="canvas-tools">
              <label>
                <input
                  type="checkbox"
                  checked={showRegions}
                  onChange={(event) => setShowRegions(event.target.checked)}
                />
                Region outlines
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={showScores}
                  onChange={(event) => setShowScores(event.target.checked)}
                />
                Field scores
              </label>
              <select
                aria-label="Document zoom"
                value={zoom}
                onChange={(event) => setZoom(event.target.value)}
              >
                {["fit", "100", "150", "200"].map((zoomOption) => (
                  <option key={zoomOption} value={zoomOption}>
                    {zoomOption === "fit" ? "Fit page" : `${zoomOption}%`}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div className="document-canvas-scroll">
            <div
              ref={paperRef}
              className={`document-paper ${view} ${showRegions ? "show-outlines" : ""}`}
              style={{
                width:
                  zoom === "fit"
                    ? "100%"
                    : `${(dimensions.width * Number(zoom)) / 100}px`,
                aspectRatio: `${dimensions.width}/${dimensions.height}`,
              }}
            >
              <img
                className="document-source"
                src={sourceImageUrl(jobId, activePage.page_index)}
                alt={`Original document page ${activePage.page_index + 1}`}
                onLoad={handleSourceImageLoad}
                onError={() => setSourceImageError(true)}
              />
              {regions.map((region) => {
                const fontSize = regionFontSize(region, canvasReady) * canvasScale;
                const textLayout = regionTextLayout(region);
                const regionWidth = Math.max(1, region.bbox.x_max - region.bbox.x_min);
                const regionHeight = Math.max(1, region.bbox.y_max - region.bbox.y_min);
                const isSelected = selectedRegionId === region.id;
                const hasDraft = pageDrafts?.[region.id] !== undefined;
                const needsReview =
                  region.mapping_warning ||
                  (region.value &&
                    (region.score == null || region.score < LOW_CONFIDENCE_SCORE));
                const ocrScore =
                  region.score == null
                    ? "unknown"
                    : `${Math.round(region.score * 100)}%`;
                const mappingScore =
                  region.mapping_score == null
                    ? "unknown"
                    : `${Math.round(region.mapping_score * 100)}%`;
                const regionLocation =
                  region.kind === "cell"
                    ? `Table ${region.table_id}, row ${(region.row ?? 0) + 1}, column ${(region.col ?? 0) + 1}`
                    : "Text";

                return (
                  <button
                    key={region.id}
                    type="button"
                    disabled={saving}
                    title={`${region.value || "Empty table cell"} · OCR ${ocrScore} · Mapping ${mappingScore}`}
                    aria-label={`${regionLocation}: ${region.value || "empty"}`}
                    aria-pressed={isSelected}
                    className={`document-region ${region.kind} ${needsReview ? "needs-review" : ""} ${isSelected ? "selected" : ""} ${hasDraft ? "changed" : ""}`}
                    style={{
                      ...getPositionStyle(region.bbox, pageBounds),
                      fontSize: `${fontSize}px`,
                    }}
                    onClick={() => selectRegion(region.id)}
                    onDoubleClick={() => {
                      selectRegion(region.id);
                      editorRef.current?.focus();
                    }}
                  >
                    {!region.edited && region.tokens?.length ? (
                      region.tokens.map((token, tokenIndex) => (

                        <span
                          key={tokenIndex}
                          className="positioned-token"
                          style={{
                            ...getPositionStyle(token.bbox, region.bbox),
                          }}
                        >
                          {token.value}
                        </span>
                      ))
                    ) : (
                      <span
                        className="region-value"
                        style={{
                          left: `${(textLayout.left / regionWidth) * 100}%`,
                          right: `${(textLayout.right / regionWidth) * 100}%`,
                          top: `${(textLayout.top / regionHeight) * 100}%`,
                          bottom: `${(textLayout.bottom / regionHeight) * 100}%`,
                          textAlign: textLayout.textAlign,
                        }}
                      >
                        {region.value}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
          {sourceImageError && view === "source" && (
            <p className="field-error" role="alert">
              The source image is unavailable. Extracted regions remain
              available.
            </p>
          )}
          <div className="canvas-legend">
            <span>
              <i className="legend-dot review" />
              Needs review
            </span>
            <span>
              <i className="legend-dot changed" />
              Unsaved change
            </span>
            <span>
              {regions.length} regions · Page {activePage.page_index + 1}
            </span>
          </div>
        </section>
        <aside className="panel unified-inspector">
          <div className="source-comparison">
            <p className="eyebrow">ORIGINAL PAGE</p>
            <button
              className="source-preview"
              aria-label="Show original page"
              onClick={() => setView("source")}
            >
              <img
                src={sourceImageUrl(jobId, activePage.page_index)}
                alt="Original page for comparison"
              />
              {selectedRegion && (
                <i
                  style={{
                    ...getPositionStyle(selectedRegion.bbox, pageBounds),
                  }}
                />
              )}
            </button>
            {selectedRegion && (
              <>
                <p className="eyebrow">SELECTED SOURCE AREA</p>
                <svg
                  role="img"
                  aria-label="Magnified original selected region"
                  viewBox={`${selectedRegion.bbox.x_min - 3} ${selectedRegion.bbox.y_min - 3} ${selectedRegion.bbox.x_max - selectedRegion.bbox.x_min + 6} ${selectedRegion.bbox.y_max - selectedRegion.bbox.y_min + 6}`}
                >
                  <image
                    href={sourceImageUrl(jobId, activePage.page_index)}
                    width={dimensions.width}
                    height={dimensions.height}
                  />
                </svg>
              </>
            )}
          </div>
          <div className="panel-heading">
            <span className="eyebrow">SELECTED DOCUMENT REGION</span>
          </div>
          {selectedRegion ? (
            <div className="region-editor">
              <h3>
                {selectedRegion.kind === "cell"
                  ? `Table ${selectedRegion.table_id} · Row ${(selectedRegion.row ?? 0) + 1} / Column ${(selectedRegion.col ?? 0) + 1}`
                  : "Document text"}
              </h3>
              <p className="muted small">
                The selected area is outlined on the document. Changes appear in
                that same position.
              </p>
              <label htmlFor="region-editor">Extracted text</label>
              <textarea
                ref={editorRef}
                id="region-editor"
                aria-label="Selected region text"
                value={selectedRegion.value}
                maxLength={10000}
                disabled={saving}
                rows={6}
                onChange={(event) => updateDraft(selectedRegion.id, event.target.value)}
              />
              <dl>
                <div>
                  <dt>Position on page</dt>
                  <dd>
                    x {Math.round(selectedRegion.bbox.x_min)} · y{" "}
                    {Math.round(selectedRegion.bbox.y_min)}
                  </dd>
                </div>
                <div>
                  <dt>Region size</dt>
                  <dd>
                    {Math.round(
                      selectedRegion.bbox.x_max - selectedRegion.bbox.x_min,
                    )}{" "}
                    ×{" "}
                    {Math.round(
                      selectedRegion.bbox.y_max - selectedRegion.bbox.y_min,
                    )}{" "}
                    px
                  </dd>
                </div>
                <div>
                  <dt>OCR confidence</dt>
                  <dd>
                    {selectedRegion.score == null
                      ? "Unavailable"
                      : `${Math.round(selectedRegion.score * 100)}%`}
                  </dd>
                </div>
              </dl>
              <dl>
                <div>
                  <dt>Mapping coverage</dt>
                  <dd>
                    {selectedRegion.mapping_score == null
                      ? "Unavailable"
                      : `${Math.round(selectedRegion.mapping_score * 100)}%`}
                  </dd>
                </div>
              </dl>
              {selectedRegion.recognition_refined && (
                <p className="muted small">
                  Targeted recognition applied. Compare with the source;
                  confidence is not a guarantee of correctness.
                </p>
              )}
              <p className="eyebrow">ORIGINAL EXTRACTED TEXT</p>
              <p className="original-text">
                {selectedRegion.original_value || "No text detected"}
              </p>
              {selectedRegion.mapping_warning && (
                <p className="notice warning">
                  This region needs a mapping review. Compare it with the
                  original source.
                </p>
              )}
            </div>
          ) : (
            <div className="region-empty">
              <FileText size={28} />
              <h3>Select anywhere on the document.</h3>
              <p>
                Click a title, address, note or table cell to see its original
                position and correct its text.
              </p>
            </div>
          )}
          {showScores && (
            <div className="field-score-list">
              <p className="eyebrow">
                FIELD CONFIDENCE ·{" "}
                {reviewCount}{" "}
                TO REVIEW
              </p>
              {textRegions.map((region) => (
                <button
                  key={region.id}
                  aria-pressed={selectedRegionId === region.id}
                  onClick={() => selectRegion(region.id)}
                  className={
                    region.mapping_warning || region.score == null || region.score < LOW_CONFIDENCE_SCORE
                      ? "review"
                      : ""
                  }
                >
                  <span>{region.value}</span>
                  <strong>
                    {region.score == null ? "?" : `${Math.round(region.score * 100)}%`}
                  </strong>
                </button>
              ))}
            </div>
          )}
        </aside>
      </div>
      <div className="save-bar">
        <span className="save-status" role="status">
          {saveMessage ||
            (unsavedChangeCount
              ? `${unsavedChangeCount} unsaved document ${unsavedChangeCount === 1 ? "change" : "changes"}`
              : "All document changes saved")}
        </span>
        <div className="save-actions">
          <button
            className="button quiet"
            disabled={!unsavedChangeCount || saving}
            onClick={discardChanges}
          >
            <Undo2 size={15} />
            Discard edits
          </button>
          <button
            className="button primary"
            disabled={!unsavedChangeCount || saving}
            onClick={saveChanges}
          >
            <Save size={15} />
            {saving ? "Saving…" : "Save document"}
          </button>
        </div>
      </div>
      <p className="muted small export-note">
        Export includes all pages and current edits in their original positions.
        HTML can be opened in a browser and printed. Original graphics and
        typography are available in the source view.
      </p>
    </section>
  );
}
