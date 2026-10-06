"use client";
import { AlertCircle, ArrowLeft, Check, LoaderCircle } from "lucide-react";
import FileUpload from "@/components/ocr/FileUpload";
import DocumentWorkspace from "@/components/ocr/DocumentWorkspace";
import { useOcrJob } from "@/hooks/useOcrJob";
/** Show upload, progress, error or editor views based on the job state. */
export default function Home() {
  const {
    jobId,
    status,
    resultData,
    uploading,
    handleUpload,
    error,
    reset,
    refetch,
  } = useOcrJob();

  let currentStep = 0;

  if (status === "done") {
    currentStep = 2;
  } else if (jobId !== null || uploading) {
    currentStep = 1;
  }

  const isProcessing =
    jobId !== null && ["uploaded", "queued", "processing"].includes(status);

  /** Clear the selected job after confirmation; unsaved edits may be lost. */
  const startNewDocument = () => {
    if (window.confirm("Start a new document? Any unsaved edits will be lost.")) {
      reset();
    }
  };

  return (
    <>
      <div className="page-heading">
        <div>
          <p className="eyebrow">EXTRACT / REVIEW / REFINE</p>
          <h1>
            Your documents,
            <br />
            <span>structured.</span>
          </h1>
          <p className="page-description">
            Read full documents and multi-page PDFs. Review text and every table
            against the source.
          </p>
        </div>
        <ol className="steps" aria-label="Extraction progress">
          {["Upload document", "Read pages", "Review & edit"].map(
            (label, stepIndex) => (
              <li
                key={label}
                className={stepIndex === currentStep ? "active" : stepIndex < currentStep ? "complete" : ""}
                aria-current={stepIndex === currentStep ? "step" : undefined}
              >
                <span className="step-number">
                  {stepIndex < currentStep ? <Check size={14} /> : `0${stepIndex + 1}`}
                </span>
                {label}
              </li>
            ),
          )}
        </ol>
      </div>
      {error && (
        <div className="notice error" role="alert">
          <AlertCircle size={18} />
          <span>{error}</span>
          {jobId !== null && status !== "failed" && (
            <button className="text-button" onClick={() => refetch()}>
              Retry connection
            </button>
          )}
        </div>
      )}
      {jobId === null && (
        <FileUpload onUpload={handleUpload} uploading={uploading} />
      )}
      {isProcessing && (
        <section className="processing-card" aria-live="polite">
          <span className="processing-icon">
            <LoaderCircle className="spin" size={30} />
          </span>
          <p className="eyebrow">STEP 02 / EXTRACTION</p>
          <h2>
            {status === "processing"
              ? "Reading your document"
              : "Your document is in the queue"}
          </h2>
          <p>
            {status === "processing"
              ? "Reading full-page text and detecting every table."
              : "Waiting for the OCR worker to begin."}
          </p>
          <p className="muted small">
            {resultData?.completed_pages ?? 0} / {resultData?.page_count ?? 1}{" "}
            pages complete · Job #{jobId}. Model startup may take longer.
          </p>
        </section>
      )}
      {status === "failed" && (
        <section className="processing-card">
          <AlertCircle size={30} />
          <h2>We couldn’t extract this document</h2>
          <p>Try an upright, sharper image or a readable PDF.</p>
        </section>
      )}
      {status === "done" && resultData && jobId !== null && (
        <DocumentWorkspace key={jobId} jobId={jobId} initial={resultData} />
      )}
      {jobId !== null && (
        <button className="text-button new-document" onClick={startNewDocument}>
          <ArrowLeft size={15} /> Start a new document
        </button>
      )}
    </>
  );
}
