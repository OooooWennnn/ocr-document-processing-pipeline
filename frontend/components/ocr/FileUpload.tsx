"use client";
/* eslint-disable @next/next/no-img-element -- Local blob preview requires the original image. */
import { type DragEvent, useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  CheckCircle2,
  FileImage,
  ImagePlus,
  ShieldCheck,
  UploadCloud,
  X,
} from "lucide-react";
/** Select, preview and submit a file; the server validates it again. */
export default function FileUpload({
  onUpload,
  uploading,
}: {
  onUpload: (selectedFile: File) => Promise<void>;
  uploading: boolean;
}) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Release old preview URLs when the file changes or the component unmounts.
  useEffect(
    () => () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    },
    [previewUrl],
  );

  /** Check file type and size, then create a preview URL for images. */
  const selectFile = (candidateFile?: File) => {
    if (!candidateFile || uploading) {
      return;
    }

    const isPdf = candidateFile.type === "application/pdf";
    const maxFileSize = (isPdf ? 50 : 10) * 1024 * 1024;
    const isSupportedType = ["image/png", "image/jpeg", "application/pdf"].includes(
      candidateFile.type,
    );

    if (!candidateFile.size || candidateFile.size > maxFileSize || !isSupportedType) {
      setErrorMessage("Choose PNG/JPEG (10 MB) or PDF (50 MB, up to 30 pages).");
      return;
    }

    setSelectedFile(candidateFile);
    setPreviewUrl(isPdf ? "" : URL.createObjectURL(candidateFile));
    setErrorMessage("");
  };

  const removeFile = () => {
    setSelectedFile(null);
    setPreviewUrl("");

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();

    if (!uploading) {
      setIsDragging(true);
    }
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    selectFile(event.dataTransfer.files[0]);
  };

  return (
    <div className="upload-layout">
      <section className="panel upload-panel">
        <div className="panel-heading">
          <span className="eyebrow">01 / SOURCE DOCUMENT</span>
          <span className="pill">PNG / JPG / PDF</span>
        </div>
        <input
          ref={fileInputRef}
          aria-label="Select a document"
          type="file"
          accept="image/png,image/jpeg,application/pdf"
          className="sr-only"
          disabled={uploading}
          onChange={(event) => selectFile(event.target.files?.[0])}
        />
        <div
          className={`drop-zone ${isDragging ? "dragging" : ""} ${selectedFile ? "has-file" : ""}`}
          onDragOver={handleDragOver}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
        >
          {selectedFile ? (
            <>
              <div className="upload-preview">
                {previewUrl ? (
                  <img src={previewUrl} alt="Selected document preview" />
                ) : (
                  <span className="pdf-preview">
                    PDF · Multiple pages supported
                  </span>
                )}
              </div>
              <div className="file-summary">
                <FileImage size={20} />
                <div>
                  <strong>{selectedFile.name}</strong>
                  <span>
                    {(selectedFile.size / 1024 / 1024).toFixed(2)} MB · Ready to extract
                  </span>
                </div>
                <button
                  className="icon-button"
                  aria-label="Remove selected image"
                  disabled={uploading}
                  onClick={removeFile}
                >
                  <X size={16} />
                </button>
              </div>
            </>
          ) : (
            <>
              <span className="upload-icon">
                <UploadCloud size={32} strokeWidth={1.5} />
              </span>
              <h2>Drop your document here</h2>
              <p>or choose a document from your computer</p>
              <button
                className="button secondary"
                disabled={uploading}
                onClick={() => fileInputRef.current?.click()}
              >
                <ImagePlus size={16} /> Browse files
              </button>
              <span className="muted small">
                Images up to 10 MB · PDF up to 50 MB / 30 pages
              </span>
            </>
          )}
        </div>
        {errorMessage && (
          <p role="alert" className="field-error">
            {errorMessage}
          </p>
        )}
        <div className="upload-bottom">
          <span className="muted small">
            <ShieldCheck size={15} /> Stored in your local workspace
          </span>
          <button
            className="button primary"
            disabled={!selectedFile || uploading}
            onClick={() => selectedFile && onUpload(selectedFile)}
          >
            {uploading ? "Uploading…" : "Extract document"}
            <ArrowRight size={16} />
          </button>
        </div>
      </section>
      <aside className="upload-guide">
        <p className="eyebrow">BETTER INPUT, BETTER RESULTS</p>
        <h2>
          A few details
          <br />
          make a difference.
        </h2>
        <ul>
          {[
            [
              "Keep it upright",
              "Use a straight, well-lit image with visible cell boundaries.",
            ],
            [
              "Include the whole document",
              "Titles, addresses, notes and multiple tables are kept on each page.",
            ],
            [
              "Make small text readable",
              "Use the original image instead of a compressed screenshot.",
            ],
          ].map(([title, detail]) => (
            <li key={title}>
              <CheckCircle2 size={18} />
              <div>
                <strong>{title}</strong>
                <p>{detail}</p>
              </div>
            </li>
          ))}
        </ul>
        <div className="guide-note">
          You’ll be able to compare the source and edit text regions and table
          cells before saving.
        </div>
      </aside>
    </div>
  );
}
