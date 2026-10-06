/** Pixel bounds measured from the page's top-left corner. */
export interface Box {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
}

/** Cell value, bounds and spans. Mapping scores measure overlap, not correctness. */
export interface Cell {
  row: number;
  col: number;
  rowspan: number;
  colspan: number;
  value: string;
  original_value?: string;
  edited?: boolean;
  score: number | null;
  mapping_score?: number | null;
  mapping_warning?: boolean;
  bbox: Box;
}

/** OCR or PDF text; unknown and native PDF scores are null. */
export interface TextItem {
  id?: string;
  original_value?: string;
  edited?: boolean;
  source?: string;
  value: string;
  score: number | null;
  bbox: Box;
}

/** Table grid lines and cells; line spacing determines row and column sizes. */
export interface TableData {
  id: string;
  x_lines: number[];
  y_lines: number[];
  cells: Cell[];
  image_width?: number;
  image_height?: number;
  warnings?: string[];
  unmapped_texts?: TextItem[];
}

/** One editable text or cell region, with tokens preserving original word positions. */
export interface DocumentRegion {
  id: string;
  kind: "text" | "cell";
  value: string;
  original_value?: string;
  edited?: boolean;
  score: number | null;
  bbox: Box;
  font_size?: number;
  table_id?: string;
  row?: number;
  col?: number;
  rowspan?: number;
  colspan?: number;
  mapping_warning?: boolean;
  mapping_score?: number | null;
  tokens?: (TextItem & { font_size?: number })[];
  recognition_refined?: boolean;
  source?: string;
}

/** Text, tables and regions for a zero-based page index. */
export interface PageData {
  regions?: DocumentRegion[];
  page_index: number;
  image_width?: number;
  image_height?: number;
  texts: TextItem[];
  tables: TableData[];
  warnings?: string[];
  elapsed_seconds?: number;
  text_source?: string;
}

/** Job state and pages; top-level tables support older first-page clients. */
export interface JobResponse {
  status: string;
  message?: string;
  tables?: TableData[];
  pages?: PageData[];
  page_count?: number;
  completed_pages?: number;
  elapsed_seconds?: number;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/** Send job requests with a 30-second HTTP timeout, separate from the OCR task timeout. */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE}/api/v1/jobs${path}`, {
      ...init,
      signal: AbortSignal.timeout(30000),
    });
  } catch {
    throw new Error(
      "Cannot reach the server. Check your connection and try again.",
    );
  }

  const responseBody = await response.json().catch(() => null);

  if (!response.ok) {
    const errorMessage =
      typeof responseBody?.detail === "string"
        ? responseBody.detail
        : "The request could not be completed.";

    throw new Error(errorMessage);
  }

  if (!responseBody) {
    throw new Error("The server returned an invalid response.");
  }

  return responseBody;
}

/** Build the source image or PDF preview URL. */
export const sourceImageUrl = (jobId: number, pageIndex = 0) =>
  `${API_BASE}/api/v1/jobs/${jobId}/image?page_index=${pageIndex}`;

/** Submit PNG, JPEG or PDF and return a job ID without waiting for OCR. */
export function uploadImageToOCR(body: FormData) {
  return request<{ job_id: number; status: string }>("/", {
    method: "POST",
    body,
  });
}

/** Fetch job progress and results, preferring saved edits. */
export function getResult(jobId: number) {
  return request<JobResponse>(`/${jobId}/result`);
}

/** Save region IDs and text across pages; the server keeps coordinates and scores. */
export function saveDocument(
  jobId: number,
  pages: { page_index: number; regions: { id: string; value: string }[] }[],
) {
  return request<JobResponse>(`/${jobId}/result`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pages }),
  });
}
