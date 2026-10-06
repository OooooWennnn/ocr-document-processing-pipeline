import { DocumentRegion, JobResponse, PageData } from "./api";

let measureContext: CanvasRenderingContext2D | null = null;

/** Fit display text to its region; Canvas measurement does not recover the original font. */
export function regionFontSize(
  region: DocumentRegion,
  precise = false,
): number {
  const referenceValue = region.original_value ?? region.value;
  const lines = referenceValue.split("\n");
  const longestLineWidth = Math.max(
    1,
    ...lines.map((line) =>
      [...line].reduce(
        (totalWidth, character) =>
          totalWidth + (/[^\u0000-\u00ff]/.test(character) ? 1 : 0.56),
        0,
      ),
    ),
  );
  const availableWidth = Math.max(
    1,
    region.bbox.x_max - region.bbox.x_min - (region.kind === "cell" ? 6 : 0),
  );
  const regionHeight = Math.max(1, region.bbox.y_max - region.bbox.y_min);

  let fontSize = Math.max(
    3,
    Math.min(
      region.font_size ?? 12,
      availableWidth / longestLineWidth,
      regionHeight / Math.max(1, lines.length) / 1.1,
    ),
  );

  if (precise && typeof window !== "undefined") {
    measureContext ??= window.document.createElement("canvas").getContext("2d");

    if (measureContext) {
      measureContext.font = `${fontSize}px Arial`;

      const measuredWidth = Math.max(
        1,
        ...lines.map((line) => measureContext!.measureText(line).width),
      );

      if (measuredWidth > availableWidth) {
        fontSize = Math.max(3, (fontSize * availableWidth) / measuredWidth);
      }
    }
  }

  for (const token of region.tokens ?? []) {
    const tokenLines = (token.original_value ?? token.value).split("\n");
    const tokenWidth = Math.max(1, token.bbox.x_max - token.bbox.x_min);
    const tokenHeight = Math.max(1, token.bbox.y_max - token.bbox.y_min);
    let textWidth: number;

    if (precise && measureContext) {
      measureContext.font = `${fontSize}px Arial`;
      textWidth = Math.max(
        1,
        ...tokenLines.map((line) => measureContext!.measureText(line).width),
      );
    } else {
      const characterWidth = Math.max(
        1,
        ...tokenLines.map((line) =>
          [...line].reduce(
            (total, character) =>
              total + (/[^\u0000-\u00ff]/.test(character) ? 1 : 0.56),
            0,
          ),
        ),
      );
      textWidth = characterWidth * fontSize;
    }

    fontSize = Math.max(
      3,
      Math.min(
        fontSize,
        (fontSize * tokenWidth) / textWidth,
        tokenHeight / Math.max(1, tokenLines.length) / 1.1,
      ),
    );
  }

  return fontSize;
}

export function regionTextLayout(region: DocumentRegion) {
  const width = Math.max(1, region.bbox.x_max - region.bbox.x_min);
  const height = Math.max(1, region.bbox.y_max - region.bbox.y_min);
  const padding = region.kind === "cell" ? 3 : 0;
  const tokens = region.tokens ?? [];
  let textAlign: "left" | "center" | "right" = "left";

  if (!tokens.length) {
    const referenceValue = region.original_value ?? region.value;
    const lineCount = Math.max(1, referenceValue.split("\n").length);
    const textHeight = regionFontSize(region) * 1.1 * lineCount;
    const top = region.kind === "cell" ? Math.max(2, (height - textHeight) / 2) : 0;

    return { left: padding, right: padding, top, bottom: 0, textAlign };
  }

  const left = Math.max(
    0,
    Math.min(...tokens.map((token) => token.bbox.x_min)) - region.bbox.x_min,
  );
  const right = Math.max(
    0,
    region.bbox.x_max - Math.max(...tokens.map((token) => token.bbox.x_max)),
  );
  const top = Math.max(
    0,
    Math.min(...tokens.map((token) => token.bbox.y_min)) - region.bbox.y_min,
  );
  const tolerance = Math.max(3, width * 0.05);

  if (Math.abs(left - right) <= tolerance && Math.min(left, right) > tolerance) {
    textAlign = "center";
  } else if (left > right + tolerance) {
    textAlign = "right";
  }

  return {
    left: textAlign === "left" ? left : padding,
    right: textAlign === "right" ? right : padding,
    top,
    bottom: 0,
    textAlign,
  };
}

/** Build an export with every page and unsaved edits; do not persist it. */
export function exportDocument(
  documentData: JobResponse,
  drafts: Record<number, Record<string, string>>,
) {
  return {
    format: "positioned-document",
    version: 1,
    pages: (documentData.pages ?? []).map((page) => {
      const pageDrafts = drafts[page.page_index];

      return {
        page_index: page.page_index,
        width: page.image_width,
        height: page.image_height,
        regions: (page.regions ?? []).map((region) => {
          const draftValue = pageDrafts?.[region.id];

          return {
            ...region,
            edited: region.edited || draftValue !== undefined,
            value: draftValue ?? region.value,
          };
        }),
      };
    }),
  };
}

/** Escape OCR text so exported HTML does not interpret it as markup. */
const escapeHtml = (text: string) =>
  text.replace(
    /[&<>"']/g,
    (character) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
      character
      ]!,
  );

/** Position original cell tokens; use replacement text for edited regions. */
function regionContent(region: DocumentRegion, fontSize: number) {
  if (region.edited || !region.tokens?.length) {
    const layout = regionTextLayout(region);
    const style = [
      "position:absolute",
      `left:${layout.left}px`,
      `right:${layout.right}px`,
      `top:${layout.top}px`,
      `bottom:${layout.bottom}px`,
      `text-align:${layout.textAlign}`,
      "white-space:pre-wrap",
      "overflow-wrap:anywhere",
    ].join(";");

    return `<span class="region-value" style="${style}">${escapeHtml(region.value)}</span>`;
  }

  return region.tokens
    .map((token) => {
      const style = [
        "position:absolute",
        "overflow:hidden",
        "white-space:pre",
        `left:${token.bbox.x_min - region.bbox.x_min}px`,
        `top:${token.bbox.y_min - region.bbox.y_min}px`,
        `width:${token.bbox.x_max - token.bbox.x_min}px`,
        `height:${token.bbox.y_max - token.bbox.y_min}px`,
        `font-size:${fontSize}px`,
      ].join(";");

      return `<span style="${style}">${escapeHtml(token.value)}</span>`;
    })
    .join("");
}

function regionHtml(region: DocumentRegion) {
  const fontSize = regionFontSize(region, true);
  const style = [
    `left:${region.bbox.x_min}px`,
    `top:${region.bbox.y_min}px`,
    `width:${Math.max(1, region.bbox.x_max - region.bbox.x_min)}px`,
    `height:${Math.max(1, region.bbox.y_max - region.bbox.y_min)}px`,
    `font-size:${fontSize}px`,
  ].join(";");

  return `<div class="region ${region.kind}" data-region="${escapeHtml(region.id)}" style="${style}">${regionContent(region, fontSize)}</div>`;
}

/** Build printable HTML with current edits and positioned regions; source graphics and fonts are not reproduced. */
export function documentHtml(
  documentData: JobResponse,
  drafts: Record<number, Record<string, string>>,
) {
  const pages = exportDocument(documentData, drafts).pages;
  const pagesHtml = pages
    .map((page) => {
      const pageWidth = page.width || 800;
      const pageHeight = page.height || 1100;
      const regionsHtml = page.regions.map(regionHtml).join("");

      return `<section class="page" aria-label="Page ${page.page_index + 1}" style="width:${pageWidth}px;height:${pageHeight}px">${regionsHtml}</section>`;
    })
    .join("");

  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Extracted document</title><style>
  *{box-sizing:border-box}body{margin:0;background:#e8edf0;font-family:Arial,sans-serif}.page{position:relative;margin:24px auto;background:white;overflow:hidden;box-shadow:0 3px 20px #0001}.region{position:absolute;white-space:pre-wrap;overflow:hidden;line-height:1.1;color:#172b39;margin:0}.cell{border:1px solid #aebbc2;padding:2px 3px;display:flex;align-items:center}.text{padding:0}@media print{body{background:white}.page{margin:0;box-shadow:none;break-after:page}.page:last-child{break-after:auto}}\n</style></head><body>${pagesHtml}</body></html>`;
}

/** Use server dimensions, then source image dimensions or a fallback. */
export function pageDimensions(
  page: PageData,
  sourceImageSize: { width: number; height: number },
) {
  return {
    width: page.image_width || sourceImageSize.width || 800,
    height: page.image_height || sourceImageSize.height || 1100,
  };
}
