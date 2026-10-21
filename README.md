# OCR Document Processing Pipeline

An OCR app for reviewing PNG, JPEG and PDF documents in the browser. Text and
table cells stay in their original page positions, so you can compare the result
with the source and correct it without switching between separate forms.

The project runs locally with Docker Compose. Next.js provides the editor,
FastAPI handles uploads and edits, and a Celery worker runs PaddleOCR on the CPU.

## Preview

<p align="center">
  <img src="docs/images/main_page.png" alt="Main page" width="800"/>
  <br />
  <em>Main page — Select and upload an image or PDF.</em>
</p>
<br />

<p align="center">
  <img src="docs/images/result.png" alt="OCR result" width="800"/>
  <br />
  <em>Result Page - Displays OCR result and confidence scores.</em>
</p>
<br />

<p align="center">
  <img src="docs/images/edit.png" alt="Edit result" width="800"/>
  <br />
  <em>Edit Result - Select inaccurate or low confidence field and edit.</em>
</p>
<br />

<details>
  <summary>More Screenshots</summary>

  <p align="center">
    <img src="docs/images/uploaded.png" alt="Image uploaded" width="800"/>
    <br />
    <em>Image Uploaded - Shows the uploaded original image.</em>
  </p>
  <br />

  <p align="center">
    <img src="docs/images/processing.png" alt="OCR processing" width="800"/>
    <br />
    <em>Processing Page - User receives Job # and workers process OCR.</em>
  </p>
  <br />

  <p align="center">
    <img src="docs/images/export_html.png" alt="HTML export" width="800"/>
    <br />
    <em>HTML Export - Exported HTML.</em>
  </p>
  <br />

  <p align="center">
    <img src="docs/images/export_json.png" alt="JSON export" width="800"/>
    <br />
    <em>JSON Export - Exported JSON.</em>
  </p>
  <br />
</details>

## Features

- Extract page text and multiple tables, including merged cells and empty cells
  in tables with visible rules.
- Process PDFs one page at a time and show progress as each page finishes.
- Select a text region or table cell, inspect its source crop, and edit its value.
- Show OCR confidence, spatial mapping scores and extraction warnings for review.
- Save changes across pages while keeping the original OCR result.
- Export all pages as JSON or positioned HTML, including current unsaved edits.

PNG and JPEG uploads are limited to 10 MB and 25 million pixels. PDFs can contain
1–30 pages and must be 50 MB or smaller. Password-protected PDFs are not supported.

## Run locally

Start Docker Desktop. In a new checkout, copy the configuration template from
the project root:

```sh
cp .env.example .env
```

Replace `change-me` in `.env` with your local database password. The template's
`DATABASE_URL` uses that password. For Supabase, replace the entire URL with your
project's connection string instead. If `.env` already exists, edit it instead of
copying the template over it. Then run:

```sh
docker compose up -d --build
```

Open [the document editor](http://localhost:3000). The API documentation is at
[localhost:8000/docs](http://localhost:8000/docs).

You do not need a local Python virtual environment or Node.js installation.
The first build installs dependencies, and the first worker start downloads the
OCR models. Model files are cached in Docker volumes for later starts.

Check model initialization with:

```sh
docker compose logs -f worker
```

With the default warmup setting, look for `Document OCR model ready` before
uploading a document. A healthy worker can answer Celery pings before the model
has finished loading.

OCR needs several gigabytes of memory. The worker is capped at 7 GB by default;
Docker Desktop also needs room for PostgreSQL, Redis and the application servers.
If the worker is killed during extraction, check the logs and Docker's memory
allocation before increasing OCR resolution or batch size.

Upload a document and wait for processing to finish. Click a region to edit it,
then use **Save document** to persist your changes. Draft edits survive page
navigation, but refreshing the browser loses unsaved changes and the current
job selection.

## Configuration

Compose requires `DATABASE_URL` and `POSTGRES_PASSWORD`. The API and worker both
use `DATABASE_URL`; `POSTGRES_PASSWORD` configures the local `db` service. Keep
real credentials in the root `.env`, which Git and image builds exclude.
Do not create separate `backend/.env` or `frontend/.env` files.

The template contains only database and optional S3 settings. OCR limits, worker
settings and local ports already have defaults in `docker-compose.yml`. Add an
override to `.env` only when you need to change one. Compose supplies
`NEXT_PUBLIC_API_URL` when building the frontend.

For an existing database volume, use its current password. Changing `.env` alone
does not change the database user's password.

| Setting                          | Default                 | Purpose                                                                   |
| -------------------------------- | ----------------------- | ------------------------------------------------------------------------- |
| `DATABASE_URL`                   | Required                | PostgreSQL connection string shared by the API and worker                 |
| `POSTGRES_PASSWORD`              | Required                | Local database password; replace the template placeholder                 |
| `FRONTEND_PORT` / `BACKEND_PORT` | `3000` / `8000`         | Browser-facing ports                                                      |
| `POSTGRES_PORT` / `REDIS_PORT`   | `5432` / `63797`        | Local database and queue ports                                            |
| `NEXT_PUBLIC_API_URL`            | `http://localhost:8000` | API address used by the browser                                           |
| `OCR_LANGUAGE`                   | `korean`                | Recognition language; use `en` for English documents                      |
| `OCR_DETECTION_MODEL`            | `PP-OCRv5_mobile_det`   | Text detector for Korean and English; `PP-OCRv5_server_det` uses the previous detector |
| `OCR_WARMUP`                     | `1`                     | Load the model after worker startup; `0` loads it on the first job        |
| `OCR_WORKER_MAX_TASKS_PER_CHILD`  | `5`                     | Replace the worker child after this many tasks, including warmup; minimum 1 |
| `OCR_CPU_THREADS`                | `4`                     | CPU threads used by the model                                             |
| `OCR_RECOGNITION_BATCH_SIZE`     | `4`                     | Text crops recognized together                                            |
| `OCR_DETECTION_SIZE`             | `2048`                  | Maximum long side for text detection                                      |
| `OCR_DETECTION_MAX_PIXELS`       | `1000000`               | Pixel budget for the text detector                                        |
| `OCR_UPSCALE_TARGET`             | `1600`                  | Target long side for small inputs, with at most 2× upscaling              |
| `OCR_REFINE_MAX_CROPS`           | `64`                    | Maximum crops considered for targeted re-recognition                      |
| `OCR_WORKER_MEMORY_LIMIT`        | `7g`                    | Worker container memory limit                                             |
| `OCR_TASK_TIME_LIMIT`            | `5400`                  | Soft document timeout in seconds; the hard limit follows 60 seconds later |

After changing `.env`, run `docker compose up -d --build` again. The browser API
address is baked into the frontend build, so a restart alone will not update it.
If you change the backend port, update `NEXT_PUBLIC_API_URL` to match.

Higher detection limits and larger batches use more memory. The detector budget
does not resize the recognition source crops, but a lower budget can make very
small text harder to locate.

Korean and English use the mobile text detector while keeping their existing
language recognizer. Other languages retain PaddleOCR's language defaults.
The mobile detector uses tighter text boxes to avoid spilling into nearby cells.
See [performance notes](docs/performance.md) for measurements, failed experiments
and commands for comparing the detectors on the same documents.

## How processing works

1. The API validates the file, saves it to `storage/uploads` or the configured S3
   bucket, and creates a `Job` row in PostgreSQL.
2. It sends the job ID to Redis and returns immediately. The browser polls for
   progress; the upload response does not mean OCR has finished.
3. The Celery task passes the job ID to `document_job.process_job`, which reads
   the file and processes its pages in order. It reuses one OCR model in the
   worker's child process.
4. Each page is converted into text regions and table cells with coordinates.
   Clear table rules can repair cell geometry, and uncertain text matches are
   left available for review rather than forced into a cell.
5. The worker saves each completed page. When all pages finish, it marks the job
   as `done`. Edits are saved separately from the extracted result.

PDF pages with embedded text and no image objects use the stored text and
coordinates instead of full-page text OCR. Layout and table detection still run.
Pages containing images use OCR, then merge embedded text with OCR-only regions.

The worker uses one prefork child and replaces it after five Celery tasks to
release retained memory. Warmup counts as one task. A replacement child reloads
the model when needed. `OCR_WORKER_MAX_TASKS_PER_CHILD` controls this interval;
keep the default until repeated jobs show stable memory use. If a child is killed
or hits its hard timeout, the parent tries to record the job as failed.

## Project layout

```text
backend/
  main.py                   FastAPI app and database initialization
  celery_app.py             Celery configuration, tasks, warmup and interruption handling
  Dockerfile.api            API container image
  Dockerfile.worker         Celery worker container image
  requirements.txt          API dependencies and shared version constraints
  requirements.ocr.txt      Additional OCR dependencies
  benchmark.py              Optional OCR performance measurements
  src/api.py                Upload, result, preview and edit endpoints
  src/database.py           Database connection and request sessions
  src/models.py             PostgreSQL table definitions
  src/schemas.py            Internal OCR data types
  src/services/document_job.py  Model reuse, document processing and result storage
  src/services/             Document input, OCR, geometry, mapping and edits
  tests/                    API, document, PDF, table and worker checks
frontend/
  app/                      Pages and styles
  components/               Upload, document editor and query provider
  hooks/                    Upload state and result polling
  lib/                      API types, requests and exports
docs/
  aws-deployment.md          S3 and EC2 deployment instructions
  performance.md            Performance changes and measurements
  benchmarks/               Shareable measurement summaries
  images/                   README screenshots
storage/                    Local uploads, previews and raw benchmarks (not committed)
docker-compose.yml          Services, storage and optional separate worker host
.env.example                Compose configuration template
```

`requirements.txt` defines the API dependencies. The worker adds
`requirements.ocr.txt` and CPU PaddlePaddle. The current OCR versions are
PaddleOCR 3.4.0, PaddleX 3.4.1 and PaddlePaddle 3.3.0. Small compatibility adapters
in `services/` depend on these versions, so check them when upgrading.

## Stored data

PostgreSQL stores job metadata and JSONB results. Uploaded files and PDF page
previews live in `./storage`, shared by the API and worker.

Set `OCR_S3_BUCKET` and `AWS_REGION` to store new uploads and PDF previews in a
private S3 bucket instead. The API keeps its existing image endpoint, and the
worker downloads one document to temporary storage for processing. Database
schema and result JSON stay unchanged; `file_path` stores an S3 URI for new S3
uploads. Leaving the bucket empty preserves local storage. Existing files are
not moved automatically.

See [AWS deployment](docs/aws-deployment.md) for IAM role permissions, separate
web/worker EC2 startup commands, network settings and CPU benchmarking limits.
To try S3 with the local Docker stack first, follow its
[local S3 setup](docs/aws-deployment.md#connect-s3-from-the-local-docker-stack).
That setup uses the optional `docker-compose.s3-local.yml` to read existing host
AWS credentials; EC2 uses IAM roles without this override.

| Field         | Contents                                     |
| ------------- | -------------------------------------------- |
| `raw_json`    | Model output and preprocessing details       |
| `result_json` | Original extracted document, grouped by page |
| `edited_json` | Saved user edits applied to the document     |

Result polling reads the latest row's original and edited documents without
loading `raw_json`. The raw model output stays in the database for inspection.

Database, Redis and model caches use named Docker volumes. `docker compose down`
stops the stack without removing them. Adding `-v` deletes those volumes, including
the database and model cache; it does not remove the bind-mounted `./storage`.

For older database rows with absolute storage paths, `OCR_STORAGE_LEGACY_ROOT`
maps the old storage root to the container path when reading files. It does not
rewrite rows or move uploads.

## Development commands

```sh
docker compose ps
docker compose logs -f backend worker
docker compose up -d --build
docker compose restart worker
docker compose down
```

Run the backend tests inside Docker:

```sh
docker compose run --rm --no-deps worker python -m unittest discover -s tests -v
```

The tests cover uploads, PDF handling, table geometry, text mapping, edit
validation, interrupted workers and S3 file handling. API tests use an isolated
SQLite database and a mocked queue; S3 tests use an in-memory fake without AWS
credentials. These checks do not measure OCR accuracy across real documents or
validate a real AWS deployment.

Frontend lint, TypeScript checks and the production build run with:

```sh
docker compose build frontend
```

## Current limitations

- Extraction still needs human review. Small text, faint rules, rotated scans and
  irregular layouts can produce missing or incorrect regions.
- Field mapping uses geometry. It does not identify business fields such as
  invoice number or shipper by their meaning. A mapping score measures overlap,
  not the probability that a value is correct.
- You can edit text values, but cannot manually redraw tables, add rows or change
  merged cells in the editor.
- A failed page leaves earlier completed pages in the database, but the browser
  shows the failed job rather than opening a partial-result editor.
- Child-process failure handling needs a live parent and database. A whole
  container or host failure can leave a job marked as `processing`.
- There is no job history screen, automatic resume, retry endpoint or edit
  version check for multiple users.
- Compose binds ports to localhost by default. S3 and separate EC2 hosts can be
  configured, but authentication and HTTPS still need to be added before public
  use. The AWS setup has not been tested against live resources.
