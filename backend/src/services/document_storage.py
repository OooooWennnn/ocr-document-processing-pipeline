"""로컬 파일과 S3 원본·PDF 미리보기의 저장 및 읽기를 담당한다."""

import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

import boto3

from src.services.document_input import preview_path, resolve_source_path


@lru_cache(maxsize=1)
def get_s3_client():
    """프로세스마다 S3 연결을 재사용한다. EC2에서는 IAM 역할로 인증한다."""
    return boto3.client("s3", region_name=os.getenv("AWS_REGION") or None)


def is_s3(source):
    """DB의 파일 경로가 S3 URI인지 확인한다."""
    return str(source).startswith("s3://")


def split_s3_uri(source):
    """S3 URI를 버킷 이름과 객체 키로 나누며 잘못된 경로는 거부한다."""
    parsed = urlsplit(str(source))
    key = parsed.path.lstrip("/")

    if parsed.scheme != "s3" or not parsed.netloc or not key or parsed.query or parsed.fragment:
        raise ValueError("Invalid S3 document path.")

    return parsed.netloc, key


def save_upload(path, content_type):
    """검증된 업로드를 S3에 저장하고 URI를 반환한다. 버킷이 없으면 로컬 경로를 유지한다."""
    bucket = os.getenv("OCR_S3_BUCKET", "").strip()

    if not bucket:
        return str(path)

    key = f"uploads/{Path(path).name}"
    get_s3_client().upload_file(str(path), bucket, key, ExtraArgs={"ContentType": content_type})

    return f"s3://{bucket}/{key}"


def delete_upload(source):
    """DB 저장에 실패한 새 업로드만 정리할 때 호출한다."""
    if is_s3(source):
        bucket, key = split_s3_uri(source)
        get_s3_client().delete_object(Bucket=bucket, Key=key)
    else:
        Path(source).unlink(missing_ok=True)


@contextmanager
def local_source(source):
    """워커에 읽을 수 있는 로컬 경로를 제공하고, S3 다운로드 임시 파일은 종료 시 삭제한다."""
    if not is_s3(source):
        yield resolve_source_path(source)
        return

    bucket, key = split_s3_uri(source)

    with TemporaryDirectory(prefix="ocr-document-") as directory:
        path = Path(directory) / PurePosixPath(key).name
        get_s3_client().download_file(bucket, key, str(path))

        yield path


def preview_location(source, page_index):
    """원본과 같은 저장소에 있는 PDF 페이지 미리보기 경로를 반환한다."""
    if not is_s3(source):
        return str(preview_path(resolve_source_path(source), page_index))

    bucket, key = split_s3_uri(source)
    prefix = PurePosixPath(key).with_suffix("")

    return f"s3://{bucket}/{prefix}/page-{page_index}.png"


def save_preview(source, path, page_index):
    """워커가 만든 PNG 미리보기를 원본의 S3 버킷에 저장한다. 로컬 미리보기는 그대로 둔다."""
    if is_s3(source):
        bucket, key = split_s3_uri(preview_location(source, page_index))
        get_s3_client().upload_file(str(path), bucket, key, ExtraArgs={"ContentType": "image/png"})


def open_s3_file(source):
    """API가 비공개 S3 파일을 스트리밍할 수 있도록 응답 본문과 메타데이터를 반환한다."""
    bucket, key = split_s3_uri(source)

    return get_s3_client().get_object(Bucket=bucket, Key=key)


def stream_s3_body(body):
    """파일 전체를 RAM에 올리지 않고 전달하며 응답 스트림 사용 후 연결을 닫는다."""
    try:
        yield from body.iter_chunks(chunk_size=64 * 1024)
    finally:
        body.close()
