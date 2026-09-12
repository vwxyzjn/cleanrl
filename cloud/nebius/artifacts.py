"""Immutable manifests and bounded, independently verified object transfers."""

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

from cloud.nebius.config import encode

MAX_FILE_BYTES = 128 * 1024**2
MAX_TOTAL_BYTES = 512 * 1024**2
MAX_MANIFEST_BYTES = 1024**2


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class S3Store:
    def __init__(self, bucket, region, client=None):
        if client is None:
            import boto3
            from botocore.config import Config

            client = boto3.client(
                "s3",
                endpoint_url=f"https://storage.{region}.nebius.cloud",
                region_name=region,
                config=Config(connect_timeout=10, read_timeout=30, retries={"total_max_attempts": 3}),
            )
        self.client, self.bucket = client, bucket

    def exists(self, key):
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def put(self, key, data):
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)
        if self.get(key, len(data)) != data:
            raise ValueError("Remote object failed read-back verification")

    def get(self, key, limit):
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        body = response["Body"]
        try:
            if response["ContentLength"] > limit:
                raise ValueError("Object exceeds declared size")
            data = body.read(limit + 1)
            if len(data) > limit:
                raise ValueError("Object exceeds size limit")
            return data
        finally:
            body.close()


class LocalStore:
    """Explicit test/export mode; never selected implicitly after an S3 failure."""

    def __init__(self, root):
        self.root = Path(root)

    def exists(self, key):
        return (self.root / key).exists()

    def put(self, key, data):
        p = self.root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def get(self, key, limit):
        with (self.root / key).open("rb") as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError("Object exceeds size limit")
        return data


def export(store, prefix, directory, metadata, success):
    """Called after the child exits and writers close; completion marker is last."""
    if store.exists(prefix + "manifest.json") or store.exists(prefix + "complete.json"):
        raise ValueError("Attempt manifest already exists; never overwrite an exported attempt")
    files, total = [], 0
    for path in sorted(Path(directory).rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlink in output directory")
        if not path.is_file():
            continue
        size = path.stat().st_size
        total += size
        if size > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
            raise ValueError("Output exceeds the bounded example's storage limit")
        data = path.read_bytes()
        relative = path.relative_to(directory).as_posix()
        store.put(prefix + "files/" + relative, data)
        files.append({"path": relative, "bytes": len(data), "sha256": sha256(data)})
    manifest = dict(schema=1, success=bool(success), metadata=metadata, files=files)
    data = encode(manifest)
    if len(data) > MAX_MANIFEST_BYTES:
        raise ValueError("Manifest too large")
    store.put(prefix + "manifest.json", data)
    if success:
        store.put(prefix + "complete.json", encode({"schema": 1, "manifest_sha256": sha256(data)}))
    return manifest


def download(store, prefix, directory, expected):
    """Only a verified success bundle can become a controller success result."""
    marker = json.loads(store.get(prefix + "complete.json", 4096))
    data = store.get(prefix + "manifest.json", MAX_MANIFEST_BYTES)
    if marker.get("schema") != 1 or marker.get("manifest_sha256") != sha256(data):
        raise ValueError("Manifest checksum mismatch")
    manifest = json.loads(data)
    if manifest.get("schema") != 1 or manifest.get("success") is not True:
        raise ValueError("No successful output manifest")
    metadata = manifest.get("metadata", {})
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError("Manifest does not match this attempt")
    entries, seen, total = manifest.get("files"), set(), 0
    if not isinstance(entries, list) or not 1 <= len(entries) <= 1000:
        raise ValueError("Invalid manifest files")
    for item in entries:
        name = item.get("path", "")
        pure = PurePosixPath(name)
        if (
            not isinstance(name, str)
            or not name
            or pure.is_absolute()
            or ".." in pure.parts
            or str(pure) != name
            or "\\" in name
            or name in seen
        ):
            raise ValueError("Unsafe or duplicate artifact path")
        seen.add(name)
        size = item.get("bytes")
        if type(size) is not int or not 0 <= size <= MAX_FILE_BYTES:
            raise ValueError("Invalid artifact size")
        total += size
        if total > MAX_TOTAL_BYTES or not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")):
            raise ValueError("Invalid artifact checksum or total size")
    target = Path(directory)
    # A new directory also prevents following pre-existing symlinks during download.
    target.mkdir(parents=True, exist_ok=False)
    for item in entries:
        data = store.get(prefix + "files/" + item["path"], item["bytes"])
        if len(data) != item["bytes"] or sha256(data) != item["sha256"]:
            raise ValueError("Artifact checksum mismatch; download directory remains incomplete")
        dest = target / item["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
