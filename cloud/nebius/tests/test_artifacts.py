import json

import boto3
import pytest
from moto import mock_aws

from cloud.nebius.artifacts import LocalStore, S3Store, download, export, sha256
from cloud.nebius.config import encode


@pytest.fixture
def bundle(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "policy.bin").write_bytes(b"training output")
    store = LocalStore(tmp_path / "remote")
    metadata = {"attempt": "example", "config_sha256": "a" * 64}
    export(store, "attempt/", source, metadata, True)
    return store, source, metadata


def test_roundtrip_and_existing_directory_refused(bundle, tmp_path):
    store, _, metadata = bundle
    dest = tmp_path / "download"
    manifest = download(store, "attempt/", dest, metadata)
    assert (dest / "policy.bin").read_bytes() == b"training output"
    assert manifest["success"]
    with pytest.raises(FileExistsError):
        download(store, "attempt/", dest, metadata)


def test_failure_has_no_complete_marker(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "stdout.txt").write_text("training failed")
    store = LocalStore(tmp_path / "remote")
    export(store, "attempt/", source, {}, False)
    assert not (store.root / "attempt/complete.json").exists()
    with pytest.raises(FileNotFoundError):
        download(store, "attempt/", tmp_path / "download", {})


@pytest.mark.parametrize("damage", ["body", "manifest", "marker", "missing", "identity"])
def test_corruption_or_wrong_identity_refused(bundle, tmp_path, damage):
    store, _, metadata = bundle
    if damage == "body":
        store.put("attempt/files/policy.bin", b"corrupt output")
    elif damage == "manifest":
        store.put("attempt/manifest.json", b"{}")
    elif damage == "marker":
        store.put("attempt/complete.json", b"{}")
    elif damage == "missing":
        (store.root / "attempt/files/policy.bin").unlink()
    else:
        metadata["attempt"] = "someone-else"
    with pytest.raises((ValueError, FileNotFoundError)):
        download(store, "attempt/", tmp_path / "download", metadata)


@pytest.mark.parametrize("path", ["../escape", "/tmp/escape", "a/../../escape", "a\\escape", "a//b"])
def test_manifest_cannot_escape_download(bundle, tmp_path, path):
    store, _, metadata = bundle
    manifest = json.loads(store.get("attempt/manifest.json", 10000))
    manifest["files"][0]["path"] = path
    data = encode(manifest)
    store.put("attempt/manifest.json", data)
    store.put("attempt/complete.json", encode({"schema": 1, "manifest_sha256": sha256(data)}))
    with pytest.raises(ValueError):
        download(store, "attempt/", tmp_path / "download", metadata)
    assert not (tmp_path / "escape").exists()


def test_interrupted_upload_never_publishes_success(tmp_path):
    class Broken(LocalStore):
        def put(self, key, data):
            if key.endswith("manifest.json"):
                raise OSError("simulated upload failure")
            super().put(key, data)

    source = tmp_path / "source"
    source.mkdir()
    (source / "out").write_bytes(b"data")
    store = Broken(tmp_path / "remote")
    with pytest.raises(OSError):
        export(store, "attempt/", source, {}, True)
    assert not (store.root / "attempt/complete.json").exists()


def test_s3_roundtrip_with_independent_reader(tmp_path):
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="cleanrl-test")
        writer = S3Store("cleanrl-test", "us-east-1", client)
        reader = S3Store("cleanrl-test", "us-east-1", boto3.client("s3", region_name="us-east-1"))
        source = tmp_path / "source"
        source.mkdir()
        (source / "policy.bin").write_bytes(b"real S3 SDK payload")
        export(writer, "attempt/", source, {"attempt": "test"}, True)
        download(reader, "attempt/", tmp_path / "download", {"attempt": "test"})
        assert (tmp_path / "download/policy.bin").read_bytes() == b"real S3 SDK payload"


def test_export_cannot_replace_an_existing_attempt(bundle):
    store, source, metadata = bundle
    with pytest.raises(ValueError):
        export(store, "attempt/", source, metadata, False)
    assert store.exists("attempt/complete.json")
