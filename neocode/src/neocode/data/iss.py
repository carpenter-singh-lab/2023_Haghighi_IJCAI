"""In Situ Sequencing images data module."""
from pathlib import Path

import boto3
import botocore
from rich import print
from tqdm import tqdm

from neocode.utils import NeocodeError, parallel

S3_BUCKET_NAME = "cellpainting-gallery"
S3_ISS_IMAGES_PREFIX = "cpg0021-periscope/broad/images/20210124_6W_CP228/images_aligned_cropped"  # noqa: E501
S3_ISS_ANALYSIS_PREFIX = "cpg0021-periscope/broad/workspace/analysis/20210124_6W_CP228"


def get_file_list(bucket: str, prefix: str | None = None) -> list[dict]:
    """Recusiverly fetch details of all files in a bucket.

    Parameters
    ----------
    bucket : str
        Bucker
    prefix : str | None
        Filter files using a prefix. by default None.

    Returns
    -------
    list[dict]
        List of details of all files in a bucket.
    """
    s3 = boto3.client("s3")
    is_truncated = True
    continuation_token = None
    file_details_list = []
    while is_truncated:
        try:
            if continuation_token is None:
                response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            else:
                response = s3.list_objects_v2(
                    Bucket=bucket, Prefix=prefix, ContinuationToken=continuation_token
                )
        except Exception as e:
            raise NeocodeError(e)
        file_details_list.extend(response["Contents"])
        is_truncated = response["IsTruncated"]
        continuation_token = response.get("NextContinuationToken")
    return file_details_list


def download_from_s3(
    file_key_list: list[str], job_idx: int, bucket: str, prefix: str, write_path: Path
) -> Path:
    """Download file from s3 bucket.

    Parameters
    ----------
    file_key_list : list[str]
        List of file name to download.
    job_idx: int
        Index of worker process.
    bucket : str
        Bucket identifier.
    prefix : str
        Filter prefix for custructing file paths.
    write_path : Path
        Path to save downloaded file.

    Returns
    -------
    Path
        Path of downloaded file.

    Raises
    ------
    NeocodeError
        Failed to download file from s3.
    """
    for file_key in (pbar := tqdm(file_key_list, position=job_idx)):
        pbar.set_description(f"Downloading {file_key}")
        s3 = boto3.resource("s3")
        current_write_path = write_path.joinpath(
            file_key.replace(prefix, "").lstrip("/")
        )
        current_write_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = current_write_path.with_suffix(current_write_path.suffix + ".part")
        try:
            s3.Bucket(bucket).download_file(file_key, str(temp_path.absolute()))
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                raise NeocodeError(f"File {file_key} is missing in bucket: {bucket}.")
            else:
                raise NeocodeError(e)
        temp_path.rename(current_write_path)


def download_iss(write_path: Path, force: bool = False, debug: bool = False) -> None:
    """Download ISS images dataset.

    Parameters
    ----------
    write_path : Path
        Path to save dataset
    force : bool, optional
        Force redownload dataset, by default False
    debug : bool, optional
        Run in debug mode, by default False
    """
    write_path = write_path.joinpath("iss")
    write_path.mkdir(parents=True, exist_ok=True)

    # Get details of all files from s3
    for prefix, write_path in [
        (S3_ISS_IMAGES_PREFIX, write_path.joinpath("images")),
        (S3_ISS_ANALYSIS_PREFIX, write_path.joinpath("analysis")),
    ]:
        file_list = get_file_list(S3_BUCKET_NAME, prefix)
        # Check for already downloaded files.
        if not force:
            bool_list = [
                write_path.joinpath(
                    file["Key"].replace(prefix, "").lstrip("/")
                ).is_file()
                for file in file_list
            ]
            file_list = [
                file for file, status in zip(file_list, bool_list) if not status
            ]
        # Download files in parallel
        remaining_files = len(file_list)
        file_size = 0
        for file in file_list:
            file_size = file_size + file["Size"]
        if remaining_files > 0:
            print(
                f"Downloading files for prefix: {prefix}\n",
                f"Remaining download size: {round(file_size/(1024 * 1024))} MB\n",
                f"Remaining files: {remaining_files}\n",
            )
            parallel(
                [file["Key"] for file in file_list],
                download_from_s3,
                [S3_BUCKET_NAME, prefix, write_path],
                3,
            )
        else:
            print(f"\nAll files are already downloaded for prefix: {prefix}")
        print("\n---------------------------------------------------------------\n")
