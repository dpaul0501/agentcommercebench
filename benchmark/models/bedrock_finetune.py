"""
Bedrock Fine-Tuning Job Submitter

Trains an AgentCommerce fraud classifier on Amazon Nova via Bedrock Model Customization.

Workflow:
  1. generate_finetune_data.py  →  finetune_data.jsonl + finetune_data_val.jsonl
  2. This script uploads to S3 and submits the fine-tuning job
  3. Poll for completion, download custom model weights reference
  4. The custom model ID becomes our novel baseline in holistic.py

Estimated cost (Nova Micro, 1000 train records, 3 epochs):
  ~$3–$10 USD (Bedrock model customization pricing)

Usage:
    python -m benchmark.models.bedrock_finetune --train benchmark/models/finetune_data.jsonl
    python -m benchmark.models.bedrock_finetune --poll <job-id>
"""
import sys, os, json, time, argparse, hashlib
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import boto3
import dotenv

dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"), override=False)

S3_BUCKET       = "v1-anon"
S3_PREFIX       = "agentcommercebench/finetune"
BASE_MODEL_ID   = "amazon.nova-micro-v1:0:128k"  # cheapest fine-tunable Nova model (128k context variant)
OUTPUT_PREFIX   = "agentcommercebench-guard"

_ROLE_ARN = os.environ.get(
    "BEDROCK_FINETUNE_ROLE_ARN",
    "arn:aws:iam::170554564926:role/BedrockFineTuneRole",
)


def _s3():
    return boto3.client(
        "s3",
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )


def _bedrock():
    return boto3.client(
        "bedrock",
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )


def upload_training_data(train_path: str, val_path: str) -> tuple[str, str]:
    """Upload JSONL files to S3, return (train_uri, val_uri)."""
    s3 = _s3()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    def upload(local_path: str, name: str) -> str:
        key = f"{S3_PREFIX}/{ts}/{name}"
        print(f"  Uploading {local_path} → s3://{S3_BUCKET}/{key}")
        s3.upload_file(local_path, S3_BUCKET, key)
        return f"s3://{S3_BUCKET}/{key}"

    train_uri = upload(train_path, "train.jsonl")
    val_uri   = upload(val_path,   "validation.jsonl")
    return train_uri, val_uri


def submit_job(train_uri: str, val_uri: str, epochs: int = 3) -> str:
    """Submit a Bedrock model customization (fine-tuning) job."""
    bedrock = _bedrock()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    job_name   = f"{OUTPUT_PREFIX}-{ts}"
    model_name = f"{OUTPUT_PREFIX}-v1"

    print(f"\nSubmitting fine-tuning job:")
    print(f"  Base model : {BASE_MODEL_ID}")
    print(f"  Job name   : {job_name}")
    print(f"  Epochs     : {epochs}")
    print(f"  Train data : {train_uri}")
    print(f"  Val data   : {val_uri}")
    print(f"  IAM role   : {_ROLE_ARN}")

    output_s3 = f"s3://{S3_BUCKET}/{S3_PREFIX}/output/{ts}/"

    resp = bedrock.create_model_customization_job(
        jobName=job_name,
        customModelName=model_name,
        roleArn=_ROLE_ARN,
        baseModelIdentifier=BASE_MODEL_ID,
        customizationType="FINE_TUNING",
        trainingDataConfig={"s3Uri": train_uri},
        validationDataConfig={"validators": [{"s3Uri": val_uri}]},
        outputDataConfig={"s3Uri": output_s3},
        hyperParameters={
            "epochCount":         str(epochs),
            "batchSize":          "4",
            "learningRate":       "0.00001",
            "learningRateWarmupSteps": "10",
        },
    )

    job_arn = resp["jobArn"]
    print(f"\n  Job ARN: {job_arn}")
    print(f"  Monitor: aws bedrock get-model-customization-job --job-identifier '{job_arn}'")
    print(f"  Or run:  python -m benchmark.models.bedrock_finetune --poll '{job_arn}'")

    # Save state for later polling
    state = {
        "job_arn": job_arn, "job_name": job_name, "model_name": model_name,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "train_uri": train_uri, "val_uri": val_uri,
        "base_model": BASE_MODEL_ID,
    }
    state_path = "benchmark/models/finetune_job.json"
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
    print(f"\n  Job state saved → {state_path}")
    return job_arn


def poll_job(job_arn: str) -> dict:
    """Poll a fine-tuning job until completion. Returns job status dict."""
    bedrock = _bedrock()
    print(f"\nPolling job: {job_arn}")

    for attempt in range(180):  # max 3 hours
        resp = bedrock.get_model_customization_job(jobIdentifier=job_arn)
        status   = resp.get("status", "")
        job_name = resp.get("jobName", "")
        model_id = resp.get("outputModelArn", "")

        print(f"  [{attempt:3d}] {status:<20}  {job_name}")

        if status == "Completed":
            print(f"\n  Fine-tuning complete!")
            print(f"  Custom model ARN: {model_id}")
            # Save the model ID for later use
            state_path = "benchmark/models/finetune_job.json"
            try:
                with open(state_path) as f:
                    state = json.load(f)
            except Exception:
                state = {}
            state["status"] = "Completed"
            state["custom_model_arn"] = model_id
            state["completed_at"] = datetime.now(timezone.utc).isoformat()
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
            print(f"\n  Next: add this model to benchmark/baselines/llm_safety.py")
            print(f"  Model ID to use in invoke_model: {model_id}")
            return {"status": "Completed", "model_arn": model_id}

        if status in ("Failed", "Stopped"):
            failure = resp.get("failureMessage", "no details")
            print(f"\n  Job {status}: {failure}")
            return {"status": status, "error": failure}

        time.sleep(60)

    print("  Timeout after 3 hours. Check AWS console.")
    return {"status": "timeout"}


def submit_from_local(
    train_path: str = "benchmark/models/finetune_data.jsonl",
    epochs: int = 3,
) -> str:
    val_path = train_path.replace(".jsonl", "_val.jsonl")
    if not os.path.exists(train_path):
        print(f"ERROR: Train data not found at {train_path}")
        print("  Run first: python -m benchmark.models.generate_finetune_data")
        return ""
    if not os.path.exists(val_path):
        print(f"ERROR: Val data not found at {val_path}")
        return ""

    train_uri, val_uri = upload_training_data(train_path, val_path)
    job_arn = submit_job(train_uri, val_uri, epochs=epochs)
    return job_arn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",  default="benchmark/models/finetune_data.jsonl",
                        help="Local path to training JSONL")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--poll",   default=None,
                        help="Job ARN to poll (skips data upload)")
    args = parser.parse_args()

    if args.poll:
        poll_job(args.poll)
    else:
        submit_from_local(args.train, epochs=args.epochs)
