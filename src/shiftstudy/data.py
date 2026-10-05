"""Uniform subset preparation from complete canonical CSVs, without extraction."""

from contextlib import contextmanager
from itertools import combinations
from pathlib import Path
import hashlib
import io
import json
import tarfile

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

META = ("fact_time", "fact_latitude", "fact_longitude", "fact_temperature",
        "fact_cwsm_class", "climate")
SPLITS = ("train", "dev_in", "eval_in", "eval_out")


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class HashingReader(io.RawIOBase):
    """Hash the actual CSV bytes consumed by pandas."""

    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def readable(self):
        return True

    def readinto(self, buffer):
        block = self.stream.read(len(buffer))
        self.digest.update(block)
        buffer[:len(block)] = block
        return len(block)


def feature_columns(columns):
    columns = list(columns)
    if tuple(columns[:6]) != META:
        raise ValueError("Expected the six canonical metadata/target columns first")
    features = columns[6:]
    if not features or any(c.startswith("fact_") or c == "climate" for c in features):
        raise ValueError("Target or metadata detected in the feature columns")
    if len(set(columns)) != len(columns):
        raise ValueError("Duplicate columns")
    return features


def integer_labels(values):
    labels = np.asarray(values, dtype=float)
    if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError("Labels must be finite integers")
    return labels.astype(np.int64)


def sample_csv(stream, size, seed, *, training=False, chunk_rows=20000):
    """Keep the k smallest independent random priorities over eligible rows.

    Every eligible row has equal inclusion probability k/N. Randomness is drawn
    for every source row, including rejected rows, so chunk size is irrelevant.
    A fixed-size chunk plus the retained sample bounds memory usage.
    """
    if not isinstance(size, int) or size < 1 or chunk_rows < 1:
        raise ValueError("Sample and chunk sizes must be positive integers")
    rng = np.random.default_rng(seed)
    chosen = None
    priorities = np.empty(0)
    offset = excluded = 0
    classes = set()
    schema = None
    reader = HashingReader(stream)
    buffered = io.BufferedReader(reader)
    for frame in pd.read_csv(buffered, chunksize=chunk_rows):
        current = feature_columns(frame.columns)
        if schema is not None and current != schema:
            raise ValueError("CSV schema changed between chunks")
        schema = current
        labels = integer_labels(frame["fact_cwsm_class"])
        # Full training label vocabulary can include upstream placeholder labels;
        # these labels define columns only, never fabricated training examples.
        classes.update(labels.tolist())
        ids = np.arange(offset, offset + len(frame), dtype=np.int64)
        key = rng.random(len(frame))
        offset += len(frame)
        times = np.asarray(frame["fact_time"], dtype=float)
        valid_meta = (np.isfinite(times) & (times == np.floor(times))
                      & frame["climate"].notna().to_numpy()
                      & frame["fact_latitude"].notna().to_numpy()
                      & frame["fact_longitude"].notna().to_numpy())
        if not training and not valid_meta.all():
            raise ValueError("Non-observational or malformed row outside training")
        excluded += int((~valid_meta).sum())
        frame = frame.loc[valid_meta].copy()
        frame["source_row"] = ids[valid_meta]
        key = key[valid_meta]
        merged = frame if chosen is None else pd.concat([chosen, frame], ignore_index=True)
        all_keys = np.concatenate([priorities, key])
        # Source row is the deterministic tie breaker for finite-precision keys.
        keep = np.lexsort((merged["source_row"].to_numpy(), all_keys))[:size]
        chosen = merged.iloc[keep].reset_index(drop=True)
        priorities = all_keys[keep]
    if chosen is None or len(chosen) != size:
        raise ValueError(f"Requested {size} observations; source has fewer eligible rows")
    chosen = chosen.sort_values("source_row").reset_index(drop=True)
    return chosen, {"total_rows": offset, "excluded_rows": excluded,
                    "csv_sha256": reader.digest.hexdigest(),
                    "label_vocabulary": sorted(classes), "features": schema}


@contextmanager
def canonical_sources(source):
    """Read only the four needed members; never extract archive paths."""
    source = Path(source)
    filenames = {s: (f"shifts_canonical_{s}.csv", f"{s}.csv") for s in SPLITS}
    if source.is_dir():
        paths = {}
        for split, names in filenames.items():
            found = [source / name for name in names if (source / name).is_file()]
            if len(found) != 1:
                raise ValueError(f"Need exactly one canonical CSV for {split}: {names}")
            paths[split] = found[0]
        yield lambda s: paths[s].open("rb")
    else:
        with tarfile.open(source, "r:*") as archive:
            members = {}
            for split, names in filenames.items():
                found = [m for m in archive.getmembers()
                         if m.isfile() and Path(m.name).name in names]
                if len(found) != 1:
                    raise ValueError(f"Need exactly one archive CSV for {split}")
                members[split] = found[0]
            yield lambda s: archive.extractfile(members[s])


def observation_keys(frame):
    return list(zip(frame["fact_time"].astype(np.int64),
                    frame["fact_latitude"].astype(float),
                    frame["fact_longitude"].astype(float)))


def validate_observations(frames):
    keys = {s: observation_keys(f) for s, f in frames.items()}
    within = {s: len(k) - len(set(k)) for s, k in keys.items()}
    cross = {f"{a}:{b}": len(set(keys[a]) & set(keys[b]))
             for a, b in combinations(keys, 2)}
    if any(within.values()) or any(cross.values()):
        raise ValueError(f"Duplicate observation keys in selected data: {within}; {cross}")
    return {"within_split_duplicates": within, "cross_split_overlap": cross}


def prepare_weather(source, output, config):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prepared data: {output}")
    frames, provenance = {}, {}
    with canonical_sources(source) as opener:
        for i, split in enumerate(SPLITS):
            print(f"Sampling complete {split} CSV...", flush=True)
            with opener(split) as stream:
                frames[split], provenance[split] = sample_csv(
                    stream, config["sizes"][split], config["seed"] + i,
                    training=split == "train")
    features = provenance["train"]["features"]
    if any(p["features"] != features for p in provenance.values()):
        raise ValueError("Feature schema differs across canonical splits")
    overlap = validate_observations(frames)
    classes = np.array(provenance["train"]["label_vocabulary"], dtype=np.int64)
    dev_order = np.random.default_rng(config["seed"] + 100).permutation(len(frames["dev_in"]))
    if len(dev_order) % 3:
        raise ValueError("dev_in size must be divisible by three")
    val, temp, conformal = np.split(dev_order, 3)
    dev = frames.pop("dev_in")
    for name, indices in zip(("validation", "temperature", "conformal"), (val, temp, conformal)):
        frames[name] = dev.iloc[indices].reset_index(drop=True)
    arrays, descriptions = {}, {}
    for split, frame in frames.items():
        raw_y = integer_labels(frame["fact_cwsm_class"])
        if not np.isin(raw_y, classes).all():
            raise ValueError(f"{split} contains labels outside full-training vocabulary")
        x = frame[features].to_numpy(dtype=np.float64)
        x[~np.isfinite(x)] = np.nan
        y = np.searchsorted(classes, raw_y)
        arrays[f"{split}_x"] = x
        arrays[f"{split}_y"] = y
        arrays[f"{split}_rows"] = frame["source_row"].to_numpy(dtype=np.int64)
        descriptions[split] = {
            "n": len(frame),
            "label_counts": {str(c): int((raw_y == c).sum()) for c in classes},
            "climate_counts": frame["climate"].value_counts().to_dict(),
            "time_min": int(frame["fact_time"].min()),
            "time_max": int(frame["fact_time"].max()),
            "missing_feature_fraction": float(np.isnan(x).mean()),
        }
    output.mkdir(parents=True)
    np.savez_compressed(output / "data.npz", **arrays)
    metadata = {"kind": "shifts_weather", "preparation": config,
                "source_name": Path(source).name,
                "archive_sha256": sha256_file(source) if Path(source).is_file() else None,
                "upstream_license": "CC BY-NC-SA 4.0", "classes": classes.tolist(),
                "features": features, "canonical_sources": provenance,
                "observation_audit": overlap, "splits": descriptions,
                "data_sha256": sha256_file(output / "data.npz")}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def fit_preprocessor(train_x):
    transformer = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                StandardScaler())
    transformer.fit(train_x)
    return transformer


def load_prepared(path):
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    if sha256_file(path / "data.npz") != metadata["data_sha256"]:
        raise ValueError("Prepared data hash does not match metadata")
    with np.load(path / "data.npz", allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    return arrays, metadata
