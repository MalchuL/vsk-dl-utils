from torchmetrics import (
    MeanAbsoluteError,
    MeanSquaredError,
    MetricCollection,
    MultiScaleStructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
)


def get_regression_metrics(prefix="val/"):
    collection = MetricCollection(
        prefix=prefix, metrics={"L1": MeanAbsoluteError(), "L2": MeanSquaredError()}
    )
    return collection


def get_alignment_metrics(prefix="val/"):
    collection = MetricCollection(
        prefix=prefix,
        metrics={
            "L1": MeanAbsoluteError(),
            "L2": MeanSquaredError(),
            "PSNR": PeakSignalNoiseRatio(),
            "MSSIM": MultiScaleStructuralSimilarityIndexMeasure(),
        },
    )
    return collection
