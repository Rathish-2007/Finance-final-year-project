from django.db import models

MODULES = [
    ("data", "M1 · Unified Data Lake"),
    ("sscdv", "M2 · SSCDV Embeddings"),
    ("regimes", "M3 · Continuous Jump Model"),
    ("connectedness", "M4 · TVP-VAR + QQC"),
    ("risk", "M5 · Threshold & Stability Lab"),
    ("drl", "M6 · Hierarchical DRL Trader"),
]


class ModuleState(models.Model):
    """State + serialized results for one pipeline module."""

    name = models.CharField(max_length=32, unique=True)
    status = models.CharField(max_length=16, default="pending")  # pending|running|done|error
    progress = models.FloatField(default=0.0)
    message = models.TextField(blank=True, default="")
    results = models.JSONField(null=True, blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["id"]

    def __str__(self):
        return f"{self.name}:{self.status}"


class RunLog(models.Model):
    """Append-only execution log for the pipeline runner."""

    ts = models.DateTimeField(auto_now_add=True)
    module = models.CharField(max_length=32)
    level = models.CharField(max_length=8, default="INFO")
    message = models.TextField()

    class Meta:
        ordering = ["ts"]
