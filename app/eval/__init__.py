__all__ = [
    "BenchmarkReport",
    "ScenarioBenchmarkResult",
    "run_packetizer_benchmark",
    "threshold_failures",
    "CertificationReport",
    "CertificationCheckResult",
    "certify_domain_pack",
]


def __getattr__(name: str):
    if name in {"CertificationReport", "CertificationCheckResult", "certify_domain_pack"}:
        from app.eval import domain_certification as _domain_cert

        return getattr(_domain_cert, name)
    if name in __all__:
        from app.eval import benchmark as _benchmark

        return getattr(_benchmark, name)
    raise AttributeError(name)
