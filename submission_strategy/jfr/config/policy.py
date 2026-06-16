"""Policy profile loader — reads data/policy.toml."""

import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore


class OpenAccessPolicy(BaseModel):
    fully_oa_allowed: bool = False
    hybrid_oa_allowed: bool = True


class PublisherPolicy(BaseModel):
    blocklist: list[str] = []
    preferlist: list[str] = ["acs", "elsevier", "wiley", "rsc"]


class QualityPolicy(BaseModel):
    impact_factor_floor: float = 4.0
    prefer_society_journals: bool = True


class ApprovalsPolicy(BaseModel):
    require_supervisor_approval: bool = True
    supervisor_contact: Optional[str] = None


class ExcludedJournal(BaseModel):
    id: str
    reason: str


class ExclusionsPolicy(BaseModel):
    journals: list[ExcludedJournal] = []


class Policy(BaseModel):
    open_access: OpenAccessPolicy = OpenAccessPolicy()
    publishers: PublisherPolicy = PublisherPolicy()
    quality: QualityPolicy = QualityPolicy()
    approvals: ApprovalsPolicy = ApprovalsPolicy()
    exclusions: ExclusionsPolicy = ExclusionsPolicy()
    # Journals explicitly allowed regardless of IF floor (e.g. historical target venues)
    allow_journals: list[str] = []

    def is_journal_excluded(self, journal_id: str) -> tuple[bool, str]:
        for ex in self.exclusions.journals:
            if ex.id == journal_id:
                return True, ex.reason
        return False, ""

    def check_open_access(self, is_fully_oa: bool, is_hybrid_oa: bool) -> tuple[bool, str]:
        if is_fully_oa and not self.open_access.fully_oa_allowed:
            return False, "fully OA not allowed by policy"
        return True, ""

    def check_impact_factor(self, impact_factor: Optional[float], journal_id: str = "") -> tuple[bool, str]:
        if impact_factor is None:
            return True, ""
        if journal_id and journal_id in self.allow_journals:
            return True, ""
        if impact_factor < self.quality.impact_factor_floor:
            return False, f"IF {impact_factor:.1f} below floor {self.quality.impact_factor_floor:.1f}"
        return True, ""

    def check_publisher(self, publisher_family: str) -> tuple[bool, str]:
        if publisher_family.lower() in [b.lower() for b in self.publishers.blocklist]:
            return False, f"publisher '{publisher_family}' is blocklisted"
        return True, ""


def load_policy(path: Path) -> Policy:
    if not path.exists():
        return Policy()
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return Policy.model_validate(data)
