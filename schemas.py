from enum import Enum
from typing import Optional, List, Literal
from pydantic import BaseModel
from datetime import date

class PlanTypeEnum(str, Enum):
    oap_open_access_plus = "OAP (Open Access Plus)"
    ppo = "PPO"
    hmo = "HMO"
    indemnity = "Indemnity"
    localplus = "LocalPlus"
    surefit = "SureFit"
    medical_network = "Medical Network"


class NetworkTypeEnum(str, Enum):
    national = "National"
    local = "Local"
    limited = "Limited"
    open = "Open"
    none_indemnity = "None (Indemnity)"


class PcpRequirementEnum(str, Enum):
    required = "Required"
    optional = "Optional"
    not_required = "Not Required"


class AutoPcpAssignmentEnum(str, Enum):
    included = "Included"
    optional = "Optional"
    not_included = "Not Included"


class ReferralRequirementEnum(str, Enum):
    required = "Required"
    not_required = "Not Required"
    varies_by_plan = "Varies by Plan"


class OutOfNetworkCoverageEnum(str, Enum):
    included_at_a_cost = "Included at a cost"
    emergencies_only = "Emergencies Only"
    not_covered = "Not Covered"
    available_no_restrictions = "Available (no network restrictions)"


class UrgentEmergentServicesCoverageEnum(str, Enum):
    included_in_network = "Included (in-network level)"
    included_any_provider = "Included (any provider)"
    varies_by_plan = "Varies by plan"


class FundingOptionEnum(str, Enum):
    fully_insured = "Fully Insured"
    self_funded_aso = "Self-Funded (ASO)"
    minimum_premium = "Minimum Premium"


class NetworkSizeEnum(str, Enum):
    large_national = "Large National"
    smaller_local = "Smaller Local"
    limited_focused = "Limited/Focused"
    none_indemnity = "None (Indemnity)"


class BusinessSizeEligibilityEnum(str, Enum):
    small_group_2_50 = "2-50 employees (Small Group)"
    employees_2_99 = "2-99 employees"
    employees_100_499 = "100-499 employees"
    employees_500_2999 = "500-2,999 employees"
    enterprise_3000_plus = "Enterprise (3,000+)"
    all_sizes = "All sizes"


class SelfFundedOptionAvailableEnum(str, Enum):
    yes = "Yes"
    no = "No"


class HsaHraFsaOptionEnum(str, Enum):
    available = "Available"
    not_available = "Not available"


class ProviderNetworkAccessEnum(str, Enum):
    in_network_only = "In-network only"
    in_and_out_of_network = "In- and out-of-network"
    any_provider_indemnity = "Any provider (Indemnity)"


class BusinessProfile(BaseModel):
    business_name: str
    business_size: Literal["2–99", "100–499", "500–2,999", "3,000+"]
    business_type: Literal[
        "Hospital and Health Systems", "Higher Education", "K-12 Education",
        "State and Local Governments", "Taft-Hartley and Federal",
        "Third Party Administrators (Payer Solutions)"
    ]
    business_state: str
    creation_date: date
    network_type: Optional[List[NetworkTypeEnum]] = None
    pcp_requirement: Optional[List[PcpRequirementEnum]] = None
    auto_pcp_assignment: Optional[List[AutoPcpAssignmentEnum]] = None
    referral_requirement: Optional[List[ReferralRequirementEnum]] = None
    out_of_network_coverage: Optional[List[OutOfNetworkCoverageEnum]] = None
    urgent_emergent_services_coverage: Optional[List[UrgentEmergentServicesCoverageEnum]] = None
    funding_option: Optional[List[FundingOptionEnum]] = None
    network_size: Optional[List[NetworkSizeEnum]] = None
    business_size_eligibility: Optional[List[BusinessSizeEligibilityEnum]] = None
    self_funded_option_available: Optional[List[SelfFundedOptionAvailableEnum]] = None
    hsa_hra_fsa_option: Optional[List[HsaHraFsaOptionEnum]] = None
    provider_network_access: Optional[List[ProviderNetworkAccessEnum]] = None


class PlanDiscoveryAnswers(BaseModel):
    business_size: Optional[Literal["2-50", "51-99", "100-499", "500-2,999", "3,000+"]] = None
    location: Optional[str] = None
    coverage_preference: Optional[Literal["National", "Local"]] = None

class PlanDiscoveryResponse(BaseModel):
    plan_discovery_answers: PlanDiscoveryAnswers
    response: str

