"""Configuration profiles for SAGE Agent v3.

Different configurations for different use cases:
- CONSERVATIVE: Maximum calibration, lower coverage (original v3)
- BALANCED: Good calibration with better coverage (recommended)
- AGGRESSIVE: Maximum coverage, lower calibration (close to v2)
- LITE: Disable expensive features for speed
"""

# Original v3 config (too conservative for When2Call)
CONSERVATIVE = {
    "confidence_threshold": 0.7,
    "uncertainty_threshold": 0.3,
    "max_attempts": 3,
    "max_execution_retries": 2,

    "enable_sgr": True,
    "per_field_uncertainty": True,
    "sgr_reasoning_steps": True,

    "enable_resampling": True,
    "base_samples": 1,
    "max_samples": 5,
    "high_uncertainty_sample_threshold": 0.6,
    "agreement_threshold": 0.7,

    "enable_saup_tracking": True,
    "saup_escalation_threshold": 0.85,
    "track_reasoning_traces": True,

    "enable_reflexion": True,
    "max_reflexion_attempts": 2,
    "reflexion_only_on_failure": True,
    "reflexion_uncertainty_threshold": 0.7,

    "structured_weight": 0.7,
    "llm_modulation": 0.5,

    "critical_patterns": ["delete", "cancel", "remove", "drop", "terminate", "destroy"],
    "critical_threshold_multiplier": 0.5,

    "escalation_uncertainty": 0.85,
    "max_high_uncertainty_steps": 3,
}

# Balanced config (recommended for When2Call)
BALANCED = {
    "confidence_threshold": 0.7,
    "uncertainty_threshold": 0.3,
    "max_attempts": 3,
    "max_execution_retries": 2,

    "enable_sgr": True,
    "per_field_uncertainty": True,
    "sgr_reasoning_steps": True,

    "enable_resampling": True,
    "base_samples": 1,
    "max_samples": 3,  # Reduced
    "high_uncertainty_sample_threshold": 0.7,  # Increased
    "agreement_threshold": 0.7,

    "enable_saup_tracking": True,
    "saup_escalation_threshold": 0.95,  # Increased
    "track_reasoning_traces": True,

    "enable_reflexion": True,
    "max_reflexion_attempts": 2,
    "reflexion_only_on_failure": True,
    "reflexion_uncertainty_threshold": 0.8,  # Increased

    "structured_weight": 0.7,
    "llm_modulation": 0.5,

    "critical_patterns": ["delete", "cancel", "remove", "drop", "terminate", "destroy"],
    "critical_threshold_multiplier": 0.5,

    "escalation_uncertainty": 0.95,  # Increased
    "max_high_uncertainty_steps": 5,  # Increased
}

# Aggressive config (maximize coverage, close to v2 behavior)
AGGRESSIVE = {
    "confidence_threshold": 0.6,  # Lower
    "uncertainty_threshold": 0.4,  # Higher
    "max_attempts": 4,
    "max_execution_retries": 3,

    "enable_sgr": True,
    "per_field_uncertainty": True,
    "sgr_reasoning_steps": True,

    "enable_resampling": False,  # DISABLED for speed
    "base_samples": 1,
    "max_samples": 1,
    "high_uncertainty_sample_threshold": 0.9,
    "agreement_threshold": 0.5,

    "enable_saup_tracking": False,  # DISABLED to avoid escalation
    "saup_escalation_threshold": 0.99,
    "track_reasoning_traces": False,

    "enable_reflexion": True,
    "max_reflexion_attempts": 1,  # Reduced
    "reflexion_only_on_failure": True,
    "reflexion_uncertainty_threshold": 0.9,

    "structured_weight": 0.8,  # Rely more on structured
    "llm_modulation": 0.3,

    "critical_patterns": ["delete", "cancel", "remove", "drop", "terminate", "destroy"],
    "critical_threshold_multiplier": 0.5,

    "escalation_uncertainty": 0.99,
    "max_high_uncertainty_steps": 10,
}

# Lite config (SGR only, no resampling/SAUP for max speed)
LITE = {
    "confidence_threshold": 0.7,
    "uncertainty_threshold": 0.3,
    "max_attempts": 3,
    "max_execution_retries": 2,

    "enable_sgr": True,  # Keep SGR - it's cheap and valuable
    "per_field_uncertainty": True,
    "sgr_reasoning_steps": False,  # Disable trace

    "enable_resampling": False,  # DISABLED
    "base_samples": 1,
    "max_samples": 1,
    "high_uncertainty_sample_threshold": 0.9,
    "agreement_threshold": 0.7,

    "enable_saup_tracking": False,  # DISABLED
    "saup_escalation_threshold": 0.99,
    "track_reasoning_traces": False,

    "enable_reflexion": False,  # DISABLED
    "max_reflexion_attempts": 0,
    "reflexion_only_on_failure": True,
    "reflexion_uncertainty_threshold": 0.9,

    "structured_weight": 0.7,
    "llm_modulation": 0.5,

    "critical_patterns": ["delete", "cancel", "remove", "drop", "terminate", "destroy"],
    "critical_threshold_multiplier": 0.5,

    "escalation_uncertainty": 0.9,
    "max_high_uncertainty_steps": 5,
}

# Default: Use balanced
DEFAULT = BALANCED

# For eval script integration
def get_config(profile: str = "balanced") -> dict:
    """Get configuration by profile name.

    Args:
        profile: One of "conservative", "balanced", "aggressive", "lite"

    Returns:
        Configuration dictionary
    """
    profiles = {
        "conservative": CONSERVATIVE,
        "balanced": BALANCED,
        "aggressive": AGGRESSIVE,
        "lite": LITE,
    }

    profile_lower = profile.lower()
    if profile_lower not in profiles:
        raise ValueError(
            f"Unknown profile '{profile}'. "
            f"Choose from: {', '.join(profiles.keys())}"
        )

    return profiles[profile_lower].copy()
