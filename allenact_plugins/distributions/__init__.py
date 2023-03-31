from allenact_plugins.distributions.distributions import (
    GaussianDistr,
    TruncatedNormal,
    DictActionSpaceDistr,
    StateDependentNoiseDistribution,
)

POSSIBLE_DIST = {
    "TruncatedNormal": TruncatedNormal,
    "GaussianDistr": GaussianDistr,
    "gSDE": StateDependentNoiseDistribution,
}
