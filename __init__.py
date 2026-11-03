from .latent_composite import H3LatentCompositeMasked
from .masked_guider import H3MaskedConditioningGuider
from .regional_attention import H3RegionalAttentionGuider
from .regional_interaction import H3RegionalInteractionGuider
from .continuation import H3ContinuationSampler, H3LoadContinuationState
from .mask_profile import H3MaskProfile
from .mask_profile_preview import H3MaskProfilePreview
from .reference_blend import H3OmniReferenceBlend
from .virtual_head_start import H3VirtualHeadStart
from .timeline_offset import H3GeneratedTimelineOffset

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "H3ExperimentsLatentCompositeMasked": H3LatentCompositeMasked,
    "H3ExperimentsMaskedConditioningGuider": H3MaskedConditioningGuider,
    "H3ExperimentsRegionalAttentionGuider": H3RegionalAttentionGuider,
    "H3ExperimentsRegionalInteractionGuider": H3RegionalInteractionGuider,
    "H3ExperimentsContinuationSampler": H3ContinuationSampler,
    "H3ExperimentsLoadContinuationState": H3LoadContinuationState,
    "H3ExperimentsMaskProfile": H3MaskProfile,
    "H3ExperimentsMaskProfilePreview": H3MaskProfilePreview,
    "H3ExperimentsOmniReferenceBlend": H3OmniReferenceBlend,
    "H3ExperimentsVirtualHeadStart": H3VirtualHeadStart,
    "H3ExperimentsGeneratedTimelineOffset": H3GeneratedTimelineOffset,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ExperimentsLatentCompositeMasked": "H3 Latent Composite Masked",
    "H3ExperimentsMaskedConditioningGuider": "H3 Masked Conditioning Guider",
    "H3ExperimentsRegionalAttentionGuider": "H3 Regional Attention Guider (Beta)",
    "H3ExperimentsRegionalInteractionGuider": "H3 Regional Interaction Guider (Beta)",
    "H3ExperimentsContinuationSampler": "H3 Continuation Sampler (Beta)",
    "H3ExperimentsLoadContinuationState": "H3 Load Continuation State (Beta)",
    "H3ExperimentsMaskProfile": "H3 Mask Profile",
    "H3ExperimentsMaskProfilePreview": "H3 Mask Profile Preview",
    "H3ExperimentsOmniReferenceBlend": "H3 Omni Reference Blend",
    "H3ExperimentsVirtualHeadStart": "H3 Virtual Head Start",
    "H3ExperimentsGeneratedTimelineOffset": "H3 Generated Timeline Forward Offset (Beta)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
