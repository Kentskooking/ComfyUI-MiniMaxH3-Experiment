from .latent_composite import H3LatentCompositeMasked
from .masked_guider import H3MaskedConditioningGuider
from .mask_profile import H3MaskProfile
from .mask_profile_preview import H3MaskProfilePreview
from .reference_blend import H3OmniReferenceBlend
from .virtual_head_start import H3VirtualHeadStart

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "H3ExperimentsLatentCompositeMasked": H3LatentCompositeMasked,
    "H3ExperimentsMaskedConditioningGuider": H3MaskedConditioningGuider,
    "H3ExperimentsMaskProfile": H3MaskProfile,
    "H3ExperimentsMaskProfilePreview": H3MaskProfilePreview,
    "H3ExperimentsOmniReferenceBlend": H3OmniReferenceBlend,
    "H3ExperimentsVirtualHeadStart": H3VirtualHeadStart,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ExperimentsLatentCompositeMasked": "H3 Latent Composite Masked",
    "H3ExperimentsMaskedConditioningGuider": "H3 Masked Conditioning Guider",
    "H3ExperimentsMaskProfile": "H3 Mask Profile",
    "H3ExperimentsMaskProfilePreview": "H3 Mask Profile Preview",
    "H3ExperimentsOmniReferenceBlend": "H3 Omni Reference Blend",
    "H3ExperimentsVirtualHeadStart": "H3 Virtual Head Start",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
