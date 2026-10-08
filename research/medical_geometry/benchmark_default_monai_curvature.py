"""Real default MONAI DiNTS activation + instance norm: conditional C2 evidence."""

import json
import torch

from research.medical_geometry.test_default_monai_curvature import (
    validate_actual, B, SPACING,
)

def run():
    with torch.no_grad():
        mod,x,proof,repaired,error=validate_actual()
        raw=mod(x)
        t=error.detach()
        assert all(
            float(t[:,i:i+1].max()) <= bound+2e-4
            for i,bound in enumerate(repaired.conditional_value_error_bounds)
        )
        assert all(
            abs(float(t[:,i:i+1].max())-bound)<2e-4
            for i,bound in enumerate(repaired.conditional_value_error_bounds)
        )
        return {
            "runtime":"actual MONAI FactorizedReduceBlock with default ReLU + InstanceNorm3d",
            "trained_model":False,
            "input_function":"synthetic f(p)=1+B/2 ||p||^2",
            "uniform_hessian_premise":B,
            "input_positive_interval_premise":[1,14],
            "patient_mm_spacing":list(SPACING),
            "actual_model_output_shape":list(raw.shape),
            "default_activation":type(mod.act).__name__,
            "default_normalization":type(mod.norm).__name__,
            "normalization_actual_means":list(proof.normalization_mean),
            "normalization_sample_scales":list(proof.normalization_scale),
            "normalized_output_curvature_bounds":list(proof.normalized_output_bounds),
            "physical_interpolation_bounds":list(repaired.conditional_value_error_bounds),
            "executed_max_abs_errors":[
                float(t[:,i:i+1].max()) for i in range(2)
            ],
            "model_and_sample_provenance_sha256":proof.weights_and_sample_sha256,
            "caveat":"input interval and Hessian are analytic assumptions, not inferred from an unknown clinical input",
        }

if __name__=="__main__":
    print(json.dumps(run(),sort_keys=True,indent=2))
