#!/usr/bin/env python3
"""Conditional quantized FINAL-FEATURE-MATRIX -> LDA rank-sign certificate.

At input we REQUIRE an externally justified exact entrywise bound
|Xtrue-Xstored|<=eta on final features. We derive a deterministic
Frobenius-based error for pooled within-class covariance, means and
positive-v-negative pair difference, then call rational
covariance-coordinate resolvent certification over ALL lambda in [0,1].
Quantization BEFORE EEG preprocessing/tangent transformations is NOT
covered: final-feature eta is NOT inferred from an EDF's int16 unit.
This is a synthetic mathematical operator test, not real EEG gain.
"""
import argparse,hashlib,json,math,random
from pathlib import Path
from fractions import Fraction as F
import sympy as sp
from p4_covariance_coordinate_resolvent_bound import certify, score

def frac(z):
    if isinstance(z,float):raise TypeError("float not justified as exact")
    return F(z)

def sqrt_upper_nonnegative(q,bits=24):
    q=frac(q)
    if q<0:raise ValueError("negative squared norm")
    scale=1<<bits
    upper_int=(q.numerator*scale*scale+q.denominator-1)//q.denominator
    return F(math.isqrt(upper_int)+1,scale)

def fit_pooled(X,labels,ridge):
    X=sp.Matrix(X)
    n,p=X.shape
    unique=sorted(set(labels))
    if len(unique)!=2 or n<=len(unique) or len(labels)!=n:
        raise ValueError("two nonempty classes and n>2 required")
    if any(labels.count(c)<2 for c in unique):
        raise ValueError("class too small")
    means={}
    R=X.copy()
    for c in unique:
        inds=[i for i,y in enumerate(labels) if y==c]
        m=sp.Matrix([sum(X[i,j] for i in inds)/len(inds) for j in range(p)])
        means[c]=m
        for i in inds:
            for j in range(p):R[i,j]=X[i,j]-m[j]
    den=n-len(unique)
    S=(R.T*R)/den+sp.eye(p)*ridge
    d=means[1]-means[0]
    return S,d,R

def error_bound(X,labels,eta):
    eta=frac(eta)
    if eta<0:raise ValueError("negative entrywise error")
    n,p=X.shape;nc=len(set(labels))
    _,_,R=fit_pooled(X,labels,F(1))
    H_upper=sqrt_upper_nonnegative(sum(frac(v)**2 for v in R))
    err=sqrt_upper_nonnegative(n*p)*eta
    epsS=(2*H_upper*err+err*err)/(n-nc)
    epsD=2*eta*sqrt_upper_nonnegative(p)
    epsV=epsD
    return epsS,epsD,epsV

def regression():
    rng=random.Random(20261013)
    counts={"POS":0,"NEG":0,"UNKNOWN":0}
    checks=0;cases=64
    examples=[]
    for k in range(cases):
        n=24;p=rng.choice((2,3,4));eta=F(1,rng.choice((400,800,1600)))
        labels=[0]*(n//2)+[1]*(n//2)
        # Frozen final stored features, no EEG raw signal input.
        center=[F(rng.randint(1,3),1) for _ in range(p)]
        X=sp.Matrix([[((F(-1) if labels[i]==0 else F(1))*center[j])+
                      F(rng.randint(-15,15),10)
                      for j in range(p)] for i in range(n)])
        S,d,_=fit_pooled(X,labels,F(1,5))
        u=rng.randrange(n//2,n);v0=rng.randrange(0,n//2)
        vec=(X.row(u)-X.row(v0)).T
        if k%4==0:vec=-vec
        es,ed,ev=error_bound(X,labels,eta)
        decision=certify(S,vec,d,es,ev,ed)
        counts[decision["status"]]+=1
        if decision["status"]!="UNKNOWN":
            # This samples independent admissible true feature matrices;
            # the algebraic covariance bound holds for ALL such matrices.
            for rep in range(3):
                Z=sp.Matrix([[X[i,j]+F(rng.choice((-1,0,1)))*eta
                              for j in range(p)] for i in range(n)])
                SZ,dZ,_=fit_pooled(Z,labels,F(1,5))
                vz=(Z.row(u)-Z.row(v0)).T
                if k%4==0:vz=-vz
                for t in (sp.Rational(0),sp.Rational(1,2),sp.Rational(1)):
                    val=score(SZ,vz,dZ,t)
                    if decision["status"]=="POS":assert val>0
                    else:assert val<0
                    checks+=1
        if len(examples)<5:
            examples.append({"fixture":k,"sign":decision["status"],
                  "eps_covariance":str(es),"eps_contrast":str(ed),
                  "epsilon_source":str(eta),
                  "uncertainty_margin":decision.get("whole_interval_score_error_upper")})
    # An adversarial and VALID final-feature quantization-box counterexample.
    # The nominal pair and class contrast are aligned, but perturbing
    # one trial in each class flips the pair's sign while group mean
    # contrast remains positive. The theorem MUST refuse certification.
    labs=[0]*12+[1]*12
    X0=sp.Matrix([[F(-1,100) if lab==0 else F(1,100),F(0)]
                  for lab in labs])
    s0,d0,_=fit_pooled(X0,labs,F(1,5))
    pair0=(X0.row(12)-X0.row(0)).T
    eps0,epsd0,epsv0=error_bound(X0,labs,F(1,10))
    cert0=certify(s0,pair0,d0,eps0,epsv0,epsd0)
    assert cert0["status"]=="UNKNOWN"
    X1=X0.copy();X1[12,0]-=sp.Rational(1,10)
    X1[0,0]+=sp.Rational(1,10)
    s1,d1,_=fit_pooled(X1,labs,F(1,5))
    pair1=(X1.row(12)-X1.row(0)).T
    for t in (sp.Rational(0),sp.Rational(1,2),sp.Rational(1)):
        assert score(s0,pair0,d0,t)>0
        assert score(s1,pair1,d1,t)<0
    signed_flip_witness={
       "status":"EXACT_ADMISSIBLE_FINAL_FEATURE_SIGN_REVERSAL",
       "entrywise_epsilon":"1/10",
       "changed_final_feature_coordinates":[[0,0],[12,0]],
       "nominal_pair_score_positive_for_tested_lambda":True,
       "admissible_perturbed_pair_score_negative_for_tested_lambda":True,
       "certifier_correctly_returned_UNKNOWN":True}
    assert sum(counts.values())==cases
    return {"status":"PASS",
       "synthetic_feature_matrices":cases,"seed":20261013,
       "decisions":counts,"admissible_rational_feature_perturbation_checks":checks,
       "stored_final_features_per_instance":24,
       "entrywise_error_bound_assumed_externally":True,
       "class_residual_covariance_and_class_means_recomputed":True,
       "sample_count_unit_not_independent_acquisition":True,
       "no_bound_through_EEG_preprocessing_or_tangent_transform":True,
       "not_IEEE754_or_actual_EEG_calibration":True,
       "adversarial_quantized_feature_sign_flip":signed_flip_witness,
       "examples":examples}

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    output=regression()
    data=json.dumps(output,sort_keys=True,indent=2)+"\n"
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(data)
    print("P4_FINAL_FEATURE_QUANTIZATION_OPERATOR_SCOPE_PASS",json.dumps(
      {"n":output["synthetic_feature_matrices"],"decisions":output["decisions"],
       "oracle":output["admissible_rational_feature_perturbation_checks"]},
      sort_keys=True))
    print("SHA256",hashlib.sha256(data.encode()).hexdigest())
