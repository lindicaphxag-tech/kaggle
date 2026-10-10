"""v0.9 third-cohort authenticated TRANSPORT diagnostic and retry.

The FIRST frozen attempt on run 38033004178 recorded 32 HTTPError
fetch failures, zero parsed PRs. This is a separate retry, not an edited
algorithm or replacement cohort. Semantic functions imported unchanged
from v0.9 pinned commit 471ea5d344b234defa6491c283395233750bcffd.
No token is ever printed, persisted or passed to upstream project code.
"""
from __future__ import annotations
import json,os,urllib.error,urllib.request
import run_v09_new_32_holdout as cohort

ORIGINAL="https://github.com/lindicaphxag-tech/kaggle/actions/runs/38033004178"

def probe():
 token=os.environ.get("GITHUB_TOKEN","")
 if not token:return {"status":"TOKEN_UNAVAILABLE"}
 url="https://api.github.com/repos/EleutherAI/bergson/pulls/488"
 req=urllib.request.Request(url,headers={
  "Accept":"application/vnd.github+json",
  "Authorization":"Bearer "+token,
  "X-GitHub-Api-Version":"2022-11-28",
  "User-Agent":"pcpr-v09-third-cohort-transport-diagnosis"})
 try:
  with urllib.request.urlopen(req,timeout=25) as x:
   return {"status":"OK","http_status":x.status,
           "rate_remaining":x.headers.get("X-RateLimit-Remaining"),
           "rate_reset_unix":x.headers.get("X-RateLimit-Reset"),
           "retry_after":x.headers.get("Retry-After")}
 except urllib.error.HTTPError as e:
  return {"status":"HTTP_ERROR","http_status":e.code,
          "rate_remaining":e.headers.get("X-RateLimit-Remaining"),
          "rate_reset_unix":e.headers.get("X-RateLimit-Reset"),
          "retry_after":e.headers.get("Retry-After"),
          "message_class":"RATE_LIMIT_OR_ACCESS_UNKNOWN"}
 except Exception as e:
  return {"status":"TRANSPORT_EXCEPTION","exception_type":type(e).__name__}

def main():
 diagnostic=probe()
 out={"status":"AUTHENTICATION_ONLY_RETRY",
      "original_first_score":ORIGINAL,
      "semantic_algorithm_commit":"471ea5d344b234defa6491c283395233750bcffd",
      "original_n_fetch_failures":32,
      "fixed_n_pr":32,
      "transport_diagnostic":diagnostic,
      "is_original_first_score":False,
      "n_fetched":0,"n_native_v09_pr":None,
      "n_correctness_decisions":0}
 if diagnostic["status"]=="OK":
  result=cohort.main()
  out["n_fetched"]=32-result["number_pr_fetch_failures"]
  out["n_native_v09_pr"]=result["n_pr_v09_native"]
  out["n_cast_candidate_pr"]=sum(
     z["v09_cast_candidates"]>0 for z in result["rows"])
  out["n_v05_candidate_pr"]=result["n_pr_v05_typed_candidates"]
  out["n_fetch_failures"]=result["number_pr_fetch_failures"]
  out["status"]="AUTHENTICATED_RETRY_COMPLETED" if out["n_fetched"]==32 else "PARTIAL_TRANSPORT_RETRY"
 else:
  out["status"]="RETRY_BLOCKED_BY_TRANSPORT"
 with open("pcpr_v09_auth_transport_retry_after_32_http_errors.json","w") as f:
  json.dump(out,f,indent=2,sort_keys=True)
 print(json.dumps(out,indent=2))
 return out
if __name__=="__main__":main()
