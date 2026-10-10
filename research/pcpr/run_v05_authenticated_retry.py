"""Transport-only retry after the 8/8 v0.5 FIRST-score FETCH_FAILUREs.

Do not change the frozen semantic analyzer (commit f68acdb9...). Authenticated
GitHub API requests substitute ONLY for unauthenticated public API GETs.
Code and PR cohort unchanged, no rules retuned or examples replaced.
This file emits a separate, NEVER overwritten, transport-retry ledger.
"""
import json,os,urllib.request
import source_effect_ir_v05 as frozen
from run_v05_unseen_eight import PR_LIST

FIRST_FAILED_RUN="https://github.com/lindicaphxag-tech/kaggle/actions/runs/38029377120"
def request_json_authorized(url):
  token=os.environ.get("GITHUB_TOKEN","")
  if not token:raise RuntimeError("GITHUB_TOKEN_NOT_CONFIGURED")
  if not url.startswith("https://api.github.com/repos/"):
    raise ValueError("REQUEST_ORIGIN_NOT_TRUSTED")
  req=urllib.request.Request(url,headers={
    "Accept":"application/vnd.github+json",
    "Authorization":"Bearer "+token,
    "X-GitHub-Api-Version":"2022-11-28",
    "User-Agent":"pcpr-v05-authenticated-transport-retry"
  })
  with urllib.request.urlopen(req,timeout=45) as r:
    return json.loads(r.read())
frozen.request_json=request_json_authorized

def main():
  results=[]
  for repo,n in PR_LIST:
    try:results.append(frozen.assess_real_pr(repo,n))
    except Exception as e:
      code=getattr(e,"code",None)
      results.append({"url":f"https://github.com/{repo}/pull/{n}",
                      "status":"FETCH_FAILURE",
                      "http_status":code,
                      "exception_class":type(e).__name__,
                      "n_typed_changed_functions":0,
                      "n_typed_facets":0})
  data={"label":"V05_AUTHENTICATED_TRANSPORT_RETRY_AFTER_FIRST_FAILED_FETCH",
        "original_first_score":FIRST_FAILED_RUN,
        "semantic_algorithm_sha":"f68acdb92d59b8cbeef9f317eef779a9569563d1",
        "n_fixed":8,
        "n_fetched":sum(x["status"]!="FETCH_FAILURE" for x in results),
        "n_transport_failures":sum(x["status"]=="FETCH_FAILURE" for x in results),
        "n_typed_candidate_prs":sum(x["n_typed_changed_functions"]>0 for x in results),
        "n_native_replays":0,"n_correctness_decisions":0,
        "is_original_first_score":False,
        "results":results}
  with open("pcpr_v05_eight_auth_transport_retry.json","w") as f:json.dump(data,f,indent=2,sort_keys=True)
  print(json.dumps({k:v for k,v in data.items() if k!="results"},indent=2))
  for v in results:print(v["url"],v["status"],v["n_typed_changed_functions"],
                         ",".join(v.get("facet_union",[])),
                         v.get("http_status",""))
if __name__=="__main__":main()
