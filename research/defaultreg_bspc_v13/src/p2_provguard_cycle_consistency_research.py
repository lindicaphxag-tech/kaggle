"""Research component: globally consistent time-offset graph for VERIFIED EEG witness edges.
Standard potential-union-find algebra; does not establish physical acquisition identity.
"""
from dataclasses import dataclass
@dataclass(frozen=True)
class Edge:
    source:str
    target:str
    lag:int
    confidence:float
def coherent_forest(nodes,edges):
    nodes=list(nodes)
    if len(set(nodes))!=len(nodes):raise ValueError('duplicate nodes')
    parent={x:x for x in nodes};delta={x:0 for x in nodes}
    def find(x):
        if parent[x]!=x:
            r,d=find(parent[x]);delta[x]+=d;parent[x]=r
        return parent[x],delta[x]
    accepted=[];rejected=[]
    for e in sorted(edges,key=lambda e:(-e.confidence,e.source,e.target,e.lag)):
        if e.source not in parent or e.target not in parent:raise ValueError('unknown source')
        if not 0<=e.confidence<=1:raise ValueError('invalid confidence')
        a,da=find(e.source);b,db=find(e.target)
        if a==b:
            error=(db-da)-e.lag
            if error:rejected.append(dict(edge=e.__dict__,residual_samples=error))
            else:accepted.append(e.__dict__)
        else:
            parent[b]=a;delta[b]=e.lag+da-db
            accepted.append(e.__dict__)
    return dict(accepted=accepted,cycle_contradictions=rejected,
                potentials={x:dict(component=find(x)[0],offset=find(x)[1]) for x in nodes})
def verify_graph(nodes,edges,certificate):
    return certificate==coherent_forest(nodes,edges)
