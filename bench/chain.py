import bench as B, prtool as P, numpy as np, json, os
lab=os.environ.get("BLEED_PREFIX", "bleed_")
LF=["--secam-lf-extra-break","50000","--secam-lf-extra-ratio","3"]; G=["--secam-luma-guide","@LUMA"]
K=lambda f: ["--secam-bleed-db",lab+f+"_db.txt","--secam-bleed-dr",lab+f+"_dr.txt"]
steps=[("main",lambda f:("upstream",[])),("+ LF shelf",lambda f:("labo",LF)),("+ luma guide",lambda f:("labo",LF+G)),("+ bleed",lambda f:("labo",LF+G+K(f)))]
out={}
for film in ("diner","shrek"):
    R={s[0]:[] for s in steps}
    for base in B.SPAN_FILMS[film]:
        n=B.SPANS[base]["n"]; l=P.luma(base,1,n); valid=B.picture_mask(base)>0
        cs={name:P.chroma(*fn(film),) if False else P.chroma(fn(film)[0],base,1,n,fn(film)[1]) for name,fn in steps}
        for i in B.good_frames(base):
            r=B.ref_cached(base,i)
            for name in cs:
                e,_=B.frame_errors(P.yuv(l,cs[name],i),r,valid); R[name].append([e["edge"],e["halo"],e["flat"]])
    base0=np.array(R["main"]); out[film]={}
    prev=base0
    for name,_ in steps:
        a=np.array(R[name]); d=a/prev-1
        rng=np.random.default_rng(1); bs=np.array([d[rng.integers(0,len(d),len(d))].mean(0) for _ in range(3000)])
        out[film][name]=dict(mean=a.mean(0).tolist(), vs_prev=d.mean(0).tolist(), ci=np.percentile(bs,[2.5,97.5],axis=0).tolist(), vs_main=(a/base0-1).mean(0).tolist())
        print(film,"%-14s edge %.4f halo %.4f flat %.4f | vs prev %+.3f %+.3f %+.3f | vs main %+.3f %+.3f %+.3f"%(name,*a.mean(0),*d.mean(0),*(a/base0-1).mean(0)),flush=True)
        prev=a
json.dump(out,open("../chain_eval.json","w"),indent=1)
